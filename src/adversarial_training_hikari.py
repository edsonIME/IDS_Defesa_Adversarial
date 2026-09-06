import argparse
import ctypes
import fcntl
import gc
import hashlib
import json
import mmap
import os
import random
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# These variables must be defined before importing TensorFlow.
# Otherwise, the runtime may be initialized without honoring the allocator and
# the gradual memory-growth settings configured for the experiment.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ---------------------------------------------------------------------
# Additional memory and reproducibility safeguards
# ---------------------------------------------------------------------
# The TensorFlow environment variables were configured before the import.


# ---------------------------------------------------------------------
# HIKARI mask of feature constraints
# ---------------------------------------------------------------------
HIKARI_RESPONSE_COLUMNS = ["traffic_category", "Label"]

HIKARI_METADATA_COLUMNS = [
    "uid",
    "originh",
    "originp",
    "responh",
    "responp",
]

HIKARI_EXPECTED_FEATURES = 79

HIKARI_ALLOWED_FEATURES_STRICT = [
    "flow_duration",
    "fwd_pkts_per_sec",
    "bwd_pkts_per_sec",
    "flow_pkts_per_sec",

    "fwd_iat.min",
    "fwd_iat.max",
    "fwd_iat.tot",
    "fwd_iat.avg",
    "fwd_iat.std",

    "bwd_iat.min",
    "bwd_iat.max",
    "bwd_iat.tot",
    "bwd_iat.avg",
    "bwd_iat.std",

    "flow_iat.min",
    "flow_iat.max",
    "flow_iat.tot",
    "flow_iat.avg",
    "flow_iat.std",

    "payload_bytes_per_second",

    "active.min",
    "active.max",
    "active.tot",
    "active.avg",
    "active.std",

    "idle.min",
    "idle.max",
    "idle.tot",
    "idle.avg",
    "idle.std",
]

HIKARI_ALLOWED_FEATURES_BY_MODE = {
    "strict": HIKARI_ALLOWED_FEATURES_STRICT,
}


def infer_hikari_feature_target_columns(
    df: pd.DataFrame,
    feature_start_col: int = 7,
    feature_end_offset: int = 2,
    label_col: int = -1,
    legacy_feature_slice: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Infer the 79 HIKARI numerical traffic features and target column.

    Supported layouts:
      1) complete HIKARI: metadata + 79 features + traffic_category + Label;
      2) filtered HIKARI: 79 features + traffic_category + Label;
      3) legacy notebook slice df.iloc[:, 7:-2], when explicitly enabled.

    traffic_category and Label are response variables and are never used as
    perturbable model features.
    """
    if "Label" in df.columns:
        y = df["Label"].copy()
    else:
        y = df.iloc[:, label_col].copy()

    if not pd.api.types.is_numeric_dtype(y):
        y = y.map({
            "0": 0, "1": 1, 0: 0, 1: 1,
            "benign": 0, "Benign": 0, "BENIGN": 0,
            "background": 0, "Background": 0,
            "malicious": 1, "Malicious": 1, "attack": 1, "Attack": 1,
            False: 0, True: 1,
        })
    y = pd.to_numeric(y, errors="coerce")

    candidate_column_groups: List[List[str]] = []

    if legacy_feature_slice:
        end = -feature_end_offset if feature_end_offset > 0 else None
        candidate_column_groups.append(list(df.columns[feature_start_col:end]))

    if set(HIKARI_RESPONSE_COLUMNS).issubset(set(df.columns)):
        candidate_column_groups.append(
            [
                col for col in df.columns
                if col not in HIKARI_RESPONSE_COLUMNS
                and col not in HIKARI_METADATA_COLUMNS
            ]
        )

    # Complete HIKARI layout: 5 metadata columns + 79 features + 2 response columns.
    if df.shape[1] >= 86:
        candidate_column_groups.append(list(df.columns[5:-2]))

    # Original notebook layout may include two extra index/metadata columns.
    if df.shape[1] >= 88:
        candidate_column_groups.append(list(df.columns[7:-2]))

    # Already filtered: 79 features + traffic_category + Label.
    if df.shape[1] >= 81:
        candidate_column_groups.append(list(df.columns[:-2]))

    # Fallback to all columns except the two final response columns.
    candidate_column_groups.append(list(df.columns[:-2]))

    required_allowed = set(HIKARI_ALLOWED_FEATURES_STRICT)

    for feature_cols in candidate_column_groups:
        feature_cols = list(feature_cols)
        feature_set = set(feature_cols)
        has_no_response_leakage = not any(
            col in feature_set for col in HIKARI_RESPONSE_COLUMNS
        )
        has_no_metadata_leakage = not any(
            col in feature_set for col in HIKARI_METADATA_COLUMNS
        )

        if (
            len(feature_cols) == HIKARI_EXPECTED_FEATURES
            and has_no_response_leakage
            and has_no_metadata_leakage
            and required_allowed.issubset(feature_set)
        ):
            X = df.loc[:, feature_cols].copy()
            X = X.apply(pd.to_numeric, errors="coerce")
            valid_rows = X.notna().all(axis=1) & y.notna()
            X = X.loc[valid_rows].copy()
            y_clean = y.loc[valid_rows].astype("int32").copy()
            return X, y_clean, feature_cols

    raise ValueError(
        "Could not infer the 79 numerical HIKARI features. "
        "Expected: a complete dataframe with metadata + 79 features + "
        "traffic_category + Label, or a filtered dataframe with 79 features + "
        "traffic_category + Label. To try the legacy slice, use "
        "--legacy-feature-slice."
    )


def build_hikari_feature_mask(
    feature_cols: Iterable[str],
    mask_mode: str = "strict",
) -> Tuple[np.ndarray, pd.Series]:
    """Builds the HIKARI mask of feature constraints.

    Mask convention:
      - 1.0: feature may be perturbed by FGSM/PGD.
      - 0.0: feature remains fixed.

    The mask follows a deny-by-default policy. Only the strict temporal/rate
    subset is enabled.
    """
    feature_cols = list(feature_cols)
    mask_mode = str(mask_mode).lower().strip()

    if mask_mode not in HIKARI_ALLOWED_FEATURES_BY_MODE:
        raise ValueError(
            f"Unknown mask_mode={mask_mode!r}. "
            f"Use one of: {sorted(HIKARI_ALLOWED_FEATURES_BY_MODE)}"
        )

    leaked_metadata = [c for c in HIKARI_METADATA_COLUMNS if c in feature_cols]
    leaked_targets = [c for c in HIKARI_RESPONSE_COLUMNS if c in feature_cols]

    if leaked_metadata:
        raise ValueError(f"Metadata/identifier columns found in model features: {leaked_metadata}")

    if leaked_targets:
        raise ValueError(f"Response columns found in model features: {leaked_targets}")

    if len(feature_cols) != HIKARI_EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {HIKARI_EXPECTED_FEATURES} HIKARI traffic features, "
            f"but received {len(feature_cols)}."
        )

    allowed_features = HIKARI_ALLOWED_FEATURES_BY_MODE[mask_mode]
    missing_allowed = [f for f in allowed_features if f not in feature_cols]

    if missing_allowed:
        raise ValueError(
            "The following perturbable HIKARI features were not found in X: "
            f"{missing_allowed}"
        )

    mask_series = pd.Series(0.0, index=feature_cols, dtype="float32")
    mask_series.loc[allowed_features] = 1.0
    return mask_series.values.astype("float32"), mask_series


def print_hikari_mask_report(mask_series: pd.Series, mask_mode: str) -> None:
    """Prints the active HIKARI mask of feature constraints."""
    allowed = mask_series[mask_series == 1.0]
    blocked = mask_series[mask_series == 0.0]

    print("\n" + "-" * 78)
    print(f"[HIKARI mask of feature constraints] mode={mask_mode}")
    print(f"Total model features: {len(mask_series)}")
    print(f"Perturbable features: {len(allowed)}")
    print(f"Blocked features: {len(blocked)}")
    print("\nPerturbable features:")
    for name in allowed.index:
        print(f"  - {name}")
    print("\nBlocked features:")
    for name in blocked.index:
        print(f"  - {name}")
    print("-" * 78)


def reshape_mask_to_2d(feature_mask_1d: np.ndarray, size: int) -> np.ndarray:
    """Pads and reshapes a 1D feature mask to the Conv2D input layout."""
    return reshape_to_2d(np.array([feature_mask_1d], dtype="float32"), size)[0:1]


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def set_seed(seed: int) -> None:
    """Sets Python, NumPy, TensorFlow, and Keras random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        tf.random.set_seed(seed)



# ---------------------------------------------------------------------
# Checkpoint and atomic-file utilities
# ---------------------------------------------------------------------
CHECKPOINT_FORMAT_VERSION = 5
DATA_CACHE_FORMAT_VERSION = 1
DATA_PIPELINE_PROTOCOL_VERSION = "HIKARI_MEMMAP_BATCH_STREAMING_V1"
MODEL_PROTOCOL_VERSION = "M1_3CONV_2ECA_TRANSFORMER_4TOKENS_DROPOUT64_HIKARI_MASKED_CIRA_ALIGNED_V6"


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Writes JSON atomically to avoid leaving a truncated state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def atomic_write_dataframe_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """Writes a CSV atomically to keep checkpoint metadata crash-consistent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def atomic_write_text(path: Path, content: str) -> None:
    """Writes a small text marker atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, path)


def atomic_save_model_weights(model: Model, path: Path) -> None:
    """Saves Keras weights atomically using a valid .weights.h5 suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.stem + ".tmp.weights.h5")
    model.save_weights(str(temporary_path))
    os.replace(temporary_path, path)


def stable_json_sha256(payload: Any) -> str:
    """Returns a stable SHA-256 digest for a JSON-serializable payload."""
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_dataset_identity(csv_path: str) -> Dict[str, Any]:
    """Builds a lightweight identity for the dataset file used by a run.

    The resolved path, file size, and nanosecond modification timestamp prevent
    accidental reuse of checkpoints with a different dataset without forcing
    every parallel worker to hash the complete CSV.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    stat = path.stat()
    return {
        "resolved_path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def get_run_checkpoint_dir(
    args: argparse.Namespace,
    run_id: int,
    seed: int,
) -> Path:
    """Returns a checkpoint directory that is unique for attack, run, and seed."""
    checkpoint_root = (
        Path(args.checkpoint_root)
        if args.checkpoint_root is not None
        else Path(args.output_dir) / "checkpoints"
    )
    return (
        checkpoint_root
        / args.train_attack.lower()
        / f"run_{run_id:02d}_seed_{seed}"
    )


def get_best_weights_path(best_weights_dir: Path, best_epoch: int) -> Path:
    """Returns the versioned best-model path referenced by a checkpoint."""
    if best_epoch <= 0:
        raise ValueError(
            f"best_epoch must be positive to resolve best weights; got {best_epoch}."
        )
    return best_weights_dir / f"best_epoch_{best_epoch:04d}.weights.h5"


def prune_unreferenced_best_weights(
    best_weights_dir: Path,
    confirmed_best_epoch: int,
) -> None:
    """Removes orphaned best-model files not referenced by the confirmed state."""
    if not best_weights_dir.exists() or confirmed_best_epoch <= 0:
        return
    confirmed_path = get_best_weights_path(
        best_weights_dir,
        confirmed_best_epoch,
    )
    for candidate in best_weights_dir.glob("best_epoch_*.weights.h5"):
        if candidate != confirmed_path:
            try:
                candidate.unlink()
            except OSError:
                pass


def derive_epoch_seed(base_seed: int, epoch: int) -> int:
    """Derives a stable positive TensorFlow-compatible seed for one epoch."""
    modulus = 2_147_483_647
    derived = (
        (int(base_seed) * 1_000_003)
        + (int(epoch) * 97_409)
        + 17
    ) % modulus
    return int(derived or 1)


def derive_stateless_attack_seed(
    base_seed: int,
    epoch: int,
    batch_index: int,
) -> Tuple[int, int]:
    """Derives a stateless two-integer seed for a PGD training batch."""
    modulus = 2_147_483_647
    first = derive_epoch_seed(base_seed, epoch)
    second = (
        (int(base_seed) * 65_537)
        + (int(epoch) * 8_191)
        + (int(batch_index) * 131_071)
        + 29
    ) % modulus
    return int(first), int(second or 1)



def build_checkpoint_signature(
    args: argparse.Namespace,
    run_id: int,
    seed: int,
    input_size: int,
    feature_columns: Iterable[str],
    feature_mask_1d: np.ndarray,
    common_cache_signature_sha256: str,
    prepared_cache_signature_sha256: str,
) -> Dict[str, Any]:
    """Builds the immutable configuration used to validate a resumed run."""
    feature_columns_list = [str(column) for column in feature_columns]
    mask_values = [
        int(value)
        for value in np.asarray(feature_mask_1d, dtype=np.float32).reshape(-1)
    ]
    dataset_identity = build_dataset_identity(args.csv)

    signature = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_protocol_version": MODEL_PROTOCOL_VERSION,
        "data_pipeline_protocol_version": DATA_PIPELINE_PROTOCOL_VERSION,
        "common_cache_signature_sha256": str(common_cache_signature_sha256),
        "prepared_cache_signature_sha256": str(prepared_cache_signature_sha256),
        "run_id": int(run_id),
        "seed": int(seed),
        "dataset": dataset_identity,
        "feature_columns": feature_columns_list,
        "feature_columns_sha256": stable_json_sha256(feature_columns_list),
        "feature_mask": mask_values,
        "feature_mask_sha256": hashlib.sha256(
            np.asarray(mask_values, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "feature_start_col": int(args.feature_start_col),
        "feature_end_offset": int(args.feature_end_offset),
        "label_col": int(args.label_col),
        "legacy_feature_slice": bool(args.legacy_feature_slice),
        "train_attack": str(args.train_attack).lower(),
        "train_epsilon": float(args.train_epsilon),
        "train_alpha": float(args.train_alpha),
        "train_pgd_steps": int(args.train_pgd_steps),
        "adv_ratio": float(args.adv_ratio),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "use_smote": bool(not args.no_smote),
        "mask_mode": str(args.mask_mode),
        "mask_perturbable_features": int(np.sum(feature_mask_1d)),
        "mask_blocked_features": int(
            len(feature_mask_1d) - np.sum(feature_mask_1d)
        ),
        "input_size": int(input_size),
        "eval_epsilons": [float(value) for value in args.eval_epsilons],
        "eval_pgd_steps": int(args.eval_pgd_steps),
        "adv_batch_size": int(args.adv_batch_size),
        "threshold_mode": str(args.threshold_mode),
        "deterministic_epoch_shuffle": True,
        "stateless_pgd_random_start_during_training": True,
        "full_training_tensor_materialized": False,
        "training_storage": "read_only_numpy_memmap",
        "index_prefetch_batches": int(args.index_prefetch_batches),
    }
    signature["signature_sha256"] = stable_json_sha256(signature)
    return signature



def load_checkpoint_history(
    history_path: Path,
    confirmed_epoch: int,
) -> List[Dict[str, Any]]:
    """Loads only history rows confirmed by the latest TensorFlow checkpoint."""
    if not history_path.exists():
        return []

    history_df = pd.read_csv(history_path)
    if "epoch" not in history_df.columns:
        raise ValueError(
            f"Checkpoint history does not contain an 'epoch' column: {history_path}"
        )

    history_df = history_df[
        pd.to_numeric(history_df["epoch"], errors="coerce") <= confirmed_epoch
    ].copy()
    history_df.sort_values("epoch", inplace=True)
    history_df.drop_duplicates(subset=["epoch"], keep="last", inplace=True)
    return history_df.to_dict("records")


def build_optimizer_slots(model: Model) -> None:
    """Creates optimizer slot variables before checkpoint restoration when supported."""
    optimizer = model.optimizer
    build_method = getattr(optimizer, "build", None)
    if callable(build_method):
        try:
            build_method(model.trainable_variables)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Deferred TensorFlow restoration remains available when a particular
            # optimizer implementation does not expose a compatible build method.
            pass




# ---------------------------------------------------------------------
# Shared memory-mapped data cache
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CommonDataCache:
    root: Path
    x_path: Path
    y_path: Path
    manifest_path: Path
    feature_columns: Tuple[str, ...]
    shape: Tuple[int, int]
    x_dtype: str
    y_dtype: str
    label_distribution: Dict[int, int]
    signature_sha256: str


@dataclass(frozen=True)
class PreparedRunCache:
    root: Path
    x_train_path: Path
    y_train_path: Path
    x_val_path: Path
    y_val_path: Path
    x_test_path: Path
    y_test_path: Path
    manifest_path: Path
    input_size: int
    n_train: int
    n_val: int
    n_test: int
    prep_timing: Dict[str, float]
    signature_sha256: str


def get_data_cache_root(args: argparse.Namespace) -> Path:
    """Returns the shared cache root used by every concurrent worker."""
    if args.data_cache_root is not None:
        return Path(args.data_cache_root).expanduser().resolve()
    return Path(args.csv).expanduser().resolve().parent / ".hikari_ram_cache"


def trim_process_heap() -> None:
    """Asks glibc to return unused heap arenas to the operating system."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (AttributeError, OSError):
        pass


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Writes an uncompressed NumPy array atomically for later mmap access."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary_path, array, allow_pickle=False)
    os.replace(temporary_path, path)


def _remove_stale_temporary_files(directory: Path) -> None:
    if not directory.exists():
        return
    for candidate in directory.glob("*.tmp.npy"):
        try:
            candidate.unlink()
        except OSError:
            pass
    for candidate in directory.glob("*.tmp"):
        try:
            candidate.unlink()
        except OSError:
            pass


@contextmanager
def exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """Acquires a Linux advisory lock shared by independent Python workers."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def preprocessing_slot(cache_root: Path, parallelism: int) -> Iterator[int]:
    """Bounds concurrent split/SMOTE/scaling operations across processes."""
    if parallelism <= 0:
        raise ValueError("parallelism must be greater than zero.")

    slot_dir = cache_root / "locks" / "preprocessing_slots"
    slot_dir.mkdir(parents=True, exist_ok=True)
    acquired_file = None
    acquired_slot = -1
    last_message = 0.0

    while acquired_file is None:
        for slot in range(parallelism):
            candidate = (slot_dir / f"slot_{slot:02d}.lock").open(
                "a+",
                encoding="utf-8",
            )
            try:
                fcntl.flock(
                    candidate.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                acquired_file = candidate
                acquired_slot = slot
                break
            except BlockingIOError:
                candidate.close()

        if acquired_file is None:
            now = time.monotonic()
            if now - last_message >= 30.0:
                print(
                    "[Data cache] Waiting for an available preprocessing "
                    f"slot (limit={parallelism})."
                )
                last_message = now
            time.sleep(2.0)

    try:
        yield acquired_slot
    finally:
        fcntl.flock(acquired_file.fileno(), fcntl.LOCK_UN)
        acquired_file.close()


def _advise_memmap(array: np.ndarray, advice: str) -> None:
    memory_map = getattr(array, "_mmap", None)
    if memory_map is None or not hasattr(memory_map, "madvise"):
        return
    advice_map = {
        "normal": getattr(mmap, "MADV_NORMAL", None),
        "random": getattr(mmap, "MADV_RANDOM", None),
        "sequential": getattr(mmap, "MADV_SEQUENTIAL", None),
    }
    selected = advice_map.get(advice)
    if selected is not None:
        try:
            memory_map.madvise(selected)
        except (OSError, ValueError):
            pass


def open_memmap_array(path: Path, advice: str = "normal") -> np.ndarray:
    """Opens a read-only .npy array without loading its full content into RAM."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    _advise_memmap(array, advice)
    return array


def close_memmap_array(array: np.ndarray) -> None:
    """Closes the mmap backing an array when one is present."""
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        try:
            memory_map.close()
        except OSError:
            pass


def release_memmap_pages(array: np.ndarray) -> None:
    """Requests that resident file-backed pages may be discarded."""
    memory_map = getattr(array, "_mmap", None)
    advice = getattr(mmap, "MADV_DONTNEED", None)
    if memory_map is not None and advice is not None and hasattr(memory_map, "madvise"):
        try:
            memory_map.madvise(advice)
        except (OSError, ValueError):
            pass


def make_epoch_index_dataset(
    n_samples: int,
    batch_size: int,
    shuffle_buffer_size: int,
    seed: int,
    prefetch_batches: int,
) -> tf.data.Dataset:
    """Creates a bounded TensorFlow pipeline containing integer indices only."""
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero.")
    with tf.device("/CPU:0"):
        dataset = tf.data.Dataset.range(n_samples)
        dataset = dataset.shuffle(
            buffer_size=min(int(shuffle_buffer_size), n_samples),
            seed=int(seed),
            reshuffle_each_iteration=False,
        )
        dataset = dataset.batch(batch_size, drop_remainder=False)
        if prefetch_batches > 0:
            dataset = dataset.prefetch(prefetch_batches)
    return dataset


def _common_cache_key_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "cache_format_version": DATA_CACHE_FORMAT_VERSION,
        "data_pipeline_protocol_version": DATA_PIPELINE_PROTOCOL_VERSION,
        "dataset": build_dataset_identity(args.csv),
        "feature_start_col": int(args.feature_start_col),
        "feature_end_offset": int(args.feature_end_offset),
        "label_col": int(args.label_col),
        "legacy_feature_slice": bool(args.legacy_feature_slice),
        "expected_features": HIKARI_EXPECTED_FEATURES,
        "cleaning": ["replace_inf_with_nan", "dropna", "drop_duplicates"],
    }


def _load_common_cache(directory: Path, expected_signature: str) -> Optional[CommonDataCache]:
    manifest_path = directory / "manifest.json"
    x_path = directory / "X_raw.npy"
    y_path = directory / "y_raw.npy"
    if not (manifest_path.exists() and x_path.exists() and y_path.exists()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature_sha256") != expected_signature:
            return None
        X_raw = np.load(x_path, mmap_mode="r", allow_pickle=False)
        y_raw = np.load(y_path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(manifest["shape"])
        if tuple(X_raw.shape) != expected_shape:
            return None
        if len(y_raw) != expected_shape[0]:
            return None
        close_memmap_array(X_raw)
        close_memmap_array(y_raw)
        return CommonDataCache(
            root=directory,
            x_path=x_path,
            y_path=y_path,
            manifest_path=manifest_path,
            feature_columns=tuple(manifest["feature_columns"]),
            shape=(int(expected_shape[0]), int(expected_shape[1])),
            x_dtype=str(manifest["x_dtype"]),
            y_dtype=str(manifest["y_dtype"]),
            label_distribution={
                int(key): int(value)
                for key, value in manifest["label_distribution"].items()
            },
            signature_sha256=str(manifest["signature_sha256"]),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def ensure_common_data_cache(args: argparse.Namespace) -> CommonDataCache:
    """Creates the cleaned HIKARI matrix once and reuses it across workers."""
    cache_root = get_data_cache_root(args)
    cache_root.mkdir(parents=True, exist_ok=True)
    key_payload = _common_cache_key_payload(args)
    signature = stable_json_sha256(key_payload)
    directory = cache_root / "common" / signature
    lock_path = cache_root / "locks" / f"common_{signature}.lock"

    existing = _load_common_cache(directory, signature)
    if existing is not None:
        print(f"[Data cache] Reusing common cache: {directory}")
        return existing

    with exclusive_file_lock(lock_path):
        existing = _load_common_cache(directory, signature)
        if existing is not None:
            print(f"[Data cache] Reusing common cache: {directory}")
            return existing

        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _remove_stale_temporary_files(directory)

        print("[Data cache] Building the shared cleaned HIKARI cache once.")
        df = load_and_clean_csv(args.csv)
        X, y, feature_cols = select_features_and_label(
            df,
            feature_start_col=args.feature_start_col,
            feature_end_offset=args.feature_end_offset,
            label_col=args.label_col,
            legacy_feature_slice=args.legacy_feature_slice,
        )
        del df
        gc.collect()

        X_array = X.to_numpy(copy=True)
        if X_array.dtype.kind not in "fiu":
            raise TypeError(
                f"The selected feature matrix has unsupported dtype {X_array.dtype}."
            )
        y_array = np.asarray(y, dtype=np.int32)
        x_path = directory / "X_raw.npy"
        y_path = directory / "y_raw.npy"
        atomic_save_npy(x_path, X_array)
        atomic_save_npy(y_path, y_array)

        label_distribution = {
            int(key): int(value)
            for key, value in pd.Series(y_array).value_counts().sort_index().items()
        }
        manifest = {
            "cache_format_version": DATA_CACHE_FORMAT_VERSION,
            "data_pipeline_protocol_version": DATA_PIPELINE_PROTOCOL_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "signature_payload": key_payload,
            "signature_sha256": signature,
            "feature_columns": list(feature_cols),
            "shape": [int(X_array.shape[0]), int(X_array.shape[1])],
            "x_dtype": str(X_array.dtype),
            "y_dtype": str(y_array.dtype),
            "label_distribution": label_distribution,
            "x_path": str(x_path),
            "y_path": str(y_path),
        }
        atomic_write_json(directory / "manifest.json", manifest)

        del X, y, X_array, y_array
        gc.collect()
        trim_process_heap()

    loaded = _load_common_cache(directory, signature)
    if loaded is None:
        raise RuntimeError(f"Failed to validate the common data cache: {directory}")
    return loaded


def build_prepared_cache_signature(
    common_cache: CommonDataCache,
    seed: int,
    use_smote: bool,
    chunk_size: int,
) -> Dict[str, Any]:
    payload = {
        "cache_format_version": DATA_CACHE_FORMAT_VERSION,
        "data_pipeline_protocol_version": DATA_PIPELINE_PROTOCOL_VERSION,
        "common_cache_signature_sha256": common_cache.signature_sha256,
        "seed": int(seed),
        "test_size_total": 0.30,
        "validation_fraction_of_temporary_split": 0.50,
        "use_smote": bool(use_smote),
        "smote_random_state": int(seed),
        "scaler": "MinMaxScaler_default",
        "output_dtype": "float32",
        "reshape": "square_zero_padding_then_NHWC",
        "preprocess_chunk_size": int(chunk_size),
    }
    payload["signature_sha256"] = stable_json_sha256(payload)
    return payload


def _prepared_cache_directory(
    args: argparse.Namespace,
    common_cache: CommonDataCache,
    signature_sha256: str,
) -> Path:
    return (
        get_data_cache_root(args)
        / "prepared"
        / common_cache.signature_sha256
        / signature_sha256
    )


def _load_prepared_cache(
    directory: Path,
    expected_signature: str,
) -> Optional[PreparedRunCache]:
    manifest_path = directory / "manifest.json"
    paths = {
        "x_train": directory / "X_train.npy",
        "y_train": directory / "y_train.npy",
        "x_val": directory / "X_val.npy",
        "y_val": directory / "y_val.npy",
        "x_test": directory / "X_test.npy",
        "y_test": directory / "y_test.npy",
    }
    if not manifest_path.exists() or not all(path.exists() for path in paths.values()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature_sha256") != expected_signature:
            return None
        arrays = {
            name: np.load(path, mmap_mode="r", allow_pickle=False)
            for name, path in paths.items()
        }
        valid_shapes = (
            arrays["x_train"].shape[0] == arrays["y_train"].shape[0]
            and arrays["x_val"].shape[0] == arrays["y_val"].shape[0]
            and arrays["x_test"].shape[0] == arrays["y_test"].shape[0]
        )
        for array in arrays.values():
            close_memmap_array(array)
        if not valid_shapes:
            return None
        return PreparedRunCache(
            root=directory,
            x_train_path=paths["x_train"],
            y_train_path=paths["y_train"],
            x_val_path=paths["x_val"],
            y_val_path=paths["y_val"],
            x_test_path=paths["x_test"],
            y_test_path=paths["y_test"],
            manifest_path=manifest_path,
            input_size=int(manifest["input_size"]),
            n_train=int(manifest["n_train"]),
            n_val=int(manifest["n_val"]),
            n_test=int(manifest["n_test"]),
            prep_timing={
                str(key): float(value)
                for key, value in manifest["prep_timing"].items()
            },
            signature_sha256=str(manifest["signature_sha256"]),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _create_scaled_2d_memmap(
    destination: Path,
    source: np.ndarray,
    scaler: MinMaxScaler,
    input_size: int,
    n_features: int,
    chunk_size: int,
    source_indices: Optional[np.ndarray] = None,
) -> None:
    """Transforms rows in chunks and writes the final NHWC float32 .npy file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.stem + ".tmp.npy")
    n_rows = len(source_indices) if source_indices is not None else len(source)
    output = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, input_size, input_size, 1),
    )
    output_flat = output.reshape(n_rows, input_size * input_size)

    try:
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            if source_indices is None:
                raw_chunk = np.asarray(source[start:end])
            else:
                raw_chunk = np.asarray(source[source_indices[start:end]])
            scaled_chunk = scaler.transform(raw_chunk)
            destination_chunk = output_flat[start:end]
            destination_chunk.fill(0.0)
            destination_chunk[:, :n_features] = np.asarray(
                scaled_chunk,
                dtype=np.float32,
            )
            del raw_chunk, scaled_chunk, destination_chunk
        output.flush()
    finally:
        del output_flat, output
    os.replace(temporary_path, destination)


def _build_prepared_run_cache(
    common_cache: CommonDataCache,
    directory: Path,
    signature: Dict[str, Any],
    seed: int,
    use_smote: bool,
    chunk_size: int,
) -> None:
    """Builds one seed-specific split while bounding temporary RAM usage."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _remove_stale_temporary_files(directory)

    X_raw = open_memmap_array(common_cache.x_path, advice="random")
    y_raw = open_memmap_array(common_cache.y_path, advice="random")
    n_samples, n_features = X_raw.shape

    split_start = time.perf_counter()
    all_indices = np.arange(n_samples, dtype=np.int64)
    train_indices, temporary_indices = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=seed,
        stratify=np.asarray(y_raw),
    )
    val_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=seed,
        stratify=np.asarray(y_raw[temporary_indices]),
    )
    X_train_raw = np.asarray(X_raw[train_indices])
    y_train_raw = np.asarray(y_raw[train_indices], dtype=np.int32)
    split_time = time.perf_counter() - split_start
    del all_indices, temporary_indices, train_indices
    gc.collect()

    smote_start = time.perf_counter()
    if use_smote:
        smote = SMOTE(random_state=seed)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_raw,
            y_train_raw,
        )
    else:
        X_train_resampled = X_train_raw
        y_train_resampled = y_train_raw
    smote_time = time.perf_counter() - smote_start
    if X_train_resampled is not X_train_raw:
        del X_train_raw, y_train_raw
        gc.collect()
        trim_process_heap()

    scale_start = time.perf_counter()
    scaler = MinMaxScaler()
    scaler.fit(X_train_resampled)
    input_size = int(np.ceil(np.sqrt(n_features)))

    _create_scaled_2d_memmap(
        destination=directory / "X_train.npy",
        source=X_train_resampled,
        scaler=scaler,
        input_size=input_size,
        n_features=n_features,
        chunk_size=chunk_size,
    )
    _create_scaled_2d_memmap(
        destination=directory / "X_val.npy",
        source=X_raw,
        source_indices=val_indices,
        scaler=scaler,
        input_size=input_size,
        n_features=n_features,
        chunk_size=chunk_size,
    )
    _create_scaled_2d_memmap(
        destination=directory / "X_test.npy",
        source=X_raw,
        source_indices=test_indices,
        scaler=scaler,
        input_size=input_size,
        n_features=n_features,
        chunk_size=chunk_size,
    )

    y_train_final = np.asarray(y_train_resampled, dtype=np.float32)
    y_val_final = np.asarray(y_raw[val_indices], dtype=np.float32)
    y_test_final = np.asarray(y_raw[test_indices], dtype=np.float32)
    atomic_save_npy(directory / "y_train.npy", y_train_final)
    atomic_save_npy(directory / "y_val.npy", y_val_final)
    atomic_save_npy(directory / "y_test.npy", y_test_final)
    scale_time = time.perf_counter() - scale_start

    prep_timing = {
        "split_time_s": float(split_time),
        "smote_time_s": float(smote_time),
        "scale_reshape_time_s": float(scale_time),
        "n_train": int(len(y_train_final)),
        "n_val": int(len(y_val_final)),
        "n_test": int(len(y_test_final)),
        "n_features": int(n_features),
        "image_size": int(input_size),
    }
    manifest = {
        "cache_format_version": DATA_CACHE_FORMAT_VERSION,
        "data_pipeline_protocol_version": DATA_PIPELINE_PROTOCOL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "signature": signature,
        "signature_sha256": signature["signature_sha256"],
        "common_cache_signature_sha256": common_cache.signature_sha256,
        "seed": int(seed),
        "use_smote": bool(use_smote),
        "input_size": int(input_size),
        "n_features": int(n_features),
        "n_train": int(len(y_train_final)),
        "n_val": int(len(y_val_final)),
        "n_test": int(len(y_test_final)),
        "prep_timing": prep_timing,
        "storage": "read_only_numpy_memmap",
        "full_training_tensor_materialized": False,
    }
    atomic_write_json(directory / "manifest.json", manifest)

    close_memmap_array(X_raw)
    close_memmap_array(y_raw)
    del (
        X_raw,
        y_raw,
        X_train_resampled,
        y_train_resampled,
        val_indices,
        test_indices,
        y_train_final,
        y_val_final,
        y_test_final,
        scaler,
    )
    gc.collect()
    trim_process_heap()


def ensure_prepared_run_cache(
    common_cache: CommonDataCache,
    seed: int,
    args: argparse.Namespace,
    signature: Optional[Dict[str, Any]] = None,
) -> PreparedRunCache:
    """Returns a seed-specific cache shared by FGSM and PGD workers."""
    signature = signature or build_prepared_cache_signature(
        common_cache=common_cache,
        seed=seed,
        use_smote=not args.no_smote,
        chunk_size=args.preprocess_chunk_size,
    )
    signature_sha256 = signature["signature_sha256"]
    directory = _prepared_cache_directory(
        args,
        common_cache,
        signature_sha256,
    )
    existing = _load_prepared_cache(directory, signature_sha256)
    if existing is not None:
        print(f"[Data cache] Reusing prepared seed cache: {directory}")
        return existing

    cache_root = get_data_cache_root(args)
    cache_lock = cache_root / "locks" / f"prepared_{signature_sha256}.lock"
    with exclusive_file_lock(cache_lock):
        existing = _load_prepared_cache(directory, signature_sha256)
        if existing is not None:
            print(f"[Data cache] Reusing prepared seed cache: {directory}")
            return existing

        with preprocessing_slot(
            cache_root=cache_root,
            parallelism=args.preprocess_parallelism,
        ) as slot:
            print(
                f"[Data cache] Building seed={seed} cache in preprocessing "
                f"slot {slot + 1}/{args.preprocess_parallelism}."
            )
            _build_prepared_run_cache(
                common_cache=common_cache,
                directory=directory,
                signature=signature,
                seed=seed,
                use_smote=not args.no_smote,
                chunk_size=args.preprocess_chunk_size,
            )

    loaded = _load_prepared_cache(directory, signature_sha256)
    if loaded is None:
        raise RuntimeError(f"Failed to validate prepared cache: {directory}")
    return loaded


# ---------------------------------------------------------------------
# M1 model
# ---------------------------------------------------------------------
def build_mtl_model_M1(input_shape: Tuple[int, int, int], learning_rate: float = 1e-4) -> Model:
    """Builds the CIRA-aligned M1 architecture for HIKARI-2021.

    Architecture: Conv2D(32) -> ECA -> Conv2D(64) -> ECA ->
    Conv2D(128) -> Transformer -> Dense(128) + Dropout ->
    Dense(64) + Dropout -> binary output.

    For the 79-feature HIKARI input, the 9x9 spatial representation is reduced
    to 2x2 by the first two convolutional blocks. The third block keeps the 2x2
    grid (no pooling), so the Transformer encoder receives four tokens instead
    of a single degenerate token.
    """
    inputs = layers.Input(shape=input_shape)

    def eca_block(input_tensor):
        """Applies canonical channel-wise Efficient Channel Attention."""
        channels = input_tensor.shape[-1]
        if channels is None:
            raise ValueError(
                "ECA requires a statically known channel dimension."
            )
        channels = int(channels)

        # Adaptive odd kernel size from the ECA-Net formulation.
        gamma = 2.0
        beta = 1.0
        kernel_estimate = int(abs((np.log2(channels) + beta) / gamma))
        k_size = kernel_estimate if kernel_estimate % 2 == 1 else kernel_estimate + 1
        k_size = max(3, k_size)

        squeeze = layers.GlobalAveragePooling2D()(input_tensor)
        squeeze = layers.Reshape((channels, 1))(squeeze)
        squeeze = layers.Conv1D(
            filters=1,
            kernel_size=k_size,
            padding="same",
            use_bias=False,
        )(squeeze)
        squeeze = layers.Activation("sigmoid")(squeeze)
        squeeze = layers.Reshape((1, 1, channels))(squeeze)
        return layers.Multiply()([input_tensor, squeeze])

    def transformer_encoder(x_in, head_size, num_heads, ff_dim, dropout=0.3):
        x = layers.LayerNormalization(epsilon=1e-6)(x_in)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + x_in
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Dense(ff_dim, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(int(x_in.shape[-1]))(x)
        return x + res

    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = eca_block(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = eca_block(x)

    # Convolutional Block 3: keep the spatial grid. No pooling is applied here,
    # so the Transformer receives a non-degenerate token sequence instead of a
    # single token. For the 79-feature HIKARI input, the 9x9 representation is
    # reduced to 2x2 across the first two blocks, yielding four tokens.
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    spatial_height = x.shape[1]
    spatial_width = x.shape[2]
    if spatial_height is None or spatial_width is None:
        raise ValueError(
            "The spatial dimensions before the Transformer must be statically known."
        )
    sequence_length = int(spatial_height) * int(spatial_width)
    if sequence_length < 2:
        raise ValueError(
            "The Transformer requires at least two spatial tokens; "
            f"received {sequence_length}."
        )

    x = layers.Reshape((sequence_length, int(x.shape[-1])))(x)
    x = transformer_encoder(x, head_size=128, num_heads=4, ff_dim=256, dropout=0.3)
    x = layers.Flatten()(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    logits = layers.Dense(1, activation=None, name="logits")(x)
    output = layers.Activation("sigmoid", dtype="float32", name="binary_output")(logits)

    model = Model(inputs=inputs, outputs=output, name="M1_ECA_CNN_Transformer")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", "Precision", "Recall", "AUC"],
    )
    return model



# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    """Loads and lightly cleans the HIKARI-2021 CSV.

    Feature selection and target conversion are handled later by
    select_features_and_label(), so traffic_category is not converted here.
    """
    start = time.perf_counter()
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    elapsed = time.perf_counter() - start
    print(f"[LOAD] Samples after cleaning: {len(df):,} | columns: {df.shape[1]} | time={elapsed:.2f}s")
    return df


def select_features_and_label(
    df: pd.DataFrame,
    feature_start_col: int = 7,
    feature_end_offset: int = 2,
    label_col: int = -1,
    legacy_feature_slice: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    X, y, feature_cols = infer_hikari_feature_target_columns(
        df,
        feature_start_col=feature_start_col,
        feature_end_offset=feature_end_offset,
        label_col=label_col,
        legacy_feature_slice=legacy_feature_slice,
    )
    if X.shape[1] <= 0:
        raise ValueError("No features were selected.")
    if y.nunique() != 2:
        print(f"[Warning] The label has {y.nunique()} classes. This script assumes binary classification.")
    return X, y, feature_cols


def reshape_to_2d(data: np.ndarray, size: int) -> np.ndarray:
    pad_size = size ** 2 - data.shape[1]
    padded = np.pad(data, pad_width=((0, 0), (0, pad_size)), mode="constant")
    return padded.reshape(-1, size, size, 1).astype("float32")









# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def get_best_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_pred_proba).reshape(-1)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_score > t).astype("int32")
        current_f1 = f1_score(y_true_flat, y_pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1, best_t = current_f1, float(t)
    return best_t, float(best_f1)


def get_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true_flat = np.asarray(y_true).reshape(-1).astype("int32")
    y_score = np.asarray(y_pred_proba).reshape(-1)
    y_pred = (y_score > threshold).astype("int32")

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred, labels=labels).ravel()

    try:
        auc = roc_auc_score(y_true_flat, y_score)
    except ValueError:
        auc = np.nan

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "Acc": float(accuracy_score(y_true_flat, y_pred)),
        "BalancedAcc": float(balanced_accuracy_score(y_true_flat, y_pred)),
        "Prec": float(precision_score(y_true_flat, y_pred, zero_division=0)),
        "Rec": float(recall_score(y_true_flat, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true_flat, y_pred, zero_division=0)),
        "AUC": float(auc),
        "MCC": float(matthews_corrcoef(y_true_flat, y_pred)),
        "TPR": float(tpr),
        "TNR": float(tnr),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def calculate_ASR_I(
    y_true: np.ndarray,
    y_clean_proba: np.ndarray,
    y_adv_proba: np.ndarray,
    threshold: float,
) -> float:
    y_true_flat = np.asarray(y_true).reshape(-1).astype("int32")
    clean_pred = (np.asarray(y_clean_proba).reshape(-1) > threshold).astype("int32")
    adv_pred = (np.asarray(y_adv_proba).reshape(-1) > threshold).astype("int32")
    originally_detected_malicious = (y_true_flat == 1) & (clean_pred == 1)
    successfully_evaded = originally_detected_malicious & (adv_pred == 0)
    denominator = int(np.sum(originally_detected_malicious))
    return float(np.sum(successfully_evaded) / denominator) if denominator else 0.0


# ---------------------------------------------------------------------
# Adversarial generation
# ---------------------------------------------------------------------
def _loss_for_attack(model: Model, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    logits_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("logits").output)
    logits = logits_model(x, training=False)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    return loss_fn(y, logits)


@tf.function
def fgsm_step(
    model: Model,
    x: tf.Tensor,
    y: tf.Tensor,
    epsilon: tf.Tensor,
    mask_tensor: tf.Tensor,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = _loss_for_attack(model, x, y)
    gradient = tape.gradient(loss, x)
    masked_gradient = gradient * mask_tensor
    x_adv = x + epsilon * tf.sign(masked_gradient)
    return tf.clip_by_value(x_adv, 0.0, 1.0)


@tf.function
def pgd_step(
    model: Model,
    x_adv: tf.Tensor,
    x_orig: tf.Tensor,
    y: tf.Tensor,
    epsilon: tf.Tensor,
    alpha: tf.Tensor,
    mask_tensor: tf.Tensor,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        loss = _loss_for_attack(model, x_adv, y)
    gradient = tape.gradient(loss, x_adv)
    masked_gradient = gradient * mask_tensor
    x_adv = x_adv + alpha * tf.sign(masked_gradient)
    perturbation = tf.clip_by_value(x_adv - x_orig, -epsilon, epsilon)

    # Enforce the mask after projection for numerical safety.
    perturbation = perturbation * mask_tensor
    return tf.clip_by_value(x_orig + perturbation, 0.0, 1.0)


def generate_adversarial_batch(
    model: Model,
    x: tf.Tensor,
    y: tf.Tensor,
    attack: str,
    epsilon: float,
    alpha: Optional[float] = None,
    steps: int = 10,
    feature_mask_2d: Optional[np.ndarray] = None,
    random_seed: Optional[Tuple[int, int]] = None,
) -> tf.Tensor:
    """Generates one constrained adversarial batch.

    When ``random_seed`` is provided, the PGD random start is stateless and
    reproducible for the same run seed, epoch, and batch index.
    """
    attack = attack.lower()
    eps = tf.constant(epsilon, dtype=tf.float32)

    if feature_mask_2d is None:
        mask_tensor = tf.ones_like(x[:1], dtype=tf.float32)
    else:
        mask_tensor = tf.convert_to_tensor(feature_mask_2d, dtype=tf.float32)

    if attack == "none" or epsilon <= 0:
        return tf.identity(x)

    if attack == "fgsm":
        return fgsm_step(model, x, y, eps, mask_tensor)

    if attack == "pgd":
        if alpha is None:
            alpha = epsilon / 4.0

        if random_seed is None:
            random_delta = tf.random.uniform(
                shape=tf.shape(x),
                minval=-epsilon,
                maxval=epsilon,
                dtype=tf.float32,
            )
        else:
            random_delta = tf.random.stateless_uniform(
                shape=tf.shape(x),
                seed=tf.convert_to_tensor(random_seed, dtype=tf.int32),
                minval=-epsilon,
                maxval=epsilon,
                dtype=tf.float32,
            )

        random_delta = random_delta * mask_tensor
        x_adv = tf.clip_by_value(x + random_delta, 0.0, 1.0)

        a = tf.constant(alpha, dtype=tf.float32)
        for _ in range(int(steps)):
            x_adv = pgd_step(model, x_adv, x, y, eps, a, mask_tensor)

        return x_adv

    raise ValueError(f"Unknown attack: {attack}. Use: none, fgsm, or pgd.")


def predict_array_in_batches_timed(
    model: Model,
    X: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    """Performs batched inference and measures the complete observable time.

    The timed interval includes:
      1) batch conversion/transfer to a tensor;
      2) forward pass with ``training=False``;
      3) conversion of the probabilities to NumPy.

    The final conversion to NumPy forces completion of asynchronous GPU
    operations before the timer is stopped.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    prediction_batches: List[np.ndarray] = []
    total_prediction_time = 0.0

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))

        prediction_start = time.perf_counter()
        xb = tf.convert_to_tensor(
            np.asarray(X[start:end], dtype=np.float32),
            dtype=tf.float32,
        )
        predictions = model(xb, training=False)
        predictions_np = np.asarray(predictions.numpy(), dtype=np.float32)
        total_prediction_time += time.perf_counter() - prediction_start

        prediction_batches.append(predictions_np)
        del xb, predictions, predictions_np

    if not prediction_batches:
        return np.empty((0, 1), dtype=np.float32), 0.0

    return np.concatenate(prediction_batches, axis=0), total_prediction_time


def predict_adversarial_on_the_fly(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    attack: str,
    epsilon: float,
    batch_size: int,
    alpha: Optional[float] = None,
    steps: int = 10,
    feature_mask_2d: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """Generates and evaluates adversarial examples batch by batch.

    Only one adversarial batch remains in memory at a time. Timing is separated
    as follows:

    - generation: batch preparation, attack execution, and materialization of
      the adversarial batch as NumPy, which forces GPU synchronization;
    - inference: conversion of the adversarial batch back to a tensor, forward
      pass, and materialization of the probabilities, also with effective
      synchronization.

    This separation prevents asynchronous generation operations from being
    attributed to inference or producing artificially near-zero measurements.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    y_flat = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    prediction_batches: List[np.ndarray] = []
    total_generation_time = 0.0
    total_prediction_time = 0.0

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))

        # Materializing adv_batch_np within the timed interval forces the GPU to
        # complete the attack before the generation timer is stopped.
        generation_start = time.perf_counter()
        xb = tf.convert_to_tensor(
            np.asarray(X[start:end], dtype=np.float32),
            dtype=tf.float32,
        )
        yb = tf.convert_to_tensor(y_flat[start:end], dtype=tf.float32)
        adv_batch = generate_adversarial_batch(
            model=model,
            x=xb,
            y=yb,
            attack=attack,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            feature_mask_2d=feature_mask_2d,
        )
        adv_batch_np = np.asarray(adv_batch.numpy(), dtype=np.float32)
        total_generation_time += time.perf_counter() - generation_start

        # Inference receives a NumPy array, as in the clean evaluation.
        # Therefore, the clean and adversarial conditions include the same cost
        # categories: transfer, forward pass, and probability retrieval.
        prediction_start = time.perf_counter()
        adv_input = tf.convert_to_tensor(adv_batch_np, dtype=tf.float32)
        predictions = model(adv_input, training=False)
        predictions_np = np.asarray(predictions.numpy(), dtype=np.float32)
        total_prediction_time += time.perf_counter() - prediction_start

        prediction_batches.append(predictions_np)

        del xb, yb, adv_batch, adv_batch_np, adv_input, predictions, predictions_np

    if not prediction_batches:
        return np.empty((0, 1), dtype=np.float32), 0.0, 0.0

    return (
        np.concatenate(prediction_batches, axis=0),
        total_generation_time,
        total_prediction_time,
    )



# ---------------------------------------------------------------------
# Adversarial training
# ---------------------------------------------------------------------

def evaluate_loss(
    model: Model,
    X_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int,
) -> float:
    """Evaluates validation loss sequentially from a memory-mapped split."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss_sum = 0.0
    loss_count = 0
    y_array = np.asarray(y_data, dtype=np.float32).reshape(-1)

    for start in range(0, len(y_array), batch_size):
        end = min(start + batch_size, len(y_array))
        xb_np = np.asarray(X_data[start:end], dtype=np.float32)
        yb_np = np.asarray(y_array[start:end], dtype=np.float32).reshape(-1, 1)
        xb = tf.convert_to_tensor(xb_np, dtype=tf.float32)
        yb = tf.convert_to_tensor(yb_np, dtype=tf.float32)
        pred = model(xb, training=False)
        loss_sum += float(loss_fn(yb, pred).numpy())
        loss_count += 1
        del xb_np, yb_np, xb, yb, pred

    return float(loss_sum / loss_count) if loss_count else np.nan




def adversarial_train(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    shuffle_buffer_size: int,
    index_prefetch_batches: int,
    drop_memmap_cache_each_epoch: bool,
    run_seed: int,
    epochs: int,
    train_attack: str,
    train_epsilon: float,
    train_alpha: Optional[float],
    train_pgd_steps: int,
    adv_ratio: float,
    patience: int,
    feature_mask_2d: Optional[np.ndarray],
    checkpoint: tf.train.Checkpoint,
    checkpoint_manager: tf.train.CheckpointManager,
    checkpoint_state: Dict[str, tf.Variable],
    best_weights_dir: Path,
    history_checkpoint_path: Path,
    initial_history: Optional[List[Dict[str, Any]]] = None,
    min_delta: float = 1e-5,
) -> Tuple[Model, pd.DataFrame, Dict[str, float]]:
    """Performs resumable min-max training directly from read-only memory maps.

    The complete training split is never converted into a persistent TensorFlow
    tensor. TensorFlow receives only a batch-sized NumPy copy at each iteration.
    Epoch ordering is generated by a TensorFlow dataset containing indices only,
    preserving the bounded-buffer shuffle semantics of the previous pipeline.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size must be greater than zero.")
    if index_prefetch_batches < 0:
        raise ValueError("index_prefetch_batches cannot be negative.")

    optimizer = model.optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    start_epoch = int(checkpoint_state["epoch"].numpy())
    best_val = float(checkpoint_state["best_val"].numpy())
    best_epoch = int(checkpoint_state["best_epoch"].numpy())
    wait = int(checkpoint_state["wait"].numpy())
    accumulated_train_time = float(
        checkpoint_state["accumulated_train_time"].numpy()
    )
    training_complete = bool(checkpoint_state["training_complete"].numpy())
    history_rows: List[Dict[str, Any]] = list(initial_history or [])

    if best_epoch > 0:
        prune_unreferenced_best_weights(best_weights_dir, best_epoch)

    if training_complete:
        best_weights_path = get_best_weights_path(best_weights_dir, best_epoch)
        if not best_weights_path.exists():
            raise FileNotFoundError(
                "The checkpoint marks training as complete, but the versioned "
                f"best-model weights are missing: {best_weights_path}"
            )
        model.load_weights(str(best_weights_path))
        print(
            f"[RESUME] Training was already complete at epoch {start_epoch}; "
            f"best epoch={best_epoch}. Proceeding to evaluation."
        )
        timing = {
            "train_time_s": accumulated_train_time,
            "effective_train_time_s": accumulated_train_time,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "epochs_executed": int(len(history_rows)),
            "resumed_from_epoch": int(start_epoch),
        }
        return model, pd.DataFrame(history_rows), timing

    should_execute_epochs = start_epoch < epochs and wait < patience

    if start_epoch > 0:
        print(
            f"[RESUME] Continuing training from epoch {start_epoch + 1}; "
            f"best epoch={best_epoch}, best_val={best_val:.6f}, wait={wait}."
        )

    n_train = int(len(y_train))
    if n_train != int(len(X_train)):
        raise ValueError("X_train and y_train have different numbers of samples.")

    if should_execute_epochs:
        for epoch in range(start_epoch + 1, epochs + 1):
            epoch_seed = derive_epoch_seed(run_seed, epoch)
            set_seed(epoch_seed)

            epoch_index_ds = make_epoch_index_dataset(
                n_samples=n_train,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                seed=epoch_seed,
                prefetch_batches=index_prefetch_batches,
            )

            epoch_start = time.perf_counter()
            loss_sum = 0.0
            loss_count = 0

            for batch_index, index_tensor in enumerate(epoch_index_ds):
                batch_indices = np.asarray(
                    index_tensor.numpy(),
                    dtype=np.int64,
                )
                xb_np = np.asarray(
                    X_train[batch_indices],
                    dtype=np.float32,
                )
                yb_np = np.asarray(
                    y_train[batch_indices],
                    dtype=np.float32,
                ).reshape(-1, 1)
                xb = tf.convert_to_tensor(xb_np, dtype=tf.float32)
                yb = tf.convert_to_tensor(yb_np, dtype=tf.float32)
                x_adv = None

                if (
                    train_attack.lower() == "none"
                    or train_epsilon <= 0
                    or adv_ratio <= 0
                ):
                    x_train_batch, y_train_batch = xb, yb
                else:
                    attack_seed = (
                        derive_stateless_attack_seed(
                            run_seed,
                            epoch,
                            batch_index,
                        )
                        if train_attack.lower() == "pgd"
                        else None
                    )
                    x_adv = generate_adversarial_batch(
                        model=model,
                        x=xb,
                        y=yb,
                        attack=train_attack,
                        epsilon=train_epsilon,
                        alpha=train_alpha,
                        steps=train_pgd_steps,
                        feature_mask_2d=feature_mask_2d,
                        random_seed=attack_seed,
                    )
                    if adv_ratio >= 1.0:
                        x_train_batch, y_train_batch = x_adv, yb
                    else:
                        x_train_batch = tf.concat([xb, x_adv], axis=0)
                        y_train_batch = tf.concat([yb, yb], axis=0)

                with tf.GradientTape() as tape:
                    pred = model(x_train_batch, training=True)
                    if (
                        0.0 < adv_ratio < 1.0
                        and train_attack.lower() != "none"
                    ):
                        clean_pred = pred[: tf.shape(yb)[0]]
                        adv_pred = pred[tf.shape(yb)[0] :]
                        clean_loss = loss_fn(yb, clean_pred)
                        adv_loss = loss_fn(yb, adv_pred)
                        loss = (
                            (1.0 - adv_ratio) * clean_loss
                            + adv_ratio * adv_loss
                        )
                    else:
                        loss = loss_fn(y_train_batch, pred)

                    if model.losses:
                        loss += tf.add_n(model.losses)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                loss_sum += float(loss.numpy())
                loss_count += 1

                del (
                    batch_indices,
                    xb_np,
                    yb_np,
                    xb,
                    yb,
                    x_train_batch,
                    y_train_batch,
                    pred,
                    loss,
                    grads,
                )
                if x_adv is not None:
                    del x_adv

            train_loss = float(loss_sum / loss_count) if loss_count else np.nan
            val_loss = evaluate_loss(
                model=model,
                X_data=X_val,
                y_data=y_val,
                batch_size=batch_size,
            )
            epoch_time = time.perf_counter() - epoch_start
            accumulated_train_time += epoch_time

            row = {
                "epoch": epoch,
                "epoch_seed": epoch_seed,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time_s": epoch_time,
                "train_attack": train_attack,
                "train_epsilon": train_epsilon,
                "train_pgd_steps": train_pgd_steps,
                "adv_ratio": adv_ratio,
            }
            history_rows.append(row)

            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} time={epoch_time:.2f}s"
            )

            improved = val_loss + min_delta < best_val
            if improved:
                best_val = val_loss
                best_epoch = epoch
                wait = 0
                best_weights_path = get_best_weights_path(
                    best_weights_dir,
                    best_epoch,
                )
                atomic_save_model_weights(model, best_weights_path)
                print(
                    f"[Checkpoint] New best model saved at epoch {epoch} "
                    f"with val_loss={best_val:.6f}."
                )
            else:
                wait += 1

            atomic_write_dataframe_csv(
                pd.DataFrame(history_rows),
                history_checkpoint_path,
            )

            checkpoint_state["epoch"].assign(epoch)
            checkpoint_state["best_epoch"].assign(best_epoch)
            checkpoint_state["best_val"].assign(best_val)
            checkpoint_state["wait"].assign(wait)
            checkpoint_state["accumulated_train_time"].assign(
                accumulated_train_time
            )
            checkpoint_state["training_complete"].assign(False)

            saved_checkpoint = checkpoint_manager.save(
                checkpoint_number=epoch
            )
            print(
                f"[Checkpoint] Completed epoch {epoch} saved to "
                f"{saved_checkpoint}."
            )

            prune_unreferenced_best_weights(best_weights_dir, best_epoch)
            del epoch_index_ds
            gc.collect()
            if drop_memmap_cache_each_epoch:
                release_memmap_pages(X_train)
            trim_process_heap()

            if wait >= patience:
                print(
                    f"[EarlyStopping] Stopped at epoch {epoch}; "
                    f"best epoch={best_epoch}."
                )
                break

    final_epoch = int(checkpoint_state["epoch"].numpy())
    if best_epoch <= 0:
        raise RuntimeError(
            "Training finished without a valid best epoch. "
            "No model can be selected for evaluation."
        )

    best_weights_path = get_best_weights_path(best_weights_dir, best_epoch)
    if not best_weights_path.exists():
        raise FileNotFoundError(
            "The checkpoint references a best epoch, but its versioned weights "
            f"were not found: {best_weights_path}"
        )

    model.load_weights(str(best_weights_path))
    checkpoint_state["training_complete"].assign(True)
    checkpoint_manager.save(checkpoint_number=1_000_000 + final_epoch)

    timing = {
        "train_time_s": float(accumulated_train_time),
        "effective_train_time_s": float(accumulated_train_time),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "epochs_executed": int(len(history_rows)),
        "resumed_from_epoch": int(start_epoch),
    }
    return model, pd.DataFrame(history_rows), timing



# ---------------------------------------------------------------------
# Clean/adversarial evaluation
# ---------------------------------------------------------------------

def evaluate_clean_and_adversarial(
    model: Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    eval_epsilons: Iterable[float],
    threshold_mode: str,
    batch_size: int,
    pgd_steps_eval: int,
    feature_mask_2d: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluates the model under clean and adversarial conditions with a low RAM peak."""
    eval_start = time.perf_counter()

    y_val_proba, val_predict_time = predict_array_in_batches_timed(
        model=model,
        X=X_val,
        batch_size=batch_size,
    )

    if threshold_mode == "best_f1":
        threshold, val_best_f1 = get_best_threshold(y_val, y_val_proba)
    else:
        threshold, val_best_f1 = 0.5, np.nan

    y_clean_proba, clean_predict_time = predict_array_in_batches_timed(
        model=model,
        X=X_test,
        batch_size=batch_size,
    )

    rows = []
    clean_metrics = get_metrics(y_test, y_clean_proba, threshold)
    clean_metrics["ASR_I"] = 0.0
    clean_metrics["epsilon"] = 0.0
    clean_metrics["attack"] = "clean"
    clean_metrics["condition"] = "clean"
    clean_metrics["threshold"] = threshold
    clean_metrics["eval_generation_time_s"] = 0.0
    clean_metrics["eval_prediction_time_s"] = clean_predict_time
    rows.append(clean_metrics)

    for attack in ["fgsm", "pgd"]:
        for eps in eval_epsilons:
            y_adv_proba, gen_time, pred_time = predict_adversarial_on_the_fly(
                model=model,
                X=X_test,
                y=y_test,
                attack=attack,
                epsilon=float(eps),
                batch_size=batch_size,
                alpha=float(eps) / 4.0,
                steps=pgd_steps_eval,
                feature_mask_2d=feature_mask_2d,
            )

            metrics = get_metrics(y_test, y_adv_proba, threshold)
            metrics["ASR_I"] = calculate_ASR_I(
                y_test,
                y_clean_proba,
                y_adv_proba,
                threshold,
            )
            metrics["epsilon"] = float(eps)
            metrics["attack"] = attack.upper()
            metrics["condition"] = "adversarial"
            metrics["threshold"] = threshold
            metrics["eval_generation_time_s"] = gen_time
            metrics["eval_prediction_time_s"] = pred_time
            rows.append(metrics)

            print(
                f"[Eval] {attack.upper()} eps={float(eps):.4g} "
                f"F1={metrics['F1']:.4f} AUC={metrics['AUC']:.4f} "
                f"ASR_I={metrics['ASR_I']:.4f} "
                f"gen={gen_time:.2f}s pred={pred_time:.2f}s"
            )

            del y_adv_proba
            gc.collect()

    timing = {
        "val_predict_time_s": float(val_predict_time),
        "clean_predict_time_s": float(clean_predict_time),
        "eval_total_time_s": float(time.perf_counter() - eval_start),
        "threshold": float(threshold),
        "val_best_f1_at_threshold": float(val_best_f1),
    }

    del y_val_proba, y_clean_proba
    gc.collect()

    return pd.DataFrame(rows), timing



# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
def mean_std_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Acc", "BalancedAcc", "Prec", "Rec", "F1", "AUC", "MCC",
        "TPR", "TNR", "FPR", "FNR", "ASR_I",
        "eval_generation_time_s", "eval_prediction_time_s",
        "train_time_s", "total_run_time_s",
    ]
    group_cols = ["train_attack", "train_epsilon", "attack", "epsilon", "condition"]

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        base["n_runs"] = int(g["run"].nunique())
        for m in metric_cols:
            if m in g.columns:
                metric_values = pd.to_numeric(g[m], errors="coerce").dropna()
                base[f"{m}_mean"] = (
                    float(metric_values.mean()) if len(metric_values) else np.nan
                )
                base[f"{m}_std"] = (
                    float(metric_values.std(ddof=1))
                    if len(metric_values) > 1
                    else 0.0
                )
        rows.append(base)
    return pd.DataFrame(rows)


def latex_mean_std_table(summary: pd.DataFrame, metric_names: Optional[List[str]] = None) -> str:
    if metric_names is None:
        metric_names = ["Acc", "Prec", "Rec", "F1", "AUC", "ASR_I"]

    lines = []
    lines.append("\\begin{table*}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Adversarial training results on HIKARI-2021. Values are reported as mean $\\pm$ standard deviation.}")
    lines.append("\\label{tab:hikari_adv_training_results}")
    col_spec = "llll" + "c" * len(metric_names)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    header = ["Training", "$\\epsilon_{train}$", "Evaluation", "$\\epsilon_{eval}$"] + metric_names
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")

    for _, r in summary.iterrows():
        train = str(r["train_attack"]).upper()
        eps_train = f"{float(r['train_epsilon']):.4g}"
        attack = str(r["attack"]).upper()
        eps_eval = f"{float(r['epsilon']):.4g}"
        vals = []
        for m in metric_names:
            mean = r.get(f"{m}_mean", np.nan)
            std = r.get(f"{m}_std", np.nan)
            vals.append(f"{mean:.4f} $\\pm$ {std:.4f}")
        lines.append(" & ".join([train, eps_train, attack, eps_eval] + vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def _mean_std_from_series(values: pd.Series) -> Tuple[float, float]:
    """Returns the mean and sample standard deviation for N independent runs."""
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    mean_value = float(clean_values.mean()) if len(clean_values) else np.nan
    # For an article/experiment with 30 independent runs, ddof=1 gives the sample standard deviation.
    std_value = float(clean_values.std(ddof=1)) if len(clean_values) > 1 else 0.0
    return mean_value, std_value


def build_timing_summaries(all_metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Builds timing summaries as mean ± standard deviation across runs.

    The all_metrics DataFrame contains multiple rows per run: one for the clean
    condition and one for each evaluation attack/epsilon. Therefore, global
    training and clean-inference times are calculated after selecting a single
    row per run.
    """
    clean_rows = all_metrics[all_metrics["condition"].eq("clean")].copy()

    # Use one row per independent run to avoid counting the same run multiple times.
    run_level = clean_rows.sort_values(["run"]).drop_duplicates(subset=["run"])

    train_mean, train_std = _mean_std_from_series(run_level["train_time_s"])
    clean_inf_mean, clean_inf_std = _mean_std_from_series(run_level["clean_predict_time_s"])
    total_mean, total_std = _mean_std_from_series(run_level["total_run_time_s"])

    overall_timing = pd.DataFrame([
        {
            "n_runs": int(run_level["run"].nunique()),
            "adversarial_train_time_s_mean": train_mean,
            "adversarial_train_time_s_std": train_std,
            "clean_test_inference_time_s_mean": clean_inf_mean,
            "clean_test_inference_time_s_std": clean_inf_std,
            "total_run_time_s_mean": total_mean,
            "total_run_time_s_std": total_std,
        }
    ])

    by_condition_rows = []
    group_cols = ["attack", "epsilon", "condition"]
    for keys, group in all_metrics.groupby(group_cols, dropna=False):
        attack, epsilon, condition = keys
        pred_mean, pred_std = _mean_std_from_series(group["eval_prediction_time_s"])
        gen_mean, gen_std = _mean_std_from_series(group["eval_generation_time_s"])
        by_condition_rows.append(
            {
                "attack": attack,
                "epsilon": float(epsilon),
                "condition": condition,
                "n_runs": int(group["run"].nunique()),
                "test_inference_time_s_mean": pred_mean,
                "test_inference_time_s_std": pred_std,
                "adversarial_example_generation_time_s_mean": gen_mean,
                "adversarial_example_generation_time_s_std": gen_std,
            }
        )
    by_condition_timing = pd.DataFrame(by_condition_rows)

    txt_lines = []
    txt_lines.append("Timing summary as mean ± standard deviation")
    txt_lines.append("Standard deviation calculated across independent runs with ddof=1.")
    txt_lines.append("")
    txt_lines.append(f"N runs: {int(run_level['run'].nunique())}")
    txt_lines.append(
        "Adversarial training time: "
        f"{train_mean:.4f} ± {train_std:.4f} s"
    )
    txt_lines.append(
        "CLEAN test/inference time: "
        f"{clean_inf_mean:.4f} ± {clean_inf_std:.4f} s"
    )
    txt_lines.append(
        "Total time per run: "
        f"{total_mean:.4f} ± {total_std:.4f} s"
    )
    txt_lines.append("")
    txt_lines.append("Inference time by evaluation condition:")
    for _, row in by_condition_timing.sort_values(["condition", "attack", "epsilon"]).iterrows():
        txt_lines.append(
            f"  - {str(row['attack']).upper()} eps={row['epsilon']:.4g} "
            f"({row['condition']}): "
            f"inference={row['test_inference_time_s_mean']:.4f} ± "
            f"{row['test_inference_time_s_std']:.4f} s; "
            f"adversarial generation={row['adversarial_example_generation_time_s_mean']:.4f} ± "
            f"{row['adversarial_example_generation_time_s_std']:.4f} s"
        )

    return overall_timing, by_condition_timing, "\n".join(txt_lines)


def save_outputs(
    output_dir: Path,
    all_metrics: pd.DataFrame,
    all_history: pd.DataFrame,
    config: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics_by_run.csv"
    history_path = output_dir / "epoch_history.csv"
    summary_path = output_dir / "metrics_summary_mean_std.csv"
    latex_path = output_dir / "latex_table_results.tex"
    config_path = output_dir / "experiment_config.json"
    timing_overall_path = output_dir / "timing_summary_overall_mean_std.csv"
    timing_by_condition_path = output_dir / "timing_summary_by_condition_mean_std.csv"
    timing_txt_path = output_dir / "timing_summary_adversarial_train_and_test_inference.txt"

    all_metrics.to_csv(metrics_path, index=False)
    all_history.to_csv(history_path, index=False)

    summary = mean_std_summary(all_metrics)
    summary.to_csv(summary_path, index=False)

    latex = latex_mean_std_table(summary)
    latex_path.write_text(latex, encoding="utf-8")

    overall_timing, by_condition_timing, timing_text = build_timing_summaries(all_metrics)
    overall_timing.to_csv(timing_overall_path, index=False)
    by_condition_timing.to_csv(timing_by_condition_path, index=False)
    timing_txt_path.write_text(timing_text + "\n", encoding="utf-8")

    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + timing_text)

    print("\n[Saved files]")
    print(f"  - {metrics_path}")
    print(f"  - {history_path}")
    print(f"  - {summary_path}")
    print(f"  - {latex_path}")
    print(f"  - {timing_overall_path}")
    print(f"  - {timing_by_condition_path}")
    print(f"  - {timing_txt_path}")
    print(f"  - {config_path}")


# ---------------------------------------------------------------------
# Per-run execution
# ---------------------------------------------------------------------


def run_single_experiment(
    common_cache: "CommonDataCache",
    feature_mask_1d: np.ndarray,
    run_id: int,
    seed: int,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_process_start = time.perf_counter()
    tf.keras.backend.clear_session()
    gc.collect()

    print("\n" + "=" * 78)
    print(
        f"Run {run_id}/{args.n_runs} | seed={seed} | "
        f"adversarial training={args.train_attack.upper()} "
        f"eps={args.train_epsilon}"
    )
    print("=" * 78)

    run_checkpoint_dir = get_run_checkpoint_dir(args, run_id, seed)
    if args.reset_checkpoint and run_checkpoint_dir.exists():
        shutil.rmtree(run_checkpoint_dir)
        print(f"[Checkpoint] Removed previous state: {run_checkpoint_dir}")
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_checkpoint_dir / "checkpoint_manifest.json"
    history_checkpoint_path = run_checkpoint_dir / "epoch_history_checkpoint.csv"
    best_weights_dir = run_checkpoint_dir / "best_models"
    latest_checkpoint_dir = run_checkpoint_dir / "latest"
    result_metrics_path = run_checkpoint_dir / "metrics_completed.csv"
    result_history_path = run_checkpoint_dir / "history_completed.csv"
    result_ready_marker = run_checkpoint_dir / "RESULT_READY"

    input_size = int(np.ceil(np.sqrt(len(common_cache.feature_columns))))
    prepared_signature = build_prepared_cache_signature(
        common_cache=common_cache,
        seed=seed,
        use_smote=not args.no_smote,
        chunk_size=args.preprocess_chunk_size,
    )
    signature = build_checkpoint_signature(
        args=args,
        run_id=run_id,
        seed=seed,
        input_size=input_size,
        feature_columns=common_cache.feature_columns,
        feature_mask_1d=feature_mask_1d,
        common_cache_signature_sha256=common_cache.signature_sha256,
        prepared_cache_signature_sha256=prepared_signature["signature_sha256"],
    )

    existing_manifest: Optional[Dict[str, Any]] = None
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("signature") != signature:
            raise ValueError(
                "Checkpoint configuration mismatch. Use the original run "
                "configuration or start again with --reset-checkpoint.\n"
                f"Checkpoint: {manifest_path}"
            )
        if not args.resume and not args.reset_checkpoint:
            raise RuntimeError(
                "Existing checkpoint state was found. Use --resume to continue "
                "or --reset-checkpoint to discard it.\n"
                f"Checkpoint: {run_checkpoint_dir}"
            )

        if (
            args.resume
            and result_ready_marker.exists()
            and result_metrics_path.exists()
            and result_history_path.exists()
        ):
            print(
                f"[RESUME] Completed local results found for run {run_id}; "
                "skipping preprocessing, model construction, training, and evaluation."
            )
            return (
                pd.read_csv(result_metrics_path),
                pd.read_csv(result_history_path),
            )

    prepared_cache = ensure_prepared_run_cache(
        common_cache=common_cache,
        seed=seed,
        args=args,
        signature=prepared_signature,
    )
    prep_timing = dict(prepared_cache.prep_timing)
    feature_mask_2d = reshape_mask_to_2d(feature_mask_1d, prepared_cache.input_size)

    X_train = open_memmap_array(prepared_cache.x_train_path, advice="random")
    y_train = open_memmap_array(prepared_cache.y_train_path, advice="random")
    X_val = open_memmap_array(prepared_cache.x_val_path, advice="sequential")
    y_val = open_memmap_array(prepared_cache.y_val_path, advice="sequential")
    X_test = open_memmap_array(prepared_cache.x_test_path, advice="sequential")
    y_test = open_memmap_array(prepared_cache.y_test_path, advice="sequential")

    configure_gpu()
    set_seed(seed)
    model_setup_start = time.perf_counter()
    model = build_mtl_model_M1(
        input_shape=(prepared_cache.input_size, prepared_cache.input_size, 1),
        learning_rate=args.learning_rate,
    )
    build_optimizer_slots(model)
    model_setup_time = time.perf_counter() - model_setup_start

    logical_preprocessing_time = float(
        prep_timing.get("split_time_s", 0.0)
        + prep_timing.get("smote_time_s", 0.0)
        + prep_timing.get("scale_reshape_time_s", 0.0)
    )
    current_setup_time = logical_preprocessing_time + model_setup_time

    if existing_manifest is None:
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "signature": signature,
            "signature_sha256": signature["signature_sha256"],
            "initial_setup_time_s": float(current_setup_time),
            "initial_model_setup_time_s": float(model_setup_time),
            "prepared_cache_dir": str(prepared_cache.root),
            "prepared_cache_signature_sha256": prepared_cache.signature_sha256,
            "initial_prep_timing": prep_timing,
        }
        atomic_write_json(manifest_path, manifest)
    else:
        manifest = existing_manifest

    logical_setup_time = float(
        manifest.get("initial_setup_time_s", current_setup_time)
    )
    prep_timing = dict(manifest.get("initial_prep_timing", prep_timing))

    checkpoint_state: Dict[str, tf.Variable] = {
        "epoch": tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch"),
        "best_epoch": tf.Variable(0, dtype=tf.int64, trainable=False, name="best_epoch"),
        "best_val": tf.Variable(np.inf, dtype=tf.float64, trainable=False, name="best_val"),
        "wait": tf.Variable(0, dtype=tf.int64, trainable=False, name="early_stopping_wait"),
        "accumulated_train_time": tf.Variable(
            0.0,
            dtype=tf.float64,
            trainable=False,
            name="accumulated_train_time_s",
        ),
        "training_complete": tf.Variable(
            False,
            dtype=tf.bool,
            trainable=False,
            name="training_complete",
        ),
    }

    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=model.optimizer,
        epoch=checkpoint_state["epoch"],
        best_epoch=checkpoint_state["best_epoch"],
        best_val=checkpoint_state["best_val"],
        wait=checkpoint_state["wait"],
        accumulated_train_time=checkpoint_state["accumulated_train_time"],
        training_complete=checkpoint_state["training_complete"],
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=str(latest_checkpoint_dir),
        max_to_keep=2,
    )

    restored_history: List[Dict[str, Any]] = []
    if args.resume and checkpoint_manager.latest_checkpoint:
        restore_status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
        restore_status.assert_existing_objects_matched()
        restore_status.expect_partial()

        restored_epoch = int(checkpoint_state["epoch"].numpy())
        restored_best_epoch = int(checkpoint_state["best_epoch"].numpy())
        restored_history = load_checkpoint_history(
            history_checkpoint_path,
            confirmed_epoch=restored_epoch,
        )
        if restored_best_epoch > 0:
            restored_best_path = get_best_weights_path(
                best_weights_dir,
                restored_best_epoch,
            )
            if not restored_best_path.exists():
                raise FileNotFoundError(
                    "The restored checkpoint references best_epoch="
                    f"{restored_best_epoch}, but the corresponding versioned "
                    f"weights are missing: {restored_best_path}"
                )
            prune_unreferenced_best_weights(
                best_weights_dir,
                restored_best_epoch,
            )

        print(
            f"[RESUME] Restored {checkpoint_manager.latest_checkpoint} | "
            f"epoch={restored_epoch} | best_epoch={restored_best_epoch} | "
            f"wait={int(checkpoint_state['wait'].numpy())} | "
            f"optimizer_iterations={int(model.optimizer.iterations.numpy())}."
        )
    elif args.resume:
        print(
            f"[RESUME] No completed-epoch checkpoint found for run {run_id}; "
            "starting from epoch 1."
        )

    model, history_df, train_timing = adversarial_train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=args.batch_size,
        shuffle_buffer_size=min(int(prepared_cache.n_train), 10000),
        index_prefetch_batches=args.index_prefetch_batches,
        drop_memmap_cache_each_epoch=args.drop_memmap_cache_each_epoch,
        run_seed=seed,
        epochs=args.epochs,
        train_attack=args.train_attack,
        train_epsilon=args.train_epsilon,
        train_alpha=args.train_alpha,
        train_pgd_steps=args.train_pgd_steps,
        adv_ratio=args.adv_ratio,
        patience=args.patience,
        feature_mask_2d=feature_mask_2d,
        checkpoint=checkpoint,
        checkpoint_manager=checkpoint_manager,
        checkpoint_state=checkpoint_state,
        best_weights_dir=best_weights_dir,
        history_checkpoint_path=history_checkpoint_path,
        initial_history=restored_history,
    )

    close_memmap_array(X_train)
    close_memmap_array(y_train)
    del X_train, y_train
    gc.collect()
    trim_process_heap()

    metrics_df, eval_timing = evaluate_clean_and_adversarial(
        model=model,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        eval_epsilons=args.eval_epsilons,
        threshold_mode=args.threshold_mode,
        batch_size=args.adv_batch_size,
        pgd_steps_eval=args.eval_pgd_steps,
        feature_mask_2d=feature_mask_2d,
    )

    effective_total_run_time = (
        logical_setup_time
        + float(train_timing["effective_train_time_s"])
        + float(eval_timing["eval_total_time_s"])
    )
    current_process_wall_time = time.perf_counter() - run_process_start

    common = {
        "run": run_id,
        "seed": seed,
        "train_attack": args.train_attack.upper(),
        "train_epsilon": float(args.train_epsilon),
        "train_alpha": float(args.train_alpha) if args.train_alpha is not None else np.nan,
        "train_pgd_steps": int(args.train_pgd_steps),
        "adv_ratio": float(args.adv_ratio),
        "mask_mode": args.mask_mode,
        "mask_perturbable_features": int(np.sum(feature_mask_1d)),
        "mask_blocked_features": int(len(feature_mask_1d) - np.sum(feature_mask_1d)),
        "best_epoch": int(train_timing["best_epoch"]),
        "epochs_executed": int(train_timing["epochs_executed"]),
        "best_val_loss": float(train_timing["best_val_loss"]),
        "train_time_s": float(train_timing["effective_train_time_s"]),
        "total_run_time_s": float(effective_total_run_time),
        "effective_train_time_s": float(train_timing["effective_train_time_s"]),
        "effective_total_run_time_s": float(effective_total_run_time),
        "current_process_wall_time_s": float(current_process_wall_time),
        "logical_initial_setup_time_s": float(logical_setup_time),
        "resumed_from_epoch": int(train_timing["resumed_from_epoch"]),
        "checkpointing_enabled": True,
        "resume_requested": bool(args.resume),
        "checkpoint_signature_sha256": signature["signature_sha256"],
        "common_cache_signature_sha256": common_cache.signature_sha256,
        "prepared_cache_signature_sha256": prepared_cache.signature_sha256,
        "prepared_cache_dir": str(prepared_cache.root),
        "training_storage": "read_only_numpy_memmap",
        "full_training_tensor_materialized": False,
        **prep_timing,
        **eval_timing,
    }

    for key, value in common.items():
        metrics_df[key] = value

    history_df["run"] = run_id
    history_df["seed"] = seed
    history_df["total_run_time_s"] = effective_total_run_time
    history_df["effective_total_run_time_s"] = effective_total_run_time
    history_df["current_process_wall_time_s"] = current_process_wall_time
    history_df["checkpoint_signature_sha256"] = signature["signature_sha256"]
    history_df["prepared_cache_signature_sha256"] = prepared_cache.signature_sha256

    print(
        f"[Run {run_id}] completed | best_epoch={common['best_epoch']} "
        f"effective_train={common['effective_train_time_s']:.2f}s "
        f"effective_total={effective_total_run_time:.2f}s "
        f"process_wall={current_process_wall_time:.2f}s"
    )

    atomic_write_dataframe_csv(metrics_df, result_metrics_path)
    atomic_write_dataframe_csv(history_df, result_history_path)
    atomic_write_text(
        result_ready_marker,
        datetime.now().isoformat(timespec="seconds") + "\n",
    )

    for array in (X_val, y_val, X_test, y_test):
        close_memmap_array(array)
    del model, X_val, X_test, y_val, y_test, feature_mask_2d
    gc.collect()
    trim_process_heap()
    tf.keras.backend.clear_session()

    return metrics_df, history_df




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adversarial training of the M1 model on HIKARI-2021 with a "
            "feature-constraint mask, per-epoch checkpoint resumption, and "
            "memory-mapped batch streaming."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to the HIKARI-2021 CSV file (ALLFLOWMETER_HIKARI2021.csv).")
    parser.add_argument(
        "--output-dir",
        default="results_hikari_m1_cira_aligned_adversarial_training_masked",
        help="Output directory.",
    )
    parser.add_argument("--n-runs", type=int, default=30, help="Number of independent runs.")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Initial seed; each run uses seed_base + run index.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--adv-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    parser.add_argument("--train-attack", choices=["none", "fgsm", "pgd"], default="pgd")
    parser.add_argument("--train-epsilon", type=float, default=0.01)
    parser.add_argument(
        "--train-alpha",
        type=float,
        default=None,
        help="PGD alpha during training. Default: epsilon/4.",
    )
    parser.add_argument("--train-pgd-steps", type=int, default=5)
    parser.add_argument(
        "--adv-ratio",
        type=float,
        default=0.5,
        help="Weight of the adversarial loss during training [0,1].",
    )

    parser.add_argument(
        "--eval-epsilons",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument("--eval-pgd-steps", type=int, default=10)
    parser.add_argument(
        "--threshold-mode",
        choices=["best_f1", "fixed_05"],
        default="best_f1",
    )

    parser.add_argument("--feature-start-col", type=int, default=7)
    parser.add_argument("--feature-end-offset", type=int, default=2)
    parser.add_argument("--label-col", type=int, default=-1)
    parser.add_argument(
        "--legacy-feature-slice",
        action="store_true",
        help=(
            "Uses the legacy selection through "
            "iloc[:, feature_start_col:-feature_end_offset]. Not recommended."
        ),
    )
    parser.add_argument(
        "--mask-mode",
        choices=sorted(HIKARI_ALLOWED_FEATURES_BY_MODE.keys()),
        default="strict",
        help="HIKARI feature-constraint mask mode. Default: strict.",
    )
    parser.add_argument("--no-smote", action="store_true", help="Disables SMOTE.")

    parser.add_argument(
        "--data-cache-root",
        default=None,
        help=(
            "Shared root for cleaned and per-seed .npy memory maps. Default: "
            "<dataset-directory>/.hikari_ram_cache. All concurrent workers "
            "must use the same value to share file-backed pages."
        ),
    )
    parser.add_argument(
        "--preprocess-parallelism",
        type=int,
        default=1,
        help=(
            "Maximum number of concurrent workers allowed to execute the "
            "memory-intensive split/SMOTE/scaling phase across this cache root."
        ),
    )
    parser.add_argument(
        "--preprocess-chunk-size",
        type=int,
        default=65536,
        help="Rows transformed and written to each memory map per chunk.",
    )
    parser.add_argument(
        "--index-prefetch-batches",
        type=int,
        default=1,
        help=(
            "Number of index batches prefetched by tf.data. Feature data are "
            "not prefetched and remain memory mapped."
        ),
    )
    parser.add_argument(
        "--drop-memmap-cache-each-epoch",
        action="store_true",
        help=(
            "Requests MADV_DONTNEED for the training memory map after each "
            "epoch. This lowers resident page cache but may increase disk I/O."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restores incomplete runs from the latest completed-epoch checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Root directory for checkpoints. Default: <output-dir>/checkpoints.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help=(
            "Deletes the checkpoint for the current run and starts from epoch 1. "
            "Cannot be combined with --resume."
        ),
    )
    return parser.parse_args()




def main() -> None:
    args = parse_args()

    if args.train_alpha is None:
        args.train_alpha = args.train_epsilon / 4.0

    if args.resume and args.reset_checkpoint:
        raise ValueError("--resume and --reset-checkpoint cannot be used together.")
    if not (0.0 <= args.adv_ratio <= 1.0):
        raise ValueError("--adv-ratio must be within [0,1].")
    if args.batch_size <= 0 or args.adv_batch_size <= 0:
        raise ValueError("--batch-size and --adv-batch-size must be greater than zero.")
    if args.epochs <= 0 or args.n_runs <= 0:
        raise ValueError("--epochs and --n-runs must be greater than zero.")
    if args.train_pgd_steps <= 0 or args.eval_pgd_steps <= 0:
        raise ValueError("The numbers of PGD steps must be greater than zero.")
    if args.train_epsilon < 0 or any(eps < 0 for eps in args.eval_epsilons):
        raise ValueError("Epsilon values cannot be negative.")
    if args.preprocess_parallelism <= 0:
        raise ValueError("--preprocess-parallelism must be greater than zero.")
    if args.preprocess_chunk_size <= 0:
        raise ValueError("--preprocess-chunk-size must be greater than zero.")
    if args.index_prefetch_batches < 0:
        raise ValueError("--index-prefetch-batches cannot be negative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partial_config_path = output_dir / "experiment_config_partial.json"
    original_started_at = datetime.now().isoformat(timespec="seconds")
    if args.resume and partial_config_path.exists():
        try:
            previous_partial_config = json.loads(
                partial_config_path.read_text(encoding="utf-8")
            )
            original_started_at = previous_partial_config.get(
                "started_at",
                original_started_at,
            )
        except (json.JSONDecodeError, OSError):
            pass

    config = vars(args).copy()
    config["started_at"] = original_started_at
    config["last_process_started_at"] = datetime.now().isoformat(timespec="seconds")
    config["script"] = Path(__file__).name

    # The shared common cache is built before the GPU is configured. Concurrent
    # workers therefore wait for the one-time CSV preparation without reserving
    # unnecessary GPU memory or duplicating the complete pandas DataFrame.
    common_cache = ensure_common_data_cache(args)
    feature_cols = list(common_cache.feature_columns)

    feature_mask_1d, mask_series = build_hikari_feature_mask(
        feature_cols,
        args.mask_mode,
    )
    print_hikari_mask_report(mask_series, args.mask_mode)

    mask_csv_path = output_dir / "hikari_mask_of_feature_constraints.csv"
    mask_df = mask_series.rename("mask").reset_index()
    mask_df.columns = ["feature", "mask"]
    mask_df.to_csv(mask_csv_path, index=False)

    config["feature_columns"] = feature_cols
    config["mask_mode"] = args.mask_mode
    config["mask_perturbable_features"] = int(mask_series.sum())
    config["mask_blocked_features"] = int((mask_series == 0.0).sum())
    config["mask_csv"] = str(mask_csv_path)
    config["perturbable_features"] = mask_series[mask_series == 1.0].index.tolist()
    config["dataset_identity"] = build_dataset_identity(args.csv)
    config["data_cache"] = {
        "format_version": DATA_CACHE_FORMAT_VERSION,
        "pipeline_protocol_version": DATA_PIPELINE_PROTOCOL_VERSION,
        "root": str(get_data_cache_root(args)),
        "common_cache_dir": str(common_cache.root),
        "common_cache_signature_sha256": common_cache.signature_sha256,
        "common_feature_matrix": str(common_cache.x_path),
        "common_label_vector": str(common_cache.y_path),
        "common_storage": "read_only_numpy_memmap",
        "per_seed_preprocessed_storage": "read_only_numpy_memmap",
        "shared_across_attacks_for_same_seed": True,
        "preprocess_parallelism": int(args.preprocess_parallelism),
        "preprocess_chunk_size": int(args.preprocess_chunk_size),
    }
    config["memory_optimization"] = {
        "adversarial_evaluation": "streaming_by_batch",
        "full_adversarial_test_array_kept_in_ram": False,
        "full_training_tensor_materialized": False,
        "training_batch_source": "read_only_numpy_memmap",
        "training_tensor_scope": "one_batch_only",
        "validation_tensor_scope": "one_batch_only",
        "shared_clean_dataset_cache": True,
        "shared_preprocessed_cache_for_same_seed": True,
        "bounded_concurrent_preprocessing": True,
        "index_only_tf_dataset": True,
        "feature_prefetch": False,
        "index_prefetch_batches": int(args.index_prefetch_batches),
        "pre_smote_precision_preserved": True,
        "malloc_trim_after_large_releases": True,
        "drop_memmap_cache_each_epoch": bool(args.drop_memmap_cache_each_epoch),
    }
    config["checkpoint_protocol"] = {
        "checkpointing_enabled": True,
        "resume_requested": bool(args.resume),
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_frequency": "after_each_completed_epoch",
        "model_state_saved": True,
        "optimizer_state_saved": True,
        "early_stopping_state_saved": True,
        "epoch_history_saved": True,
        "accumulated_completed_epoch_time_saved": True,
        "best_model_storage": "versioned_by_confirmed_best_epoch",
        "configuration_signature_validated": True,
        "dataset_identity_validated": True,
        "data_cache_signature_validated": True,
        "evaluation_configuration_validated": True,
        "optimizer_restore_validation": "assert_existing_objects_matched",
        "deterministic_epoch_shuffle": True,
        "stateless_pgd_random_start_during_training": True,
        "mid_epoch_resume_supported": False,
        "evaluation_resume_granularity": "restart_full_evaluation",
        "checkpoint_root": str(
            Path(args.checkpoint_root)
            if args.checkpoint_root is not None
            else output_dir / "checkpoints"
        ),
    }
    config["timing_protocol"] = {
        "generation_time": (
            "batch preparation, adversarial generation and materialization "
            "of the adversarial batch as NumPy"
        ),
        "prediction_time": (
            "memory-map batch copy, NumPy-to-Tensor conversion, forward pass "
            "with training=False, and materialization of probabilities as NumPy"
        ),
        "gpu_synchronization": "enforced by Tensor.numpy() inside each timed interval",
        "clean_and_adversarial_prediction_api": "same direct batched model call",
        "standard_deviation": (
            "sample standard deviation across independent runs, ddof=1; "
            "0.0 when N=1"
        ),
        "train_time_s": (
            "effective completed-epoch computation time accumulated across "
            "checkpoint segments; checkpoint I/O, downtime, and discarded "
            "partial epochs are excluded"
        ),
        "total_run_time_s": (
            "effective logical time: preprocessing computation + initial model "
            "setup + completed-epoch training + final evaluation"
        ),
        "current_process_wall_time_s": (
            "elapsed wall-clock time of the process invocation that completed "
            "the run; previous interrupted invocations are not included"
        ),
    }

    atomic_write_json(partial_config_path, config)

    print(
        f"[Data cache] X={common_cache.shape} | "
        f"y distribution={common_cache.label_distribution}"
    )
    print(f"[Data cache] Common memory map: {common_cache.root}")
    print(f"[Mask] CSV saved to: {mask_csv_path}")

    all_metrics: List[pd.DataFrame] = []
    all_history: List[pd.DataFrame] = []

    for run_idx in range(args.n_runs):
        seed = args.seed_base + run_idx
        metrics_df, history_df = run_single_experiment(
            common_cache=common_cache,
            feature_mask_1d=feature_mask_1d,
            run_id=run_idx + 1,
            seed=seed,
            args=args,
        )
        all_metrics.append(metrics_df)
        all_history.append(history_df)

        partial_metrics = pd.concat(all_metrics, ignore_index=True)
        partial_history = pd.concat(all_history, ignore_index=True)
        atomic_write_dataframe_csv(
            partial_metrics,
            output_dir / "metrics_by_run_partial.csv",
        )
        atomic_write_dataframe_csv(
            partial_history,
            output_dir / "epoch_history_partial.csv",
        )

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_history_df = pd.concat(all_history, ignore_index=True)

    config["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_outputs(output_dir, all_metrics_df, all_history_df, config)

    for run_idx in range(args.n_runs):
        seed = args.seed_base + run_idx
        completed_marker = get_run_checkpoint_dir(args, run_idx + 1, seed) / "COMPLETED"
        atomic_write_text(
            completed_marker,
            datetime.now().isoformat(timespec="seconds") + "\n",
        )

    print("\nMain summary:")
    summary = mean_std_summary(all_metrics_df)
    display_cols = [
        "train_attack", "train_epsilon", "attack", "epsilon", "condition",
        "n_runs", "Acc_mean", "F1_mean", "AUC_mean", "ASR_I_mean",
        "train_time_s_mean", "total_run_time_s_mean",
    ]
    print(
        summary[[column for column in display_cols if column in summary.columns]].to_string(index=False)
    )



if __name__ == "__main__":
    main()
