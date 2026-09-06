"""
Isolated computational-cost benchmark for the M1 detector under the three
training regimes (No-AT / FGSM-AT / PGD-AT) on HIKARI-2021 and
CIRA-CIC-DoHBrw-2020.

The benchmark reproduces, on a dedicated GPU, the per-epoch / per-step training
cost, the clean-inference latency, and the masked adversarial-generation cost
(FGSM and PGD), each measured as mean +/- sample standard deviation over R
independent repetitions after excluding warm-up epochs. It mirrors the loading,
cleaning, masking, preprocessing, and M1 architecture used by the training
engines, so the timings reflect the same computation.

Typical usage:
    python benchmark_cost.py \
        --hikari-csv path/to/ALLFLOWMETER_HIKARI2021.csv \
        --cira-csv   path/to/CIRA-CIC-DoHBrw-2020.csv \
        --datasets hikari cira \
        --output-dir benchmark_outputs

Use --smoke-test to validate the pipeline on synthetic data without the CSVs.
"""

import argparse
import os
import threading
import time
from typing import Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

try:
    from imblearn.over_sampling import SMOTE
    HAVE_SMOTE = True
except Exception:
    HAVE_SMOTE = False


# ----------------------------------------------------------------------------
# Dataset feature definitions and masks (identical to the training engines).
# ----------------------------------------------------------------------------
# HIKARI-2021
HIKARI_RESPONSE_COLUMNS = ["traffic_category", "Label"]                 # 2 targets
HIKARI_METADATA_COLUMNS = ["uid", "originh", "originp", "responh", "responp"]  # 5 ids
HIKARI_EXPECTED_FEATURES = 79
HIKARI_ALLOWED_FEATURES_STRICT = [
    "flow_duration", "fwd_pkts_per_sec", "bwd_pkts_per_sec", "flow_pkts_per_sec",
    "fwd_iat.min", "fwd_iat.max", "fwd_iat.tot", "fwd_iat.avg", "fwd_iat.std",
    "bwd_iat.min", "bwd_iat.max", "bwd_iat.tot", "bwd_iat.avg", "bwd_iat.std",
    "flow_iat.min", "flow_iat.max", "flow_iat.tot", "flow_iat.avg", "flow_iat.std",
    "payload_bytes_per_second",
    "active.min", "active.max", "active.tot", "active.avg", "active.std",
    "idle.min", "idle.max", "idle.tot", "idle.avg", "idle.std",
]  # 30 perturbable features

# CIRA-CIC-DoHBrw-2020
CIRA_IDENTIFIER_COLUMNS = ["SourceIP", "DestinationIP", "SourcePort",
                           "DestinationPort", "TimeStamp"]  # 5 strong identifiers
CIRA_TARGET_CANDIDATES = ["Label", "DoH"]
CIRA_EXPECTED_FEATURES = 29
CIRA_TRAFFIC_FEATURES = [
    "Duration",
    "FlowBytesSent", "FlowSentRate", "FlowBytesReceived", "FlowReceivedRate",
    "PacketLengthVariance", "PacketLengthStandardDeviation", "PacketLengthMean",
    "PacketLengthMedian", "PacketLengthMode", "PacketLengthSkewFromMedian",
    "PacketLengthSkewFromMode", "PacketLengthCoefficientofVariation",
    "PacketTimeVariance", "PacketTimeStandardDeviation", "PacketTimeMean",
    "PacketTimeMedian", "PacketTimeMode", "PacketTimeSkewFromMedian",
    "PacketTimeSkewFromMode", "PacketTimeCoefficientofVariation",
    "ResponseTimeTimeVariance", "ResponseTimeTimeStandardDeviation",
    "ResponseTimeTimeMean", "ResponseTimeTimeMedian", "ResponseTimeTimeMode",
    "ResponseTimeTimeSkewFromMedian", "ResponseTimeTimeSkewFromMode",
    "ResponseTimeTimeCoefficientofVariation",
]  # 29 features
CIRA_ALLOWED_FEATURES_OPERATIONAL = [
    "Duration",
    "FlowBytesSent", "FlowSentRate", "FlowBytesReceived", "FlowReceivedRate",
    "PacketLengthVariance", "PacketLengthStandardDeviation", "PacketLengthMean",
    "PacketLengthMedian", "PacketLengthMode", "PacketLengthSkewFromMedian",
    "PacketLengthSkewFromMode", "PacketLengthCoefficientofVariation",
    "PacketTimeVariance", "PacketTimeStandardDeviation", "PacketTimeMean",
    "PacketTimeMedian", "PacketTimeMode", "PacketTimeSkewFromMedian",
    "PacketTimeSkewFromMode", "PacketTimeCoefficientofVariation",
]  # 21 perturbable features

ALLOWED_FEATURES_BY_DATASET = {
    "hikari": HIKARI_ALLOWED_FEATURES_STRICT,
    "cira": CIRA_ALLOWED_FEATURES_OPERATIONAL,
}


class BenchmarkConfig:
    """Measurement parameters."""
    seed = 42
    batch_size = 64
    epochs_total = 100      # E used to project the total cost: total = (s/epoch) * E
    warmup_epochs = 1       # warm-up epochs (NOT timed)
    timed_epochs = 3        # timed epochs per repetition
    repeats = 3             # R independent repetitions -> mean +/- std
    train_epsilon = 0.01
    pgd_steps = 10
    adv_lambda = 0.5        # weight of the adversarial term in the hybrid loss
    eval_epsilons = [0.01]  # budgets used for the adversarial-generation cost
    modes = ["normal", "fgsm_at", "pgd_at"]


# ----------------------------------------------------------------------------
# Dataset loaders (faithful to the training engines).
# ----------------------------------------------------------------------------
def load_hikari_df(csv_path: str):
    """Loads, cleans, and selects the 79 HIKARI numerical traffic features."""
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    y = df["Label"].copy() if "Label" in df.columns else df.iloc[:, -1].copy()
    if not pd.api.types.is_numeric_dtype(y):
        y = y.map({
            "0": 0, "1": 1, 0: 0, 1: 1,
            "benign": 0, "Benign": 0, "BENIGN": 0,
            "background": 0, "Background": 0,
            "malicious": 1, "Malicious": 1, "attack": 1, "Attack": 1,
            False: 0, True: 1,
        })
    y = pd.to_numeric(y, errors="coerce")

    # Candidate feature-column groups, in the same order used by the engine.
    groups = []
    if set(HIKARI_RESPONSE_COLUMNS).issubset(df.columns):
        groups.append([c for c in df.columns
                       if c not in HIKARI_RESPONSE_COLUMNS
                       and c not in HIKARI_METADATA_COLUMNS])
    if df.shape[1] >= 86:                      # metadata(5) + 79 + 2 targets
        groups.append(list(df.columns[5:-2]))
    if df.shape[1] >= 88:                      # legacy layout with 2 extra columns
        groups.append(list(df.columns[7:-2]))
    if df.shape[1] >= 81:                      # already filtered: 79 + 2 targets
        groups.append(list(df.columns[:-2]))
    groups.append(list(df.columns[:-2]))       # fallback

    required = set(HIKARI_ALLOWED_FEATURES_STRICT)
    for cols in groups:
        cols = list(cols)
        cset = set(cols)
        if (len(cols) == HIKARI_EXPECTED_FEATURES
                and not (cset & set(HIKARI_RESPONSE_COLUMNS))
                and not (cset & set(HIKARI_METADATA_COLUMNS))
                and required.issubset(cset)):
            X = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
            valid = X.notna().all(axis=1) & y.notna()
            X = X.loc[valid].copy()
            y_clean = y.loc[valid].astype("int32").values
            return X, y_clean, list(cols)
    raise ValueError(
        "Could not infer the 79 HIKARI numerical features "
        "(expected: metadata + 79 features + traffic_category + Label)."
    )


def load_cira_df(csv_path: str):
    """Loads, cleans, and selects the 29 CIRA numerical traffic features."""
    df = pd.read_csv(csv_path)
    drop_cols = [c for c in CIRA_IDENTIFIER_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)        # remove strong identifiers

    label_column = df.columns[-1]              # target = last column
    mapping = {"DoH": 0, "NonDoH": 0, "Benign": 0, "Malicious": 1,
               0: 0, 1: 1, "0": 0, "1": 1}
    if not pd.api.types.is_numeric_dtype(df[label_column]):
        df[label_column] = df[label_column].map(mapping)
    else:
        df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[label_column], inplace=True)
    df[label_column] = df[label_column].astype("int32")

    feat_cols = list(df.columns[:-1])
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    y = df.iloc[:, -1].astype("int32")
    target_column = df.columns[-1]
    named = [c for c in CIRA_TRAFFIC_FEATURES if c in df.columns]
    if len(named) == CIRA_EXPECTED_FEATURES:
        X = df.loc[:, named].copy()
    else:
        blocked = set(CIRA_IDENTIFIER_COLUMNS + CIRA_TARGET_CANDIDATES + [target_column])
        cand = [c for c in df.columns if c not in blocked]
        numeric = [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) != CIRA_EXPECTED_FEATURES:
            raise ValueError(
                "Could not identify the 29 CIRA features "
                f"(named={len(named)}, numeric={len(numeric)})."
            )
        X = df.loc[:, numeric].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].copy()
    y_clean = y.loc[valid].astype("int32").values
    return X, y_clean, list(X.columns)


LOADERS = {"hikari": load_hikari_df, "cira": load_cira_df}


def build_mask_1d(dataset: str, feature_cols, n_features: int, n_perturbable: int) -> np.ndarray:
    """Builds the 1-D mask (1.0 = perturbable) by feature name, as in the engines."""
    if feature_cols is None:                    # smoke test: no column names
        m = np.zeros(n_features, np.float32)
        m[:min(n_perturbable, n_features)] = 1.0
        return m
    s = pd.Series(0.0, index=list(feature_cols), dtype="float32")
    present = [f for f in ALLOWED_FEATURES_BY_DATASET[dataset] if f in s.index]
    s.loc[present] = 1.0
    return s.values.astype("float32")


def load_dataset(dataset: str, csv_path: str, n_features_expected: int,
                 n_perturbable: int, smoke_test: bool):
    if smoke_test:
        d, n = n_features_expected, 6000
        rng = np.random.default_rng(BenchmarkConfig.seed)
        X = rng.random((n, d)).astype(np.float32)
        y = (rng.random(n) < 0.2).astype(int)
        return X, y, None                       # feature_cols=None -> synthetic mask
    X_df, y, feature_cols = LOADERS[dataset](csv_path)
    return X_df.values.astype(np.float32), np.asarray(y), feature_cols


# ----------------------------------------------------------------------------
# Leakage-free common preprocessing.
# ----------------------------------------------------------------------------
def preprocess_common(X, y, seed, feature_mask_1d, smoke_test):
    # Stratified 70/15/15 split (same as the engines).
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=seed)
    # SMOTE on the training partition only.
    if HAVE_SMOTE and not smoke_test:
        Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    # MinMax scaler fit on the training partition only.
    sc = MinMaxScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    # Pseudo-spatial reshape (s, s, 1) with zero-padding (same as reshape_to_2d).
    d = Xtr.shape[1]
    s = int(np.ceil(np.sqrt(d)))
    pad = s * s - d
    to_img = lambda a: np.pad(a, ((0, 0), (0, pad))).reshape((-1, s, s, 1)).astype(np.float32)
    Xtr_img, Xte_img = to_img(Xtr), to_img(Xte)
    # Mask (s, s, 1): 1-D mask by name -> pad to s*s -> reshape.
    flat = np.zeros(s * s, np.float32)
    flat[:d] = np.asarray(feature_mask_1d, np.float32)[:d]
    mask = flat.reshape((s, s, 1))
    info = {"n_features": d, "s": s, "pad": pad, "n_perturbable": int(flat.sum()),
            "n_train": len(Xtr_img), "n_test": len(Xte_img)}
    return Xtr_img, ytr, Xte_img, yte, mask, info


# ----------------------------------------------------------------------------
# M1 architecture (two faithful variants: HIKARI 9x9 and CIRA 6x6).
# ----------------------------------------------------------------------------
def _eca_block(input_tensor):
    """Canonical channel-wise Efficient Channel Attention."""
    channels = input_tensor.shape[-1]
    if channels is None:
        raise ValueError("ECA requires a statically known channel dimension.")
    channels = int(channels)
    gamma, beta = 2.0, 1.0
    kernel_estimate = int(abs((np.log2(channels) + beta) / gamma))
    k_size = kernel_estimate if kernel_estimate % 2 == 1 else kernel_estimate + 1
    k_size = max(3, k_size)
    squeeze = layers.GlobalAveragePooling2D()(input_tensor)
    squeeze = layers.Reshape((channels, 1))(squeeze)
    squeeze = layers.Conv1D(filters=1, kernel_size=k_size, padding="same", use_bias=False)(squeeze)
    squeeze = layers.Activation("sigmoid")(squeeze)
    squeeze = layers.Reshape((1, 1, channels))(squeeze)
    return layers.Multiply()([input_tensor, squeeze])


def _transformer_encoder(x_in, head_size, num_heads, ff_dim, dropout=0.3):
    x = layers.LayerNormalization(epsilon=1e-6)(x_in)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + x_in
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(int(x_in.shape[-1]))(x)
    return x + res


def _m1_head(x):
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    logits = layers.Dense(1, activation=None, name="logits")(x)
    return layers.Activation("sigmoid", dtype="float32", name="binary_output")(logits)


def _compile_m1(inputs, output, learning_rate):
    model = Model(inputs=inputs, outputs=output, name="M1_ECA_CNN_Transformer")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy", "Precision", "Recall", "AUC"])
    return model


def build_m1_hikari(input_shape: Tuple[int, int, int], learning_rate: float = 1e-4) -> Model:
    """M1 for the 9x9 HIKARI input. Blocks 1-2 pool to 2x2; block 3 keeps 4 tokens."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.Activation("relu")(layers.BatchNormalization()(x))
    x = _eca_block(layers.MaxPooling2D((2, 2))(x))
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.Activation("relu")(layers.BatchNormalization()(x))
    x = _eca_block(layers.MaxPooling2D((2, 2))(x))
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.Activation("relu")(layers.BatchNormalization()(x))
    seq = int(x.shape[1]) * int(x.shape[2])
    if seq < 2:
        raise ValueError(f"Transformer needs >= 2 tokens; got {seq}.")
    x = layers.Reshape((seq, int(x.shape[-1])))(x)
    x = _transformer_encoder(x, head_size=128, num_heads=4, ff_dim=256, dropout=0.3)
    x = layers.Flatten()(x)
    return _compile_m1(inputs, _m1_head(x), learning_rate)


def build_m1_cira(input_shape: Tuple[int, int, int], learning_rate: float = 1e-4) -> Model:
    """M1 for the 6x6 CIRA input. Block 2 uses padding='same' pooling to keep 4 tokens."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.Activation("relu")(layers.BatchNormalization()(x))
    x = _eca_block(layers.MaxPooling2D((2, 2))(x))
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.Activation("relu")(layers.BatchNormalization()(x))
    x = _eca_block(layers.MaxPooling2D((2, 2), padding="same")(x))
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.Activation("relu")(layers.BatchNormalization()(x))
    seq = int(x.shape[1]) * int(x.shape[2])
    if seq < 2:
        raise ValueError(f"Transformer needs >= 2 tokens; got {seq}.")
    x = layers.Reshape((seq, int(x.shape[-1])))(x)
    x = _transformer_encoder(x, head_size=128, num_heads=4, ff_dim=256, dropout=0.3)
    x = layers.Flatten()(x)
    return _compile_m1(inputs, _m1_head(x), learning_rate)


BUILDERS = {"hikari": build_m1_hikari, "cira": build_m1_cira}


# ----------------------------------------------------------------------------
# Masked attacks, training steps, and timing utilities.
# ----------------------------------------------------------------------------
bce = tf.keras.losses.BinaryCrossentropy()


def gen_fgsm(model, x, y, eps, mask):
    with tf.GradientTape() as t:
        t.watch(x)
        loss = bce(y, model(x, training=False))
    g = t.gradient(loss, x)
    return tf.clip_by_value(x + eps * mask * tf.sign(g), 0.0, 1.0)


def gen_pgd(model, x, y, eps, alpha, steps, mask):
    xa = tf.clip_by_value(x + mask * tf.random.uniform(tf.shape(x), -eps, eps), 0.0, 1.0)
    for _ in range(steps):
        with tf.GradientTape() as t:
            t.watch(xa)
            loss = bce(y, model(xa, training=False))
        g = t.gradient(loss, xa)
        xa = xa + alpha * mask * tf.sign(g)
        delta = tf.clip_by_value(xa - x, -eps, eps)
        xa = tf.clip_by_value(x + mask * delta, 0.0, 1.0)
    return xa


def _reg(model):
    return tf.add_n(model.losses) if model.losses else tf.constant(0.0)


@tf.function
def step_normal(model, opt, x, y):
    with tf.GradientTape() as t:
        loss = bce(y, model(x, training=True)) + _reg(model)
    opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables))
    return loss


@tf.function
def step_fgsm_at(model, opt, x, y, eps, mask, lam):
    xa = gen_fgsm(model, x, y, eps, mask)
    with tf.GradientTape() as t:
        loss = (1 - lam) * bce(y, model(x, training=True)) \
            + lam * bce(y, model(xa, training=True)) + _reg(model)
    opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables))
    return loss


@tf.function
def step_pgd_at(model, opt, x, y, eps, alpha, steps, mask, lam):
    xa = gen_pgd(model, x, y, eps, alpha, steps, mask)
    with tf.GradientTape() as t:
        loss = (1 - lam) * bce(y, model(x, training=True)) \
            + lam * bce(y, model(xa, training=True)) + _reg(model)
    opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables))
    return loss


def device_sync(t):
    """Forces device synchronization by reading one element back to host."""
    try:
        return float(np.asarray(t).reshape(-1)[0])
    except Exception:
        return None


def gpu_peak_mb():
    try:
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2 ** 20
    except Exception:
        return None


def gpu_reset_mem():
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Timing: training, clean inference, and adversarial generation.
# ----------------------------------------------------------------------------
def time_training(mode, Xtr, ytr, mask, build_fn, cfg: BenchmarkConfig):
    bs = cfg.batch_size
    spe = int(np.ceil(len(Xtr) / bs))
    mt = tf.constant(mask)
    eps = tf.constant(cfg.train_epsilon, tf.float32)
    alpha = tf.constant(cfg.train_epsilon / 4, tf.float32)
    lam = tf.constant(cfg.adv_lambda, tf.float32)
    steps = int(cfg.pgd_steps)
    ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr.astype(np.float32).reshape(-1, 1)))
          .shuffle(min(len(Xtr), 10000), seed=cfg.seed).batch(bs).prefetch(tf.data.AUTOTUNE))

    def epoch(model, opt):
        last = None
        for xb, yb in ds:
            if mode == "normal":
                last = step_normal(model, opt, xb, yb)
            elif mode == "fgsm_at":
                last = step_fgsm_at(model, opt, xb, yb, eps, mt, lam)
            else:
                last = step_pgd_at(model, opt, xb, yb, eps, alpha, steps, mt, lam)
        device_sync(last)
        return last

    times, mems = [], []
    for _ in range(cfg.repeats):
        model = build_fn(Xtr.shape[1:])
        opt = tf.keras.optimizers.Adam(1e-3)
        try:
            opt.build(model.trainable_variables)  # build optimizer slots outside @tf.function (Keras 3)
        except Exception:
            pass
        for _ in range(cfg.warmup_epochs):
            epoch(model, opt)                     # warm-up (not timed)
        gpu_reset_mem()
        for _ in range(cfg.timed_epochs):
            t0 = time.perf_counter()
            epoch(model, opt)
            times.append(time.perf_counter() - t0)
        m = gpu_peak_mb()
        if m:
            mems.append(m)
        del model, opt

    a = np.array(times)
    se = a.mean()
    ss = a.std(ddof=1) if len(a) > 1 else 0.0
    return {"mode": mode, "s_epoch_mean": se, "s_epoch_std": ss,
            "ms_step_mean": se / spe * 1e3, "ms_step_std": ss / spe * 1e3,
            "samples_per_s": len(Xtr) / se, "total_s_mean": se * cfg.epochs_total,
            "total_s_std": ss * cfg.epochs_total, "steps_per_epoch": spe,
            "peak_mem_mb": float(np.mean(mems)) if mems else None}


def time_inference_and_generation(Xte, yte, mask, build_fn, cfg: BenchmarkConfig):
    bs = cfg.batch_size
    mt = tf.constant(mask)
    steps = int(cfg.pgd_steps)
    model = build_fn(Xte.shape[1:])
    yb_all = yte.astype(np.float32).reshape(-1, 1)
    n = len(Xte)
    ds = tf.data.Dataset.from_tensor_slices((Xte, yb_all)).batch(bs).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def infer(xb):
        return model(xb, training=False)

    for xb, _ in ds:
        infer(xb)
        break                                     # warm-up
    it = []
    for _ in range(cfg.repeats):
        t0 = time.perf_counter()
        last = None
        for xb, _ in ds:
            last = infer(xb)
        device_sync(last)
        it.append(time.perf_counter() - t0)
    it = np.array(it)
    inference = {"op": "clean_inference", "total_s_mean": it.mean(),
                 "total_s_std": it.std(ddof=1) if len(it) > 1 else 0.0,
                 "us_per_sample": it.mean() / n * 1e6, "samples_per_s": n / it.mean()}

    @tf.function
    def fgsm_tf(xb, yb, eps):
        return gen_fgsm(model, xb, yb, eps, mt)

    @tf.function
    def pgd_tf(xb, yb, eps, al):
        return gen_pgd(model, xb, yb, eps, al, steps, mt)

    gen = []
    for ev in cfg.eval_epsilons:
        ec = tf.constant(ev, tf.float32)
        ac = tf.constant(ev / 4, tf.float32)
        for xb, yb in ds:
            fgsm_tf(xb, yb, ec)
            pgd_tf(xb, yb, ec, ac)
            break                                 # warm-up
        for atk, fn, needs_alpha in (("FGSM", fgsm_tf, False), ("PGD", pgd_tf, True)):
            ts = []
            for _ in range(cfg.repeats):
                t0 = time.perf_counter()
                last = None
                for xb, yb in ds:
                    last = fn(xb, yb, ec, ac) if needs_alpha else fn(xb, yb, ec)
                device_sync(last)
                ts.append(time.perf_counter() - t0)
            ts = np.array(ts)
            gen.append({"op": f"generation_{atk}", "epsilon": ev, "total_s_mean": ts.mean(),
                        "total_s_std": ts.std(ddof=1) if len(ts) > 1 else 0.0,
                        "us_per_sample": ts.mean() / n * 1e6, "samples_per_s": n / ts.mean()})
    del model
    return inference, gen


# ----------------------------------------------------------------------------
# Reporting (CSV + combined LaTeX tables).
# ----------------------------------------------------------------------------
OP_LABELS = {"clean_inference": "Clean inference",
             "generation_FGSM": "FGSM generation",
             "generation_PGD": "PGD generation"}
MODE_LABELS = {"normal": "Standard (No-AT)", "fgsm_at": "FGSM-AT", "pgd_at": "PGD-AT"}
DATASET_LABELS = {"hikari": "HIKARI-2021", "cira": "CIRA-CIC-DoHBrw-2020"}


def write_reports(all_train, all_infer, datasets, out_dir):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    df_train = pd.DataFrame(all_train)
    df_infer = pd.DataFrame(all_infer)
    df_train.to_csv(os.path.join(out_dir, "cost_training.csv"), index=False)
    df_infer.to_csv(os.path.join(out_dir, "cost_inference_generation.csv"), index=False)

    def factor(row):
        base = next((x for x in all_train if x["dataset"] == row["dataset"] and x["mode"] == "normal"), None)
        return row["total_s_mean"] / base["total_s_mean"] if base and base["total_s_mean"] else float("nan")

    lines = []
    lines.append("% ===== Training cost (isolated, mean +/- std) =====")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Regime & s/epoch & ms/step & samples/s & Total (s) & Factor \\\\")
    lines.append("\\midrule")
    for ds_name in datasets:
        for r in [x for x in all_train if x["dataset"] == ds_name]:
            lines.append(f"{DATASET_LABELS[ds_name]} & {MODE_LABELS[r['mode']]} & "
                         f"{r['s_epoch_mean']:.2f} $\\pm$ {r['s_epoch_std']:.2f} & "
                         f"{r['ms_step_mean']:.2f} $\\pm$ {r['ms_step_std']:.2f} & "
                         f"{r['samples_per_s']:.0f} & "
                         f"{r['total_s_mean']:.0f} $\\pm$ {r['total_s_std']:.0f} & "
                         f"{factor(r):.2f}$\\times$ \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    lines.append("% ===== Clean inference + adversarial generation (isolated) =====")
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Operation & Total (s) & Per sample ($\\mu$s) & samples/s \\\\")
    lines.append("\\midrule")
    for ds_name in datasets:
        for g in [x for x in all_infer if x["dataset"] == ds_name]:
            op = OP_LABELS.get(g["op"], g["op"].replace("_", r"\_"))
            if "epsilon" in g and not pd.isna(g.get("epsilon")):
                op += f" ($\\epsilon$={g['epsilon']})"
            lines.append(f"{DATASET_LABELS[ds_name]} & {op} & "
                         f"{g['total_s_mean']:.3f} $\\pm$ {g['total_s_std']:.3f} & "
                         f"{g['us_per_sample']:.1f} & {g['samples_per_s']:.0f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    tex_path = os.path.join(out_dir, "cost_tables.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Saved: cost_training.csv, cost_inference_generation.csv, cost_tables.tex (in {out_dir})")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def configure_gpu() -> None:
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    print("TF:", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated computational-cost benchmark for M1.")
    parser.add_argument("--hikari-csv", default=None, help="Path to the HIKARI-2021 CSV file.")
    parser.add_argument("--cira-csv", default=None, help="Path to the CIRA-CIC-DoHBrw-2020 CSV file.")
    parser.add_argument("--datasets", nargs="+", default=["hikari", "cira"], choices=["hikari", "cira"])
    parser.add_argument("--output-dir", default="benchmark_outputs")
    parser.add_argument("--smoke-test", action="store_true", help="Use synthetic data (no CSV required).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_gpu()
    cfg = BenchmarkConfig()

    dataset_specs = {
        "hikari": {"csv": args.hikari_csv, "n_features_expected": 79, "n_perturbable": 30},
        "cira": {"csv": args.cira_csv, "n_features_expected": 29, "n_perturbable": 21},
    }

    all_train, all_infer = [], []
    for name in args.datasets:
        spec = dataset_specs[name]
        if not args.smoke_test and not spec["csv"]:
            raise ValueError(f"Provide --{name}-csv or use --smoke-test.")
        print(f"\n========== {name.upper()} ==========")
        tf.keras.backend.clear_session()
        X, y, feature_cols = load_dataset(
            name, spec["csv"], spec["n_features_expected"], spec["n_perturbable"], args.smoke_test
        )
        mask_1d = build_mask_1d(name, feature_cols, X.shape[1], spec["n_perturbable"])
        Xtr, ytr, Xte, yte, mask, info = preprocess_common(X, y, cfg.seed, mask_1d, args.smoke_test)
        build_fn = BUILDERS[name]
        print(f"  features={info['n_features']}  perturbable={info['n_perturbable']}  "
              f"reshape={info['s']}x{info['s']}  n_train={info['n_train']}  n_test={info['n_test']}")
        for mode in cfg.modes:
            print(f"  - training: {mode}")
            r = time_training(mode, Xtr, ytr, mask, build_fn, cfg)
            r["dataset"] = name
            all_train.append(r)
        print("  - inference + adversarial generation")
        inf, gen = time_inference_and_generation(Xte, yte, mask, build_fn, cfg)
        inf["dataset"] = name
        all_infer.append(inf)
        for g in gen:
            g["dataset"] = name
            all_infer.append(g)

    write_reports(all_train, all_infer, args.datasets, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
