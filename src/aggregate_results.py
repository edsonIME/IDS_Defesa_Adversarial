import argparse
from pathlib import Path

import pandas as pd

# Metrics aggregated as mean and sample standard deviation.
METRIC_COLUMNS = [
    "Acc", "BalancedAcc", "Prec", "Rec", "F1", "AUC", "MCC",
    "TPR", "TNR", "FPR", "FNR", "ASR_I",
    "eval_generation_time_s",
    "eval_prediction_time_s",
    "train_time_s",
    "total_run_time_s",
]

# Columns that define an experimental cell.
GROUP_COLUMNS = [
    "train_attack",
    "train_epsilon",
    "attack",
    "epsilon",
    "condition",
]


def merge_runs(base_output: Path) -> pd.DataFrame:
    """Concatenates per-run metric and history CSVs found under base_output."""
    all_metrics = []
    all_history = []

    for idx, run_dir in enumerate(sorted(base_output.glob("*_run_*")), start=1):
        metrics_path = run_dir / "metrics_by_run.csv"
        history_path = run_dir / "epoch_history.csv"

        # The training regime is encoded in the run-folder name.
        attack_name = run_dir.name.split("_run_")[0].upper()

        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            metrics["global_run"] = idx
            metrics["source_run_dir"] = run_dir.name
            metrics["parallel_train_attack"] = attack_name
            all_metrics.append(metrics)

        if history_path.exists():
            history = pd.read_csv(history_path)
            history["global_run"] = idx
            history["source_run_dir"] = run_dir.name
            history["parallel_train_attack"] = attack_name
            all_history.append(history)

    if not all_metrics:
        raise FileNotFoundError(f"No 'metrics_by_run.csv' files found under {base_output}")

    metrics_merged = pd.concat(all_metrics, ignore_index=True)
    metrics_merged.to_csv(base_output / "metrics_by_run_merged.csv", index=False)

    if all_history:
        history_merged = pd.concat(all_history, ignore_index=True)
        history_merged.to_csv(base_output / "epoch_history_merged.csv", index=False)

    print("Consolidated files saved:")
    print(" ", base_output / "metrics_by_run_merged.csv")
    if all_history:
        print(" ", base_output / "epoch_history_merged.csv")
    return metrics_merged


def summarize(metrics_merged: pd.DataFrame, base_output: Path) -> pd.DataFrame:
    """Recomputes mean and sample standard deviation per experimental cell."""
    summary_rows = []
    for keys, group in metrics_merged.groupby(GROUP_COLUMNS, dropna=False):
        row = dict(zip(GROUP_COLUMNS, keys))
        row["n_runs"] = group["source_run_dir"].nunique()
        for metric in METRIC_COLUMNS:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = base_output / "metrics_summary_mean_std_merged.csv"
    summary.to_csv(summary_path, index=False)
    print(" ", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-run AT results.")
    parser.add_argument("--output-dir", required=True, help="Base output directory used by run_experiments.py.")
    args = parser.parse_args()

    base_output = Path(args.output_dir)
    metrics_merged = merge_runs(base_output)
    summarize(metrics_merged, base_output)


if __name__ == "__main__":
    main()
