"""
Parallel experiment launcher for the adversarial-training study.

This script orchestrates the Monte Carlo runs of the three training regimes
(No-AT, FGSM-AT, PGD-AT) by spawning one independent operating-system process
per run via a training engine (``adversarial_training_cira.py`` or
``adversarial_training_hikari.py``). Runs are dispatched through a bounded queue
so that at most ``--max-parallel`` processes run concurrently, and the launcher
is resumption-safe: a run whose expected output files already exist (or whose
COMPLETED marker is present) is skipped.

Typical usage:
    python run_experiments.py \
        --engine adversarial_training_cira.py \
        --csv path/to/CIRA-CIC-DoHBrw-2020.csv \
        --output-dir results_cira \
        --runs-per-attack 10 \
        --seed-base 42 \
        --epochs 100 \
        --train-epsilon 0.01 \
        --train-pgd-steps 10 \
        --max-parallel 6
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Directory holding the launcher and the training engines.
SCRIPT_DIR = Path(__file__).resolve().parent

# Training regimes: "none" = standard training (No-AT baseline),
# "fgsm" = FGSM-AT, "pgd" = PGD-AT.
TRAIN_ATTACKS = ["none", "fgsm", "pgd"]

# Evaluation perturbation budgets passed to every run.
EVAL_EPSILONS = ["0.001", "0.005", "0.01", "0.02", "0.05"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel launcher for the AT experiments.")
    parser.add_argument("--csv", required=True, help="Path to the dataset CSV file.")
    parser.add_argument(
        "--engine",
        default="adversarial_training_cira.py",
        help="Training engine: adversarial_training_cira.py or adversarial_training_hikari.py.",
    )
    parser.add_argument("--output-dir", default="results_adversarial_training", help="Base output directory.")
    parser.add_argument("--runs-per-attack", type=int, default=10, help="Independent Monte Carlo runs per regime.")
    parser.add_argument("--seed-base", type=int, default=42, help="First seed; run i uses seed_base + i.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-epsilon", type=float, default=0.01)
    parser.add_argument("--train-pgd-steps", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=6, help="Maximum number of concurrent processes.")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval for finished processes.")
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list:
    """Builds the list of pending jobs, skipping runs that are already complete."""
    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    training_script = (SCRIPT_DIR / args.engine).resolve()
    if not training_script.is_file():
        raise FileNotFoundError(f"Training engine not found: {training_script}")

    jobs = []
    for attack in TRAIN_ATTACKS:
        for i in range(args.runs_per_attack):
            run_number = i + 1
            seed = args.seed_base + i
            output_dir = base_output / f"{attack}_run_{run_number:02d}_seed_{seed}"
            log_path = base_output / f"{attack}_run_{run_number:02d}_seed_{seed}.log"

            required_outputs = [
                output_dir / "metrics_by_run.csv",
                output_dir / "epoch_history.csv",
                output_dir / "metrics_summary_mean_std.csv",
                output_dir / "experiment_config.json",
            ]
            completed_marker = (
                output_dir / "checkpoints" / attack / f"run_01_seed_{seed}" / "COMPLETED"
            )
            outputs_complete = all(
                p.exists() and p.stat().st_size > 0 for p in required_outputs
            )
            if completed_marker.exists() or outputs_complete:
                print(f"Skipping {attack.upper()} run {run_number} | seed={seed}: already complete.")
                continue

            command = [
                sys.executable, "-u", str(training_script),
                "--csv", args.csv,
                "--output-dir", str(output_dir),
                "--n-runs", "1",
                "--seed-base", str(seed),
                "--epochs", str(args.epochs),
                "--train-attack", attack,
                "--train-epsilon", str(args.train_epsilon),
                "--train-pgd-steps", str(args.train_pgd_steps),
                "--eval-epsilons", *EVAL_EPSILONS,
                "--resume",
            ]
            jobs.append(
                {
                    "attack": attack,
                    "run_number": run_number,
                    "seed": seed,
                    "output_dir": output_dir,
                    "log_path": log_path,
                    "command": command,
                }
            )

    print(f"Total pending jobs created: {len(jobs)}")
    return jobs


def run_jobs(jobs: list, max_parallel: int, poll_seconds: int) -> None:
    """Runs the jobs with a bounded number of concurrent processes."""
    running = []
    pending = list(jobs)

    while pending or running:
        # Launch new processes while there is a free slot.
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            job["output_dir"].mkdir(parents=True, exist_ok=True)

            # Append to the log so a resumed run does not erase the previous log.
            log_file = open(job["log_path"], "a", buffering=1)
            log_file.write(
                f"\n\n===== START/RESUME {job['attack'].upper()} "
                f"run {job['run_number']} | seed={job['seed']} =====\n\n"
            )
            log_file.flush()
            print(f"Starting {job['attack'].upper()} run {job['run_number']} | seed={job['seed']}")

            process = subprocess.Popen(
                job["command"], stdout=log_file, stderr=subprocess.STDOUT
            )
            running.append({"process": process, "log_file": log_file, **job})

        # Check which processes have finished.
        still_running = []
        for item in running:
            process = item["process"]
            if process.poll() is None:
                still_running.append(item)
            else:
                item["log_file"].write(
                    f"\n\n===== FINISHED {item['attack'].upper()} "
                    f"run {item['run_number']} | seed={item['seed']} | "
                    f"code={process.returncode} =====\n\n"
                )
                item["log_file"].flush()
                item["log_file"].close()
                print(
                    f"Finished {item['attack'].upper()} run {item['run_number']} | "
                    f"seed={item['seed']} | code={process.returncode}"
                )
        running = still_running

        time.sleep(poll_seconds)

    print("All experiments (No-AT, FGSM-AT, PGD-AT) have finished.")


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    run_jobs(jobs, max_parallel=args.max_parallel, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
