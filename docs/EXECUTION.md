# Execution Guide

## 1. Environment

Python 3.10–3.11 with an NVIDIA GPU (CUDA 12) is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If your CUDA/cuDNN stack differs, install the matching TensorFlow build and keep
the remaining pinned versions.

## 2. Data

Download CIRA-CIC-DoHBrw-2020 (and/or HIKARI-2021) as described in
[`DATA.md`](DATA.md) and note the consolidated CSV path.

## 3. Run all regimes in parallel

`run_experiments.py` launches the three regimes (No-AT, FGSM-AT, PGD-AT), each
with `--runs-per-attack` independent Monte Carlo runs, dispatched through a
bounded queue. It is **resume-safe**: completed runs are skipped on relaunch.

```bash
# CIRA-CIC-DoHBrw-2020
python src/run_experiments.py \
    --engine adversarial_training_cira.py \
    --csv path/to/CIRA-CIC-DoHBrw-2020.csv \
    --output-dir results_cira \
    --runs-per-attack 10 \
    --seed-base 42 \
    --epochs 100 \
    --train-epsilon 0.01 \
    --train-pgd-steps 10 \
    --max-parallel 6

# HIKARI-2021 (same protocol, HIKARI engine + CSV)
python src/run_experiments.py \
    --engine adversarial_training_hikari.py \
    --csv path/to/ALLFLOWMETER_HIKARI2021.csv \
    --output-dir results_hikari \
    --runs-per-attack 10
```

Set `--max-parallel` according to available GPU memory and cores. Each run writes
its own subfolder and `.log` under `--output-dir`.

## 4. Run a single regime/run (optional)

```bash
python src/adversarial_training_cira.py \
    --csv path/to/CIRA-CIC-DoHBrw-2020.csv \
    --output-dir results_pgd_at \
    --train-attack pgd \
    --train-epsilon 0.01 \
    --train-pgd-steps 10 \
    --adv-ratio 0.5 \
    --n-runs 10 \
    --eval-epsilons 0.001 0.005 0.01 0.02 0.05 \
    --resume
```

Use `--train-attack none` for the standard-training baseline and `fgsm` for
FGSM-AT. Use `adversarial_training_hikari.py` for HIKARI-2021. Run
`python src/adversarial_training_cira.py --help` for every option.

## 5. Aggregate results

```bash
python src/aggregate_results.py --output-dir results_adversarial_training
```

This writes, under the output directory:
- `metrics_by_run_merged.csv` — all per-run metrics concatenated;
- `epoch_history_merged.csv` — per-epoch training histories;
- `metrics_summary_mean_std_merged.csv` — mean ± sample std per experimental cell
  (training regime × training budget × evaluation attack × budget × condition).

## 6. Per-run outputs

Each run directory contains, among others:
- `metrics_by_run.csv`, `metrics_summary_mean_std.csv` — clean and under-attack
  metrics (Accuracy, Precision, Recall, F1, AUC, MCC, ASR-I, timing);
- `epoch_history.csv` — per-epoch training/validation history;
- `experiment_config.json` — full resolved configuration and signatures;
- `checkpoints/` — per-epoch checkpoints enabling safe resumption.

## 7. Computational-cost benchmark

Training cost scales with the regime (No-AT < FGSM-AT < PGD-AT); inference cost
is independent of the training regime. The isolated benchmark should run as a
single process with the GPU allocated exclusively and the datasets executed
sequentially, excluding a warm-up epoch:

```bash
python src/benchmark_cost.py \
    --hikari-csv path/to/ALLFLOWMETER_HIKARI2021.csv \
    --cira-csv   path/to/CIRA-CIC-DoHBrw-2020.csv \
    --datasets hikari cira \
    --output-dir benchmark_outputs
```

It writes `cost_training.csv`, `cost_inference_generation.csv`, and a combined
`cost_tables.tex` under `--output-dir`. Use `--smoke-test` to validate the
pipeline on synthetic data without the CSV files.
