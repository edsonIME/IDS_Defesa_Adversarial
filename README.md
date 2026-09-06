# Robustness–Cost Trade-offs in Adversarial Training for DL-Based NIDS

# Replication Package

Official code repository for the paper "Robustness--Cost Trade-offs of Adversarial Training for Network Intrusion Detection: An Empirical Comparison of FGSM and PGD", accepted at LADC 2026.

Authors:

    Edson B. de Souza (Military Institute of Engineering - IME, Brazil)

    Paulo Cesar Pellanda (Military Institute of Engineering - IME, Brazil)

    Ronaldo Moreira Salles (CIICESI, ESTG, Polytechnic of Porto, Portugal)
	
## Objective

The repository reproduces the adversarial-training (AT) study for the hybrid
**M1** detector (CNN + Efficient Channel Attention + Transformer encoder) on
encrypted-traffic intrusion detection. It compares three training regimes —
standard training (**No-AT**), single-step **FGSM-AT**, and multi-step
**PGD-AT** — under a binary **feature-constraint mask** that restricts
perturbations to operationally manipulable traffic features. Models are
evaluated on clean inputs and under FGSM and PGD attacks across five
perturbation budgets, using the `F1`-score and an IDS-specific Attack Success
Rate (**ASR-I**), and the training/inference cost of each regime is measured in
an isolated benchmark.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── adversarial_training_cira.py    # Training + evaluation engine (CIRA-CIC-DoHBrw-2020)
│   ├── adversarial_training_hikari.py  # Training + evaluation engine (HIKARI-2021)
│   ├── run_experiments.py              # Parallel launcher (No-AT / FGSM-AT / PGD-AT)
│   ├── aggregate_results.py            # Merges per-run outputs, recomputes mean ± std
│   └── benchmark_cost.py               # Isolated training/inference/generation cost benchmark
└── docs/
    ├── DATA.md                   # Dataset access and preprocessing notes
    └── EXECUTION.md              # Step-by-step execution guide
```
## Dataset Access

The datasets are not redistributed in this repository due to licensing and storage constraints. Users must download them from their official sources before running the experiments.


## Quick start

```bash
# 1) Environment (Python 3.10–3.11 recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Obtain the datasets (see docs/DATA.md) and note the CSV path.

# 3) Launch all regimes (No-AT, FGSM-AT, PGD-AT), 10 runs each.
#    Use --engine to select the dataset engine.
python src/run_experiments.py \
    --engine adversarial_training_cira.py \
    --csv path/to/CIRA-CIC-DoHBrw-2020.csv \
    --output-dir results_cira \
    --runs-per-attack 10 \
    --max-parallel 6

# 4) Aggregate per-run outputs into summary tables:
python src/aggregate_results.py --output-dir results_cira

# 5) (Optional) Reproduce the isolated training/inference/generation cost:
python src/benchmark_cost.py --cira-csv path/to/CIRA-CIC-DoHBrw-2020.csv --datasets cira
```

For HIKARI-2021, pass `--engine adversarial_training_hikari.py` and the HIKARI CSV.
A single regime/run can also be executed directly through an engine:

```bash
python src/adversarial_training_cira.py \
    --csv path/to/CIRA-CIC-DoHBrw-2020.csv \
    --output-dir results_pgd_at \
    --train-attack pgd --n-runs 10 --resume
```

See `docs/EXECUTION.md` for the full guide and
`python src/adversarial_training_cira.py --help` for all options.

## Default training configuration

The engine defaults match the protocol reported in the paper:

| Parameter | Flag | Default |
|---|---|---|
| Training budget (L-inf) | `--train-epsilon` | `0.01` |
| PGD training step size | `--train-alpha` | `epsilon / 4` |
| PGD training steps | `--train-pgd-steps` | `10` (launcher) |
| Clean/adversarial loss weight (lambda) | `--adv-ratio` | `0.5` |
| Epochs (early stopping) | `--epochs` / `--patience` | `100` / `10` |
| Optimizer / learning rate | (Adam) / `--learning-rate` | `1e-4` |
| Batch size / adv. batch | `--batch-size` / `--adv-batch-size` | `32` / `64` |
| Evaluation budgets | `--eval-epsilons` | `0.001 0.005 0.01 0.02 0.05` |
| Evaluation PGD steps | `--eval-pgd-steps` | `10` |
| Decision threshold | `--threshold-mode` | `best_f1` (on validation) |
| Seeds | `--seed-base` | `42` (run i uses 42 + i) |

## Reproducibility notes

- Each run cleans and splits the data, applies **SMOTE on the training partition
  only**, fits the **MinMaxScaler on training data only**, and selects the
  decision threshold on the validation set — i.e., a leakage-free protocol.
- Adversarial examples are generated **on the fly** and constrained by the
  feature mask in initialization, gradient updates, and projection.
- The engine is **checkpoint/resume-capable** (per-epoch) and uses memory-mapped
  arrays so that concurrent runs share a single cleaned-data cache.
- Results are reported as **mean ± sample standard deviation** over the
  independent runs.

Two engines are provided, one per dataset (`adversarial_training_cira.py` and
`adversarial_training_hikari.py`); they share the same protocol and differ only
in the dataset-specific feature selection and mask (see `docs/DATA.md`).


