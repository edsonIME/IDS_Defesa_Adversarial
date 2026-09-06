# Dataset Access and Preprocessing
# Dataset Access and Preprocessing

This study uses two public encrypted-traffic datasets. Neither dataset is
redistributed in this repository; download them from their official sources and
pass the resulting CSV path to the scripts via `--csv`.

## Datasets

### CIRA-CIC-DoHBrw-2020
- DNS-over-HTTPS (DoH) traffic, organized in two layers (DoH vs. non-DoH, and
  benign vs. malicious DoH).
- Official source: Canadian Institute for Cybersecurity (CIC) dataset portal
  (`https://www.unb.ca/cic/datasets/`). Search for **CIRA-CIC-DoHBrw-2020**.
- The engine expects a single consolidated CSV containing the identifier columns
  plus the **29 numerical traffic features** and the target column.

### HIKARI-2021
- Encrypted (HTTPS) application-layer traffic combining real benign/background
  flows with synthetically generated attacks.
- Publicly available (Ferriyan et al., 2021); Official download page: 
[https://zenodo.org/records/5199540](https://zenodo.org/records/5199540)

- Detection is cast as a binary task (benign vs. malicious), using the
  **79 numerical traffic features**.

## Binary task and label mapping

Both datasets are reduced to a binary problem: `1 = malicious`, `0 = benign`.
For CIRA-CIC-DoHBrw-2020 the target is mapped as `{DoH, NonDoH, Benign} -> 0`
and `Malicious -> 1`.

## Leakage-free preprocessing (applied per run)

1. **Strong identifiers are removed** before feature selection and are never
   used as model features. For CIRA-CIC-DoHBrw-2020 these are
   `SourceIP, DestinationIP, SourcePort, DestinationPort, TimeStamp`.
2. Rows with missing, infinite, or duplicated values are dropped.
3. Stratified split into training / validation / test partitions.
4. **SMOTE** is applied to the **training partition only**.
5. **MinMaxScaler** is fit on the **training partition only** and then applied to
   validation and test.
6. The normalized feature vector is zero-padded and reshaped to a 2-D
   pseudo-spatial layout for the convolutional front-end.

## Feature-constraint mask

A binary mask `m in {0,1}^d` restricts adversarial perturbations to operationally
manipulable features (`1 = perturbable`, `0 = blocked`). Blocked attributes
include discrete protocol-state fields (e.g., TCP flags, header and window-size
fields) and any retained identifier-like fields; perturbable attributes are
temporal/rate features and selected volumetric features.

| Dataset | Perturbable features | Total model features | Mask mode |
|---|---|---|---|
| CIRA-CIC-DoHBrw-2020 | 21 | 29 | `operational` |
| HIKARI-2021 | 30 | 79 | strict (deny-by-default) |

For CIRA-CIC-DoHBrw-2020 the exact per-feature mapping is defined in
`src/adversarial_training_cira.py` (constants `CIRA_ALLOWED_FEATURES_BY_MODE`,
`CIRA_IDENTIFIER_COLUMNS`). The engine prints the active mask report at the start
of each run.



The HIKARI-2021 experiments
> use the same protocol and an analogous mask (30 of 79 features perturbable):
`src/adversarial_training_hikari.py`