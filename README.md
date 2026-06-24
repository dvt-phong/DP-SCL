# DP-SCL

DP-SCL is a PyTorch implementation of supervised contrastive learning for
MOOC dropout prediction. Each learner-course enrollment is represented as a
temporal activity tensor, encoded by a shared LSTM and multi-head attention
network, optimized with a supervised contrastive objective, and classified with
a binary dropout head.

The current repository focuses on DP-SCL only. Legacy non-DP-SCL training paths
and alternative model families have been removed.

## Key Features

- Supervised contrastive training with two stochastic views of each learner
  activity sequence.
- Shared LSTM + multi-head attention encoder with attentive pooling.
- BCE dropout classification loss combined with SupCon loss.
- Stratified train/validation/test splits across one or more random seeds.
- Support for XuetangX, OULAD, and SNAP ACT-MOOC temporal `.npz` inputs.
- CSV and text reports for per-seed metrics, epoch history, and summary results.

## Repository Layout

```text
.
|-- train_experiment.py                  # main DP-SCL experiment runner
|-- train.py                             # compatibility wrapper for old CLI style
|-- run_scripts/
|   |-- run_dp_scl.py                    # convenience launcher from repo root
|   `-- inspect_xuetangx_npz.py          # read-only NPZ inspection utility
|-- src/
|   |-- dataset_config.py                # dataset metadata and expected tensor shapes
|   |-- mode_registry.py                 # DP-SCL backend mode resolution
|   |-- data_validator.py                # data validation helper
|   |-- models/
|   |   |-- common.py                    # SupCon loss, augmentation, shared layers
|   |   `-- supcon.py                    # DP-SCL encoder, projection head, classifier
|   |-- dataprocess/                     # dataset preprocessing scripts
|   `-- tests/test_dp_scl.py             # model/loss smoke test
|-- statistical_significance_test.py     # paired tests from report.txt
|-- draw-curve.py                        # loss/AUC curve plotting script
|-- excel-data.py                        # exports plotting history to CSV
|-- requirements.txt
`-- LICENSE
```

Local datasets, checkpoints, generated results, and virtual environments are
ignored by Git.

## Data Format

Temporal `.npz` files are expected under `datastore/`.

| Dataset | Expected file | Shape suffix |
| --- | --- | --- |
| `xuetangx` | `datastore/all_data_std.npz` | `(5, 7, 22)` |
| `oulad` | `datastore/oulad_data_std.npz` | `(5, 7, 20)` |
| `snap` | `datastore/snap_data_std.npz` | `(5, 7, 6)` |

Each file must contain:

```text
t_data   train temporal tensor, shape (N_train, week_count, days_per_week, activity_num)
t_label  train labels, shape (N_train,)
v_data   held-out temporal tensor, shape (N_test, week_count, days_per_week, activity_num)
v_label  held-out labels, shape (N_test,)
```

`train_experiment.py` concatenates `t_*` and `v_*`, then creates stratified
train/validation/test splits for each seed.

## Installation

Create and activate a Python environment, then install the repository
dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes the packages used by training, preprocessing,
statistics, plotting, and smoke tests. Install a CUDA-enabled PyTorch build if
GPU training is required.

## Quick Start

Run DP-SCL on XuetangX with the default five seeds:

```bash
python run_scripts/run_dp_scl.py -indir . -outdir . --dataset xuetangx --max-epochs 15
```

Run with explicit seeds:

```bash
python train_experiment.py -indir . -outdir . --dataset xuetangx --seeds 1 11 111 1111 11111 --max-epochs 15
```

Compatibility wrapper for the previous single-run style:

```bash
python train.py -indir . -outdir . -mode dp_scl --dataset xuetangx -r 42 -e 15
```

Supported datasets:

```text
xuetangx, oulad, snap
```

## Training Objective

For each batch, DP-SCL minimizes:

```text
loss = BCEWithLogitsLoss(logits, label)
     + lambda_con * SupConLoss([z1, z2], label)
```

Default hyperparameters:

```text
lambda_con = 0.1
temperature = 0.07
mask_ratio = 0.15
noise_std = 0.05
hidden_size = 128
batch_size = 256
learning_rate = 1e-4
max_epochs = 200
patience = 30
split = 0.60 train / 0.10 validation / 0.30 test
```

During evaluation, the stochastic augmentations and projection outputs are
disabled; the model returns dropout logits only.

## Outputs

Each run writes results to:

```text
results/dp_scl_<timestamp>/
```

Main outputs:

```text
config.json
splits/seed_<seed>_train.npy
splits/seed_<seed>_val.npy
splits/seed_<seed>_test.npy
checkpoints/dp_scl_seed_<seed>.pt
per_seed_results.csv
epoch_history.csv
summary_results.csv
report.txt
```

Metrics include AUC, accuracy, precision, recall, and F1. The classification
threshold is selected on the validation split by maximizing F1, then applied to
the test split.

## Utilities

Inspect a XuetangX temporal NPZ sample:

```bash
python run_scripts/inspect_xuetangx_npz.py --split train --sample-index 0 --top-k 12
```

Run statistical significance analysis from `report.txt`:

```bash
python statistical_significance_test.py
```

Export training history used by the curve plot:

```bash
python excel-data.py
```

Regenerate the loss/AUC figure:

```bash
python draw-curve.py
```

`draw-curve.py` reads:

```text
datastore/loss_curve/history_dp_scl.json
datastore/loss_curve/history_wo_supcon.json
```

and writes:

```text
loss_curve_3subplots.png
```

## Validation

Run the smoke test:

```bash
python -m pytest src/tests/test_dp_scl.py
```

Run a syntax check over the main scripts:

```bash
python -m py_compile train.py train_experiment.py statistical_significance_test.py draw-curve.py excel-data.py run_scripts/run_dp_scl.py run_scripts/inspect_xuetangx_npz.py
```

## Attribution

This project implements DP-SCL for MOOC dropout prediction. The model and
training objective follow the project manuscript:

```text
Student Dropout Prediction in Online Courses Based on Supervised Contrastive
Learning, Doan Van Thanh Phong et al.
```

Component-level source notes:

- `src/models/supcon.py` is a project-specific DP-SCL implementation using
  standard PyTorch LSTM and multi-head attention modules.
- `src/models/common.py::SupConLoss`, the two-view training path, and the
  contrastive objective are based on supervised contrastive learning:
  Khosla et al., NeurIPS 2020, and the SupContrast PyTorch reference
  implementation: <https://github.com/HobbitLong/SupContrast>.
- `src/dataprocess/0_process_user_activity_logs.py`,
  `src/dataprocess/1_process_user_contextual_features.py`, and
  `src/dataprocess/2_table_2_numpy.py` are adapted from the corresponding
  CA-TFHN preprocessing scripts:
  <https://github.com/codeds27/CA-TFHN/tree/main/src/dataprocess>.
- `src/models/common.py::MySelfAttention` is adapted from CA-TFHN's
  `MySelfAttention` implementation:
  <https://github.com/codeds27/CA-TFHN/blob/main/src/models.py>.
- `src/dataprocess/oulad_preprocess.py` and
  `src/dataprocess/snap_preprocess.py` are DP-SCL dataset adapters.

Dataset references:

- XuetangX / KDD Cup 2015 MOOC data.
- OULAD: <https://analyse.kmi.open.ac.uk/open_dataset>.
- SNAP ACT-MOOC: <https://snap.stanford.edu/data/act-mooc.html>.

If adapted CA-TFHN source is redistributed publicly, verify upstream
permissions because the CA-TFHN repository did not expose a license file during
the attribution pass.
