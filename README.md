# DP-SCL

DP-SCL is a PyTorch implementation for MOOC dropout prediction using supervised contrastive learning. The model builds two augmented views of each learner activity sequence, encodes them with a shared LSTM + multi-head attention encoder, optimizes a supervised contrastive projection space, and predicts dropout with a binary classifier.

This repository has been trimmed to DP-SCL only. Alternative model families and non-DP-SCL training paths have been removed.

## Repository Layout

- `train_experiment.py`: DP-SCL protocol runner for one or more seeds.
- `train.py`: compatibility wrapper for a single DP-SCL run.
- `run_scripts/run_dp_scl.py`: convenience wrapper for `train_experiment.py`.
- `src/models/supcon.py`: DP-SCL model.
- `src/models/common.py`: DP-SCL layers, augmentation, and `SupConLoss`.
- `src/mode_registry.py`: DP-SCL mode resolution.
- `src/dataset_config.py`: dataset-specific temporal filenames and tensor dimensions.
- `src/dataprocess/`: temporal preprocessing scripts.

## Data

Raw datasets, generated `.npz` files, checkpoints, logs, and experiment outputs are not included.

Generated temporal data is expected under:

```text
datastore/
  all_data_std.npz
```

Dataset-specific temporal filenames:

| Dataset | Temporal file |
| --- | --- |
| `xuetangx` | `all_data_std.npz` |
| `oulad` | `oulad_data_std.npz` |
| `snap` | `snap_data_std.npz` |

The temporal `.npz` format is:

```text
t_data:  (N_train, week_count, days_per_week, activity_num)
t_label: (N_train,)
v_data:  (N_test, week_count, days_per_week, activity_num)
v_label: (N_test,)
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

Run DP-SCL with the default five seeds:

```bash
python run_scripts/run_dp_scl.py -indir . -outdir . --dataset xuetangx --max-epochs 15
```

Run DP-SCL with explicit seeds:

```bash
python train_experiment.py -indir . -outdir . --dataset xuetangx --seeds 1 11 111 1111 11111 --max-epochs 15
```

Compatibility single-run style:

```bash
python train.py -indir . -outdir . -mode dp_scl --dataset xuetangx -r 42 -e 15
```

Outputs are written under `results/dp_scl_<timestamp>/`:

- `config.json`
- `splits/seed_*_{train,val,test}.npy`
- `checkpoints/dp_scl_seed_*.pt`
- `per_seed_results.csv`
- `epoch_history.csv`
- `summary_results.csv`
- `report.txt`

## DP-SCL Objective

During training, DP-SCL minimizes:

```text
loss = BCEWithLogits(logits, label) + lambda_con * SupConLoss(z1, z2, label)
```

Default settings:

```text
lambda_con = 0.1
temperature = 0.07
mask_ratio = 0.15
noise_std = 0.05
hidden_size = 128
```

During evaluation, augmentation and projection outputs are disabled; the model returns dropout logits only.
