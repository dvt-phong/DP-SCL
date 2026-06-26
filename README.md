# DP-SCL

DP-SCL is a model for predicting student dropout in MOOC courses using
Supervised Contrastive Learning. Each learner-course record is represented as a
temporal activity sequence; the model learns a behavior representation and then
classifies the dropout probability.

## Model

DP-SCL has three main components:

- **Temporal encoder**: reshapes the activity tensor
  `(week_count, days_per_week, activity_num)` into a temporal sequence, then
  encodes it with LSTM and Multi-Head Attention.
- **Projection head**: creates embeddings for Supervised Contrastive Loss.
  During training, each sample is augmented into two views using time masking,
  feature masking, and noise.
- **Classifier**: receives the encoder representation and predicts dropout with
  `BCEWithLogitsLoss`.

Training loss:

```text
loss = BCEWithLogitsLoss(logits, label)
     + lambda_con * SupConLoss([z1, z2], label)
```

The repository supports three datasets: `xuetangx`, `oulad`, and `snap`.

## Main Structure

```text
.
|-- train_experiment.py        # main training/evaluation script
|-- train.py                   # compatibility wrapper for the old CLI
|-- run_scripts/run_dp_scl.py  # convenience launcher
|-- src/models/                # DP-SCL model, SupCon loss, augmentation
|-- src/dataprocess/           # data preprocessing scripts
|-- datastore/                 # .npz data files
|-- requirements.txt
`-- README.md
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches the CUDA version on your
machine before running experiments.

## Data

Place the `.npz` files in the `datastore/` directory:

| Dataset | File |
| --- | --- |
| `xuetangx` | `datastore/all_data_std.npz` |
| `oulad` | `datastore/oulad_data_std.npz` |
| `snap` | `datastore/snap_data_std.npz` |

Each file must contain:

```text
t_data, t_label
v_data, v_label
```

The training script combines the original train/test arrays, then creates
stratified splits for each seed using the default ratio
`60% train / 10% validation / 30% test`.

## Usage

Quick run on XuetangX:

```bash
python run_scripts/run_dp_scl.py -indir . -outdir . --dataset xuetangx --max-epochs 15
```

Run the main training script directly:

```bash
python train_experiment.py -indir . -outdir . --dataset xuetangx --seeds 1 11 111 1111 11111 --max-epochs 15
```

Run on another dataset:

```bash
python train_experiment.py -indir . -outdir . --dataset oulad --max-epochs 15
python train_experiment.py -indir . -outdir . --dataset snap --max-epochs 15
```

Common arguments:

```text
--batch-size      batch size, default 256
--lr              learning rate, default 1e-4
--hidden-size     encoder hidden size, default 128
--lambda-con      SupCon loss weight, default 0.1
--temperature     SupCon temperature, default 0.07
--max-epochs      maximum number of epochs
--patience        early stopping patience
```

## Outputs

Each run creates a directory:

```text
results/dp_scl_<timestamp>/
```

Main files:

```text
config.json
checkpoints/
splits/
per_seed_results.csv
epoch_history.csv
summary_results.csv
report.txt
```

Main metrics include AUC, accuracy, precision, recall, and F1. The classification
threshold is selected on the validation split by maximizing F1, then applied to
the test split.

## Quick Test

```bash
python -m pytest src/tests/test_dp_scl.py
```
