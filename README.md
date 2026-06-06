# DP-SCL

DP-SCL is a PyTorch implementation for MOOC dropout prediction using supervised contrastive learning. The model builds two augmented views of each learner activity sequence, encodes them with a shared LSTM + multi-head attention encoder, optimizes a supervised contrastive projection space, and predicts dropout with a binary classifier.

This repository has been trimmed to DP-SCL only. Alternative model families and non-DP-SCL training paths have been removed.

## Code Origin and Attribution

This project is a DP-SCL implementation for MOOC dropout prediction. It is not
based wholesale on CA-TFHN. CA-TFHN is cited only for components that were
copied, adapted, or directly referenced. The supervised contrastive learning
parts use the SupCon paper and SupContrast implementation as their primary
references.

Source notes by component:

- DP-SCL architecture and training objective follow the project manuscript:
  *Student Dropout Prediction in Online Courses Based on Supervised Contrastive
  Learning*, Doan Van Thanh Phong et al. The manuscript describes two augmented
  views, a shared LSTM + Multi-Head Attention + attentive pooling encoder, a
  projection head for SupCon loss, and a BCE classifier.
- `src/dataprocess/0_process_user_activity_logs.py`,
  `src/dataprocess/1_process_user_contextual_features.py`, and
  `src/dataprocess/2_table_2_numpy.py` are adapted from the corresponding
  CA-TFHN preprocessing files:
  <https://github.com/codeds27/CA-TFHN/tree/main/src/dataprocess>
- `src/models/common.py::MySelfAttention` is adapted from CA-TFHN's
  `MySelfAttention` in `src/models.py`:
  <https://github.com/codeds27/CA-TFHN/blob/main/src/models.py>
- CA-TFHN paper reference for the adapted MOOC preprocessing/attention context:
  Liang, G., Qian, Z., Wang, S., Hao, P. (2023). *MOOCs Dropout Prediction via
  Classmates Augmented Time-Flow Hybrid Network*. ICONIP 2023, pp. 405-416.
- `src/models/supcon.py::SupConEncoder` is a project-specific DP-SCL encoder.
  It is not copied from CA-TFHN or SupContrast. It uses standard PyTorch
  `nn.LSTM` and `nn.MultiheadAttention` modules to encode temporal learner
  activity sequences.
- `src/models/common.py::SupConLoss`,
  `src/models/supcon.py::SupConProjectionHead`, the two-view training path,
  and the contrastive objective are based primarily on supervised contrastive
  learning:
  <https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html>
  and the PyTorch reference implementation SupContrast:
  <https://github.com/HobbitLong/SupContrast>
- `src/dataprocess/oulad_preprocess.py` and
  `src/dataprocess/snap_preprocess.py` are DP-SCL dataset adapters, not copied
  from CA-TFHN. Dataset references:
  OULAD <https://analyse.kmi.open.ac.uk/open_dataset> and SNAP ACT-MOOC
  <https://snap.stanford.edu/data/act-mooc.html>.

Note: if redistributing adapted CA-TFHN source publicly, verify upstream
permissions because the CA-TFHN repository did not expose a license file during
this attribution pass.

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
