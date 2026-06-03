# DP-SCL

DP-SCL is a PyTorch implementation for MOOC dropout prediction using supervised contrastive learning (SupCon). The proposed model builds two augmented views of each learner activity sequence, encodes them with a shared temporal encoder, optimizes a SupCon projection space, and predicts dropout with a binary classifier.

## Repository Layout

- `train.py`: single-run training entry point.
- `train_experiment.py`: protocol runner for ML baselines, DL baselines, and DP-SCL.
- `train_ml_baseline.py`: standalone ML baseline runner.
- `summarize_baseline_results.py`: helper for summarizing baseline result files.
- `run_scripts/run_dp_scl.py`: convenience wrapper for running DP-SCL through `train_experiment.py`.
- `src/models/supcon.py`: DP-SCL/SupCon model implementation.
- `src/models/common.py`: shared layers, augmentation, and `SupConLoss`.
- `src/mode_registry.py`: supported modes and alias resolution.
- `src/dataset_config.py`: dataset-specific filenames and tensor dimensions.
- `src/dataprocess/`: preprocessing scripts.
- `src/graphgeneration/` and `src/linkprediction/`: graph construction utilities.
- `baselines/`: ML and DL baseline implementations.

## Data

Raw datasets, generated `.npz` files, graph files, checkpoints, logs, and experiment outputs are not included.

Generated data is expected under:

```text
datastore/
  all_data_std.npz
  StrongClassmatesGraph.pkl
```

Dataset-specific filenames:

| Dataset | Temporal file | Graph file |
| --- | --- | --- |
| `xuetangx` | `all_data_std.npz` | `StrongClassmatesGraph.pkl` |
| `oulad` | `oulad_data_std.npz` | `oulad_StrongClassmatesGraph.pkl` |
| `snap` | `snap_data_std.npz` | `snap_StrongClassmatesGraph.pkl` |

DP-SCL only needs the temporal `.npz` file. Graph files are only required for graph-enhanced modes.

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

PyTorch and PyTorch Geometric wheels depend on the local CUDA setup. If installation fails, install the matching PyTorch/PyG wheels first, then rerun the requirements install.

## Quick Start

Run one DP-SCL training job:

```bash
python train.py -indir . -outdir . -mode dp_scl --dataset xuetangx -e 15
```

Run the DP-SCL protocol experiment:

```bash
python train_experiment.py \
  -indir . \
  -outdir . \
  --dataset xuetangx \
  --models proposed \
  --proposed-name DP-SCL \
  --proposed-mode dp_scl \
  --seeds 42 \
  --max-epochs 15
```

Equivalent convenience wrapper:

```bash
python run_scripts/run_dp_scl.py -indir . -outdir . --dataset xuetangx --seeds 42 --max-epochs 15
```

Run all protocol groups:

```bash
python train_experiment.py -indir . -outdir . --dataset xuetangx --models all
```

Outputs are written under `results/`.

## Main Modes

- `dp_scl`: public DP-SCL mode. Internally resolves to `supcon_lstm_attn`.
- `tsn_supcon`: legacy public alias for DP-SCL.
- `supcon_lstm_attn`: concrete SupCon backend used by DP-SCL.
- `supcon_lstm`, `supcon_bilstm`, `supcon_lstm_mha`, `supcon_bilstm_attn`, `supcon_lstm_sa`, `supcon_bilstm_sa`: SupCon encoder variants.
- `simclr_*` and `byol_*`: self-supervised contrastive alternatives kept in the mode registry.
- `dl_*` and `ml_*`: baseline families used by `train_experiment.py`.

## DP-SCL Objective

During training, DP-SCL minimizes:

```text
loss = BCEWithLogits(logits, label) + lambda_con * SupConLoss(z1, z2, label)
```

Default proposed settings:

```text
lambda_con = 0.1
temperature = 0.07
mask_ratio = 0.15
noise_std = 0.05
hidden_size = 128
```

During evaluation, augmentation and projection outputs are disabled; the model returns dropout logits only.

## Citation

If you use this code, cite the repository and the related work describing DP-SCL.
