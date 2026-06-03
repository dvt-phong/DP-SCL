# DP-SCL

DP-SCL is a PyTorch implementation for MOOC dropout prediction with a Siamese temporal encoder and supervised contrastive learning.

The main model uses two augmented views of each learner activity sequence, a shared LSTM + multi-head attention encoder, a projection head for supervised contrastive learning, and a binary classifier for dropout prediction.

## Repository Layout

- `train.py`: single-run training entry point.
- `train_experiment.py`: protocol runner for ML, DL, and DP-SCL experiments.
- `experiment_sensitivity_runner.py`: shared runner for lambda/tau sensitivity studies.
- `run_scripts/`: reproducible experiment scripts.
- `src/models/siamese.py`: DP-SCL model implementation.
- `src/mode_registry.py`: mode aliases and backend resolution.
- `baselines/`: ML and DL baselines.
- `docs/`: technical notes and method descriptions.

## Data

This repository does not include raw datasets, generated `.npz` files, graph files, checkpoints, logs, or experiment outputs.

Expected generated data layout:

```text
datastore/
  all_data_std.npz
  StrongClassmatesGraph.pkl
```

For temporal-only DP-SCL experiments, `all_data_std.npz` is sufficient. Graph-based legacy modes additionally require `StrongClassmatesGraph.pkl`.

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric installation can depend on CUDA version. If the default install fails, install the matching wheels from the official PyTorch and PyG instructions, then install the remaining requirements.

## Quick Start

Run DP-SCL for 15 epochs:

```bash
python train.py -indir . -outdir . -mode dp_scl -e 15 --dataset xuetangx
```

Run the protocol experiment for DP-SCL:

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

Run lambda and tau sensitivity with epoch-average summaries:

```bash
bash run_scripts/run_lambda_tau_sensitivity_5seed.sh --seeds 42 --force
```

Outputs are written under `results/` and `result_write/`, both ignored by git.

## Main Modes

- `dp_scl`: public DP-SCL mode, mapped to the `siamese_lstm_attn` backend.
- `tsn_supcon`: legacy alias kept for backward compatibility.
- `siamese_lstm_attn`: concrete backend implementation used by DP-SCL.

## Citation

If you use this code, cite the repository and the related paper/work describing DP-SCL.
