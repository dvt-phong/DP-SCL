import argparse
import os
import pickle
import time
import torch
import numpy as np
import random

# ============================================================
# Bước 0 — Fix seed toàn bộ để reproducible results
# ============================================================
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch import optim
import torch.nn.functional as F
from src.models import *
from src.mode_registry import (
    ALL_MODES, GRAPH_MODES, NO_GRAPH_MODES,
    SUPCON_MODES, SIMCLR_MODES, BYOL_MODES, DL_BASELINE_MODES, CL_MODES,
    is_graph_mode, is_no_graph_mode, get_framework, describe_mode, resolve_backend_mode,
)
from src.data_validator import validate_data_for_mode, get_regeneration_command
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score

myparser = argparse.ArgumentParser(description='optional parameters')
myparser.add_argument('-indir', type=str, help='input dir (default: current dir)', default='.')
myparser.add_argument('-outdir', type=str, help='output dir (default: current dir)', default='.')
myparser.add_argument('-e', type=int, default=15, help='epoch (default: 15)')
myparser.add_argument('-r', type=int, default=0, help='random seed (default: 0)')
myparser.add_argument('-lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
myparser.add_argument('-mode', type=str, default='default',
                      choices=sorted(ALL_MODES),
                      help='model mode (default: default)')
myparser.add_argument('--contrastive', action='store_true', default=False,
                      help='Enable Supervised Contrastive Learning (default: off)')
myparser.add_argument('--lambda-con', type=float, default=0.1,
                      help='Weight for contrastive loss: Total = BCE + lambda * SupCon (default: 0.1)')
# --- Hyperparameter tuning args (Framework 1 & 2) ---
myparser.add_argument('--temperature', type=float, default=None,
                      help='Contrastive loss temperature τ (SupCon default: 0.07, SimCLR default: 0.1)')
myparser.add_argument('--mask-ratio', type=float, default=None,
                      help='Augmentation mask ratio for time & feature masking (default: 0.15)')
myparser.add_argument('--noise-std', type=float, default=None,
                      help='Augmentation Gaussian noise σ (default: 0.05)')
myparser.add_argument('--hidden-size', type=int, default=None,
                      help='Encoder hidden size / output dim (default: 128)')
myparser.add_argument('--num-layers', type=int, default=None,
                      help='Number of LSTM/BiLSTM layers in encoder (default: 1)')
myparser.add_argument('--cls-layers', type=int, default=1,
                      help='Number of hidden layers in classifier head (default: 1, max recommended: 2)')
myparser.add_argument('--batch-size', type=int, default=256,
                      help='Training batch size (default: 256); test batch = batch_size // 2')
myparser.add_argument('--dataset', type=str, default='xuetangx',
                      choices=['xuetangx', 'oulad', 'snap'],
                      help='Dataset to use (default: xuetangx)')
myparser.add_argument('--sampling', type=str, default='none',
                      choices=['none', 'oversample', 'undersample',
                               'smote', 'adasyn', 'borderline_smote',
                               'smote_tomek', 'smote_enn',
                               'weighted_loss', 'focal_loss'],
                      help='Sampling strategy for class imbalance (default: none)')
# --- Action Weighting & Early Prediction ---
myparser.add_argument('--action-weight', action='store_true', default=False,
                      help='Enable learnable action type importance weighting')
myparser.add_argument('--early-prediction', action='store_true', default=False,
                      help='Enable curriculum masking for early prediction (eval per-week)')
myparser.add_argument('--early-min-weeks', type=int, default=2,
                      help='Minimum weeks to keep during curriculum training (default: 2)')

args = myparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dir = os.path.abspath(os.path.expanduser(str(args.indir)))
output_dir = os.path.abspath(os.path.expanduser(str(args.outdir)))
epoch_num = args.e
learning_rate = args.lr
mode = args.mode
backend_mode = resolve_backend_mode(mode)
use_contrastive = args.contrastive
lambda_con = args.lambda_con
if args.mode == 'supcon_lstm_attn_lambda0':
    lambda_con = 0.0
    args.lambda_con = 0.0
elif args.mode in {'dp_scl', 'tsn_supcon'}:
    lambda_con = 0.1
    args.lambda_con = 0.1
    if args.temperature is None:
        args.temperature = 0.07
dataset_name = args.dataset

# --- Load dataset config ---
from src.dataset_config import get_dataset_config
ds_config = get_dataset_config(dataset_name)
print(f"=== Dataset: {ds_config['name']} (activity_num={ds_config['activity_num']}) ===")

# --- Mode classification (from centralized registry) ---
print(f"=== Running DP-SCL with mode: {mode} ===")
print(f"=== {describe_mode(mode)} ===")
if backend_mode != mode:
    print(f"=== Backend mode: {backend_mode} ===")
is_supcon  = mode in SUPCON_MODES
is_simclr   = mode in SIMCLR_MODES
is_byol     = mode in BYOL_MODES
is_dl_baseline = mode in DL_BASELINE_MODES
is_cl       = mode in CL_MODES
_uses_graph = is_graph_mode(mode)     # GRAPH branch
_no_graph   = is_no_graph_mode(mode)  # NO-GRAPH branch
if is_supcon:
    print(f"=== Version 2.1: SupCon Network (encoder: {mode.replace('supcon_', '')}) ===")
    print(f"=== SupCon Contrastive Learning: ON (λ={lambda_con}) ===")
elif is_simclr:
    print(f"=== Framework 2A: SimCLR (encoder: {mode.replace('simclr_', '')}, λ={lambda_con}) ===")
elif is_byol:
    print(f"=== Framework 2B: BYOL   (encoder: {mode.replace('byol_', '')},   λ={lambda_con}) ===")
elif is_dl_baseline:
    print(f"=== Baseline DL: {mode} ===")
elif use_contrastive:
    print(f"=== Supervised Contrastive Learning: ON (λ={lambda_con}) ===")

# --- Data Validation: kiểm tra data tồn tại + đúng cấu trúc ---
_datastore_dir = os.path.join(input_dir, 'datastore')
print(f"\n--- Data Validation ({('GRAPH' if _uses_graph else 'NO-GRAPH')} branch) ---")
_data_valid, _data_results = validate_data_for_mode(mode, dataset_name, _datastore_dir)
for _key, (_v, _msg) in _data_results.items():
    print(f"  [{_key.upper():>5}] {_msg}")
if not _data_valid:
    print(f"\n  ❌ Data chưa sẵn sàng! Cần chạy:")
    for _desc, _cmd in get_regeneration_command(mode, dataset_name, _datastore_dir, _data_results):
        print(f"    {_desc}")
        print(f"      $ {_cmd}")
    raise SystemExit(1)
print(f"  ✅ Data validated — bắt đầu training...\n")

# === Timing: bắt đầu đo ===
_t_pipeline_start = time.time()
_t_data_start = time.time()

if _no_graph:
    # === no_graph / supcon / CL: Load temporal data directly from numpy (no graph dependency) ===
    from torch.utils.data import DataLoader, TensorDataset

    npz_path = os.path.join(input_dir, 'datastore', ds_config['npz_filename'])
    print(f"Loading temporal data from: {npz_path}")
    data = np.load(npz_path)

    # t_data: (N_train, W, D, F), t_label: (N_train,)
    # v_data: (N_test, W, D, F),  v_label: (N_test,)
    train_seq = torch.from_numpy(data['t_data']).to(torch.float)
    train_labels = torch.from_numpy(data['t_label']).to(torch.float)
    test_seq = torch.from_numpy(data['v_data']).to(torch.float)
    test_labels = torch.from_numpy(data['v_label']).to(torch.float)

    # Flatten weekly data: (N, W, D, F) → (N, W*D*F) to match seq_feat format
    N_train = train_seq.shape[0]
    N_test = test_seq.shape[0]
    _flat_dim = train_seq.shape[1] * train_seq.shape[2] * train_seq.shape[3]
    train_seq_flat = train_seq.view(N_train, -1)   # (N, flat_dim)
    test_seq_flat = test_seq.view(N_test, -1)       # (N, flat_dim)
    print(f"  Data shape: {list(train_seq.shape)} → flat {_flat_dim}")

    train_dataset = TensorDataset(train_seq_flat, train_labels)
    test_dataset = TensorDataset(test_seq_flat, test_labels)

    # --- Sampling strategy ---
    n_pos = int(train_labels.sum().item())
    n_neg = N_train - n_pos
    print(f"  Class distribution: pos(dropout)={n_pos} ({n_pos/N_train:.3f}), neg={n_neg} ({n_neg/N_train:.3f})")

    if args.sampling in ('oversample', 'undersample'):
        from torch.utils.data import WeightedRandomSampler
        if args.sampling == 'oversample':
            w_pos = 1.0 / n_pos if n_pos > 0 else 1.0
            w_neg = 1.0 / n_neg if n_neg > 0 else 1.0
            sample_weights = torch.tensor([w_pos if l == 1 else w_neg for l in train_labels])
            num_samples = 2 * max(n_pos, n_neg)
        else:  # undersample
            w_pos = 1.0 / n_pos if n_pos > 0 else 1.0
            w_neg = 1.0 / n_neg if n_neg > 0 else 1.0
            sample_weights = torch.tensor([w_pos if l == 1 else w_neg for l in train_labels])
            num_samples = 2 * min(n_pos, n_neg)
        sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        print(f"  Sampling: {args.sampling} → {num_samples} samples/epoch")
    elif args.sampling in ('smote', 'adasyn', 'borderline_smote', 'smote_tomek', 'smote_enn'):
        try:
            import importlib
            _RESAMPLERS = {
                'smote':            ('imblearn.over_sampling', 'SMOTE'),
                'adasyn':           ('imblearn.over_sampling', 'ADASYN'),
                'borderline_smote': ('imblearn.over_sampling', 'BorderlineSMOTE'),
                'smote_tomek':      ('imblearn.combine',       'SMOTETomek'),
                'smote_enn':        ('imblearn.combine',       'SMOTEENN'),
            }
            _mod, _cls = _RESAMPLERS[args.sampling]
            resampler = getattr(importlib.import_module(_mod), _cls)(random_state=42)
        except ModuleNotFoundError:
            print("  ❌ 'imbalanced-learn' chưa được cài. Chạy: pip install imbalanced-learn")
            raise SystemExit(1)
        X_resampled, y_resampled = resampler.fit_resample(
            train_seq_flat.numpy(), train_labels.numpy().astype(int)
        )
        train_seq_flat = torch.from_numpy(X_resampled).to(torch.float)
        train_labels   = torch.from_numpy(y_resampled).to(torch.float)
        N_train    = len(train_labels)
        n_pos_new  = int(train_labels.sum().item())
        n_neg_new  = N_train - n_pos_new
        train_dataset = TensorDataset(train_seq_flat, train_labels)
        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"  Sampling: {args.sampling.upper()} → {N_train} samples (pos={n_pos_new}, neg={n_neg_new})")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        if args.sampling != 'none':
            print(f"  Sampling: {args.sampling} (applied at loss level)")

    test_loader = DataLoader(test_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False)

    print(f"  Train samples: {N_train}, Test samples: {N_test}")
    print(f"  Seq feat shape: {train_seq_flat.shape}")
else:
    # === All other modes: Load from graph file ===
    # Try NeighborLoader first; fallback to FullGraphBatchLoader if torch-sparse unavailable
    graph_path = os.path.join(input_dir, 'datastore', ds_config['graph_filename'])
    print(f"Loading graph from: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    graph = graph.to(device)

    try:
        from torch_geometric.loader import NeighborLoader
        train_loader = NeighborLoader(
            data=graph, num_neighbors=[8, 4],
            input_nodes=graph.train_mask,
            batch_size=args.batch_size, shuffle=True
        )
        test_loader = NeighborLoader(
            data=graph, num_neighbors=[8, 4],
            input_nodes=graph.test_mask,
            batch_size=max(1, args.batch_size // 2), shuffle=True
        )
        # Quick test: trigger actual sampling to catch ImportError from NeighborSampler
        _test_iter = iter(train_loader)
        _test_batch = next(_test_iter)
        del _test_iter, _test_batch
        # Recreate train_loader since we consumed one batch
        train_loader = NeighborLoader(
            data=graph, num_neighbors=[8, 4],
            input_nodes=graph.train_mask,
            batch_size=args.batch_size, shuffle=True
        )
        print("  Using NeighborLoader (pyg-lib/torch-sparse)")
    except (ImportError, Exception) as e:
        print(f"  NeighborLoader unavailable ({type(e).__name__}), using FullGraphBatchLoader")
        # Full-graph mini-batch: every batch sees all edges, but only a subset of nodes
        from torch.utils.data import DataLoader, TensorDataset

        class FullGraphBatchLoader:
            """Mini-batch loader that wraps full graph — each batch is a sub_graph dict."""
            def __init__(self, graph, node_mask, batch_size, shuffle=True):
                self.graph = graph
                self.node_indices = node_mask
                self.batch_size = batch_size
                self.shuffle = shuffle

            def __iter__(self):
                indices = self.node_indices.clone()
                if self.shuffle:
                    perm = torch.randperm(len(indices), device=indices.device)
                    indices = indices[perm]
                N = self.graph.labels.shape[0]
                all_nodes = torch.arange(N, device=indices.device)
                for start in range(0, len(indices), self.batch_size):
                    batch_nodes = indices[start:start + self.batch_size]
                    bs = len(batch_nodes)
                    # NeighborLoader format: batch nodes first, then all other nodes
                    # This lets GNN do message passing over full graph, model takes [:bs]
                    batch_set = set(batch_nodes.cpu().tolist())
                    other_nodes = torch.tensor([n for n in range(N) if n not in batch_set],
                                               device=indices.device, dtype=torch.long)
                    ordered = torch.cat([batch_nodes, other_nodes])
                    # Build remapping: old_id → new_id
                    remap = torch.empty(N, dtype=torch.long, device=indices.device)
                    remap[ordered] = torch.arange(N, device=indices.device)
                    # Remap edge_index
                    edge_index = remap[self.graph.edge_index]
                    sub_graph = {
                        'batch_size': bs,
                        'labels': self.graph.labels[ordered],
                        'seq_feat': self.graph.seq_feat[ordered],
                        'edge_index': edge_index,
                    }
                    if hasattr(self.graph, 'org_context'):
                        sub_graph['org_context'] = self.graph.org_context[ordered]
                    if hasattr(self.graph, 'enhanced_context'):
                        sub_graph['enhanced_context'] = self.graph.enhanced_context[ordered]
                    yield sub_graph

            def __len__(self):
                return (len(self.node_indices) + self.batch_size - 1) // self.batch_size

        train_loader = FullGraphBatchLoader(graph, graph.train_mask, args.batch_size, shuffle=True)
        test_loader = FullGraphBatchLoader(graph, graph.test_mask, max(1, args.batch_size // 2), shuffle=False)

# set the hyper parameters
param_dict = dict({
    # Constant — from dataset config
    'activity_num': ds_config['activity_num'], 'sta_day': ds_config['sta_day'],
    'week_count': ds_config['week_count'], 'select_count': ds_config['week_count'],
    # Context-Embedding: enhanced_context dims differ by dataset
    # XuetangX: 32 (user_feat + link_predict_feat), SNAP/OULAD: 7 (same as org_context)
    'org_context_feat_len': 7, 'enhanced_context_feat_len': 32 if dataset_name == 'xuetangx' else 7, 'context_each_embed': 16, 'context_all_len': 16,
    # GraphSage / GAT (shared dimensions)
    'input_features': 16, 'hidden_features': 32, 'output_features': 16,
    # LSTM of first block in TFHN
    # Manual preprocess output: (D+1)×(F+1) = 8×(F+1). XuetangX:184, OULAD:168, SNAP:56
    'lstm_input_features': 8 * (ds_config['activity_num'] + 1), 'lstm_hidden_features': 128, 'lstm_hidden_num_layers': 1,
    # Self-Attention of first block in TFHN
    'num_attention_heads': 1, 'attention_features': 64,
    # LSTM2 of second block in TFHN
    'l2_input_features': 64, 'l2_hidden_features': 32, 'l2_hidden_num_layers': 1,
    # Self-Attention of second block in TFHN
    's2_num_attention_heads': 1, 's2_attention_features': 16,
    # Weighted-Sum
    'ws_num_attention_heads': 1, 'ws_input_features': 32, 'ws_attention_features': 16,
    # DNN
    'dnn_input_f1': 16, 'dnn_hidden_f1': 16, 'dnn_hidden_f2': 8, 'dnn_hidden_f3': 4, 'dnn_output': 1
})

# Add mode-specific hyperparameters
if mode in ('cnn', 'cnn_gat'):
    param_dict.update({
        'cnn_in_channels': 7,        # days_per_week (sta_day / week_count = 35/5)
        'cnn_out_channels_1': 32,
        'cnn_out_channels_2': 64,
        'cnn_kernel_size': 3,
        'cnn_fc_output': 128,        # CNN output dim → becomes LSTM1 input
    })

if mode in ('gat', 'cnn_gat', 'mba_cnn_gat'):
    param_dict.update({
        'gat_heads': 4,              # number of attention heads in GAT layer 1
        'gat_dropout': 0.3,          # dropout rate for GAT
    })

if mode == 'cnn2d':
    param_dict.update({
        'cnn2d_out_channels_1': 32,
        'cnn2d_out_channels_2': 64,
        'cnn2d_kernel_size': 3,
        'cnn2d_fc_output': 128,      # CNN2D output dim → becomes LSTM1 input
    })

if mode == 'cross_attn':
    param_dict.update({
        'ca_num_heads': 4,           # number of cross-attention heads
        'ca_output_dim': 16,         # output dim (match dnn_input_f1)
        'ca_ffn_dim': 32,            # FFN hidden dim
    })

if mode in ('mba_cnn', 'mba_cnn_gat'):
    param_dict.update({
        'mba_cnn_temporal_channels_1': 32,   # Temporal branch conv1 output channels
        'mba_cnn_temporal_channels_2': 64,   # Temporal branch conv2 output channels
        'mba_cnn_daily_channels': 32,        # Daily branch output channels
        'mba_cnn_weekly_channels': 32,       # Weekly branch output channels
        'mba_cnn_fc_hidden': 256,            # FC hidden dim after concat
        'mba_cnn_output': 128,               # Final output dim → becomes LSTM1 input
        'mba_cnn_dropout': 0.3,              # Dropout rate in FC layers
    })

if mode == 'bilstm_cnn':
    param_dict.update({
        # CNN params — activities as channels, days as sequence
        'cnn_in_channels': 22,           # activity_num (channels)
        'cnn_out_channels_1': 64,
        'cnn_out_channels_2': 128,
        'cnn_kernel_size': 3,
        'cnn_fc_output': 128,            # CNN output dim → becomes BiLSTM1 input
        # BiLSTM1: input=128 (CNN output), hidden=64 → output=64×2=128 (matches LSTM1)
        'lstm_input_features': 256,      # 128 orig + 128 temporal diff
        'lstm_hidden_features': 64,
        # SA1: input=128 (BiLSTM1 output), attention_features=64 → GIỮA NGUYÊN paper
        # (already default: num_attention_heads=1, attention_features=64)
        # BiLSTM2: input=64 (SA1 output), hidden=16 → output=16×2=32 (matches LSTM2)
        'l2_hidden_features': 16,
        # SA2: input=32 (BiLSTM2 output), s2_attention_features=16 → GIỮA NGUYÊN paper
        # (already default: s2_num_attention_heads=1, s2_attention_features=16)
    })

if mode == 'bilstm_mha':
    param_dict.update({
        # CNN — giống bilstm_cnn
        'cnn_in_channels': 22,
        'cnn_out_channels_1': 64,
        'cnn_out_channels_2': 128,
        'cnn_kernel_size': 3,
        'cnn_fc_output': 128,
        # BiLSTM1: input=128, hidden=64 → out=128
        'lstm_input_features': 128,
        'lstm_hidden_features': 64,
        'lstm_hidden_num_layers': 1,
        # BiLSTM2: input=128 (MHA giữ dim), hidden=32 → out=64
        'l2_input_features': 128,      # ← khác bilstm_cnn (bilstm_cnn là 64)
        'l2_hidden_features': 32,
        'l2_hidden_num_layers': 1,
        # MHA
        'mha_num_heads': 4,
    })

if mode == 'bilstm_cross':
    param_dict.update({
        'cnn_in_channels': 22,
        'cnn_out_channels_1': 64,
        'cnn_out_channels_2': 128,
        'cnn_kernel_size': 3,
        'cnn_fc_output': 128,
        'lstm_input_features': 128,
        'lstm_hidden_features': 64,
        'lstm_hidden_num_layers': 1,
        'l2_input_features': 128,
        'l2_hidden_features': 32,
        'l2_hidden_num_layers': 1,
        'mha_num_heads': 4,
    })

if mode == 'bilstm_graph':
    param_dict.update({
        # CNN — activities as channels, days as sequence
        'cnn_in_channels': 22,
        'cnn_out_channels_1': 64,
        'cnn_out_channels_2': 128,
        'cnn_kernel_size': 3,
        'cnn_fc_output': 128,
        # BiLSTM1: in=128 (CNN output), hid=64 → out=128
        'lstm_input_features': 128,
        'lstm_hidden_features': 64,
        'lstm_hidden_num_layers': 1,
        # SA1: input=128 (BiLSTM1 output)
        'attention_features': 64,
        # BiLSTM2: in=64 (SA1 output), hid=16 → out=32
        'l2_input_features': 64,
        'l2_hidden_features': 16,
        'l2_hidden_num_layers': 1,
        # SA2: input=32 (BiLSTM2 output)
        's2_attention_features': 16,
        # Fusion: 16(graph) + 16(temporal) = 32
        'ws_input_features': 32,
        'ws_attention_features': 16,
    })

if mode == 'mba_bilstm':
    param_dict.update({
        'cnn_in_channels': 7,                  # days_per_week (cho MBA-CNN 2D)
        'mba_cnn_temporal_channels_1': 32,
        'mba_cnn_temporal_channels_2': 64,
        'mba_cnn_daily_channels': 32,
        'mba_cnn_weekly_channels': 32,
        'mba_cnn_fc_hidden': 256,
        'mba_cnn_output': 128,
        'mba_cnn_dropout': 0.3,
        # BiLSTM1: in=128, hid=64 → out=128
        'lstm_input_features': 128,
        'lstm_hidden_features': 64,
        'lstm_hidden_num_layers': 1,
        # SA1: input=128
        'attention_features': 64,
        # BiLSTM2: in=64, hid=16 → out=32
        'l2_input_features': 64,
        'l2_hidden_features': 16,
        'l2_hidden_num_layers': 1,
        # SA2: input=32
        's2_attention_features': 16,
    })

if mode == 'cnn_only':
    param_dict.update({
        # CNN 1D: activities (22) as channels, days (7) as sequence
        'cnn_in_channels': 22,
        'cnn_out_channels_1': 64,
        'cnn_out_channels_2': 128,
        'cnn_kernel_size': 3,
        'cnn_fc_output': 128,            # CNN output → becomes LSTM1 input
    })

if mode == 'mba_only':
    param_dict.update({
        # MBA-CNN: 3-branch asymmetric 2D-CNN, input (1, 7, 22)
        'cnn_in_channels': 7,                  # days_per_week
        'mba_cnn_temporal_channels_1': 32,
        'mba_cnn_temporal_channels_2': 64,
        'mba_cnn_daily_channels': 32,
        'mba_cnn_weekly_channels': 32,
        'mba_cnn_fc_hidden': 256,
        'mba_cnn_output': 128,                 # MBA-CNN output → becomes LSTM1 input
        'mba_cnn_dropout': 0.3,
    })

if mode == 'cnn_day':
    param_dict.update({
        # CNN 1D: days (7) as channels, activities (22) as sequence
        'cnn_in_channels': 7,
        'cnn_out_channels_1': 64,
        'cnn_out_channels_2': 128,
        'cnn_kernel_size': 3,
        'cnn_fc_output': 128,            # CNN output → becomes LSTM1 input
    })

if mode == 'bilstm_day':
    param_dict.update({
        # CNN 1D: days (7) as channels, activities (22) as sequence
        'cnn_in_channels': 7,
        'cnn_out_channels_1': 64,
        'cnn_out_channels_2': 128,
        'cnn_kernel_size': 3,
        'cnn_fc_output': 128,
        # BiLSTM1: in=128, hid=64 → out=128
        'lstm_input_features': 128,
        'lstm_hidden_features': 64,
        'lstm_hidden_num_layers': 1,
        # SA1: input=128
        'attention_features': 64,
        # BiLSTM2: in=64, hid=16 → out=32
        'l2_input_features': 64,
        'l2_hidden_features': 16,
        'l2_hidden_num_layers': 1,
        # SA2: input=32
        's2_attention_features': 16,
    })

# --- Version 2.1: SupCon Network ---
if is_supcon:
    _hid = args.hidden_size if args.hidden_size is not None else 128
    param_dict.update({
        'supcon_hidden_size': _hid,                                                      # encoder output dim
        'supcon_proj_dim': _hid,                                                         # projection head output dim (match hidden)
        'supcon_temperature': args.temperature if args.temperature is not None else 0.07, # SupCon temperature τ
        'supcon_mask_ratio': args.mask_ratio if args.mask_ratio is not None else 0.15,    # augmentation: time & feature mask ratio
        'supcon_noise_std': args.noise_std if args.noise_std is not None else 0.05,       # augmentation: Gaussian noise σ
        'supcon_attn_heads': 4,                                                           # attention heads (lstm_attn, bilstm_attn)
        'supcon_cls_dropout': 0.3,                                                        # classifier dropout
        'supcon_num_layers': args.num_layers if args.num_layers is not None else 1,       # LSTM/BiLSTM num layers in encoder
        'supcon_cls_hidden_layers': args.cls_layers,                                      # hidden layers in classifier head
        'use_action_weight': args.action_weight,                                            # learnable action importance
        'use_early_prediction': args.early_prediction,                                     # curriculum masking
        'early_min_weeks': args.early_min_weeks,                                           # min weeks for curriculum
    })
    _extra = []
    if args.action_weight: _extra.append('ActionWeight=ON')
    if args.early_prediction: _extra.append(f'EarlyPred=ON(min={args.early_min_weeks}w)')
    print(f"  [SupCon HP] hidden={param_dict['supcon_hidden_size']}, τ={param_dict['supcon_temperature']}, "
          f"mask={param_dict['supcon_mask_ratio']}, noise={param_dict['supcon_noise_std']}, "
          f"enc_layers={param_dict['supcon_num_layers']}, cls_layers={param_dict['supcon_cls_hidden_layers']}, λ={lambda_con}")
    if _extra: print(f"  [SupCon++] {', '.join(_extra)}")

# --- Framework 2: SimCLR + BYOL ---
if is_cl:
    _hid = args.hidden_size if args.hidden_size is not None else 128
    param_dict.update({
        'cl_hidden_size':  _hid,                                                          # encoder output dim
        'cl_proj_dim':     _hid,                                                          # projection head output dim (match hidden)
        'cl_temperature':  args.temperature if args.temperature is not None else 0.1,     # NT-Xent temperature τ (SimCLR)
        'cl_momentum':     0.996,                                                          # BYOL EMA momentum m
        'cl_mask_ratio':   args.mask_ratio if args.mask_ratio is not None else 0.15,      # augmentation: time & feature mask ratio
        'cl_noise_std':    args.noise_std if args.noise_std is not None else 0.05,        # augmentation: Gaussian noise σ
        'cl_attn_heads':   4,                                                              # attention heads (attn variants)
        'cl_cls_dropout':  0.3,                                                            # classifier dropout
        'cl_num_layers':   args.num_layers if args.num_layers is not None else 1,          # LSTM/BiLSTM num layers in encoder
        'cl_cls_hidden_layers': args.cls_layers,                                           # hidden layers in classifier head
        'use_action_weight': args.action_weight,                                            # learnable action importance
        'use_early_prediction': args.early_prediction,                                     # curriculum masking
        'early_min_weeks': args.early_min_weeks,                                           # min weeks for curriculum
    })
    _extra = []
    if args.action_weight: _extra.append('ActionWeight=ON')
    if args.early_prediction: _extra.append(f'EarlyPred=ON(min={args.early_min_weeks}w)')
    print(f"  [CL HP] hidden={param_dict['cl_hidden_size']}, τ={param_dict['cl_temperature']}, "
          f"mask={param_dict['cl_mask_ratio']}, noise={param_dict['cl_noise_std']}, "
          f"enc_layers={param_dict['cl_num_layers']}, cls_layers={param_dict['cl_cls_hidden_layers']}, λ={lambda_con}")
    if _extra: print(f"  [CL++] {', '.join(_extra)}")

# --- Standalone DL baselines ---
if is_dl_baseline:
    _hid = args.hidden_size if args.hidden_size is not None else 128
    param_dict.update({
        'days_per_week': ds_config['days_per_week'],
        'dl_hidden_size': _hid,
        'dl_num_layers': args.num_layers if args.num_layers is not None else 1,
        'dl_dropout': 0.3,
        'dl_attn_heads': 4,
        'dl_attn_dropout': 0.1,
    })
    print(f"  [DL Baseline HP] hidden={param_dict['dl_hidden_size']}, "
          f"layers={param_dict['dl_num_layers']}, dropout={param_dict['dl_dropout']}")

_t_data_end = time.time()
_t_data_elapsed = _t_data_end - _t_data_start
print(f"\n⏱ Data loading: {_t_data_elapsed:.2f}s")

# create the model
_t_model_start = time.time()
_is_graph_enhanced = (is_supcon or is_cl) and backend_mode.endswith('_graph')

if is_supcon:
    from src.models import SupConLGB
    model = SupConLGB(mode=backend_mode, param_dict=param_dict)
    if _is_graph_enhanced:
        from src.models import GraphEnhancedWrapper
        model = GraphEnhancedWrapper(model, param_dict, framework='supcon')
elif is_simclr:
    from src.models import CLSimCLR, NTXentLoss
    model = CLSimCLR(mode=backend_mode, param_dict=param_dict)
    if _is_graph_enhanced:
        from src.models import GraphEnhancedWrapper
        model = GraphEnhancedWrapper(model, param_dict, framework='simclr')
    ntxent_criterion = NTXentLoss(
        temperature=param_dict.get('cl_temperature', 0.1)
    ).to(device)
elif is_byol:
    from src.models import CLBYOL
    model = CLBYOL(mode=backend_mode, param_dict=param_dict)
    if _is_graph_enhanced:
        from src.models import GraphEnhancedWrapper
        model = GraphEnhancedWrapper(model, param_dict, framework='byol')
elif is_dl_baseline:
    from src.models import build_dl_baseline
    model = build_dl_baseline(mode, param_dict)
else:
    model = LGB(param_dict, mode=mode, contrastive=use_contrastive)
model = model.to(device)

# create contrastive loss if enabled (supcon always uses SupCon)
if use_contrastive or is_supcon:
    from src.models import SupConLoss
    supcon_criterion = SupConLoss(temperature=param_dict.get('supcon_temperature', 0.07)).to(device)

# select the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

_t_model_end = time.time()
_t_model_elapsed = _t_model_end - _t_model_start
_total_params = sum(p.numel() for p in model.parameters())
_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"⏱ Model creation: {_t_model_elapsed:.2f}s")
print(f"  Parameters: {_total_params:,} total, {_trainable_params:,} trainable")
print(f"  Device: {device}")
print()

# === Sampling: Loss function setup ===
class FocalLoss(torch.nn.Module):
    """Focal Loss for class imbalance: down-weights easy examples.
    FL(p) = -α(1-p)^γ log(p) for positive class
    FL(p) = -(1-α)p^γ log(1-p) for negative class
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none',
                                                  pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()

# Compute class weight for weighted_loss / focal_loss
_pos_weight = None
if args.sampling in ('weighted_loss', 'focal_loss'):
    # Compute from training data — need n_pos, n_neg
    if 'n_pos' in dir() and 'n_neg' in dir():
        _pw = n_neg / max(n_pos, 1)
    else:
        # Graph mode: compute from graph
        _train_labels = graph.labels[graph.train_mask]
        _n_pos = int(_train_labels.sum().item())
        _n_neg = len(_train_labels) - _n_pos
        _pw = _n_neg / max(_n_pos, 1)
    _pos_weight = torch.tensor([_pw], device=device)
    print(f"  [Sampling] pos_weight={_pw:.4f}")

if args.sampling == 'focal_loss':
    bce_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=_pos_weight).to(device)
    print(f"  [Sampling] FocalLoss (α=0.25, γ=2.0)")
elif args.sampling == 'weighted_loss':
    bce_loss_fn = lambda logits, targets: F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=_pos_weight)
    print(f"  [Sampling] Weighted BCE Loss")
else:
    bce_loss_fn = lambda logits, targets: F.binary_cross_entropy_with_logits(logits, targets)
print(f"  [Sampling] Strategy: {args.sampling}")

# train
_t_training_start = time.time()
_epoch_times = []

for epoch in range(epoch_num):
    _t_epoch_start = time.time()
    total_loss, total_examples = 0, 0
    model.train()
    _t_train_start = time.time()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        if _no_graph:
            seq_feat_batch, labels_batch = batch
            seq_feat_batch = seq_feat_batch.to(device)
            labels_batch = labels_batch.to(device)
            batch_size = seq_feat_batch.shape[0]
            sub_graph = {
                'batch_size': batch_size,
                'seq_feat': seq_feat_batch,
            }
            ground_truth = labels_batch.view(-1, 1).to(torch.float)
        else:
            sub_graph = batch
            batch_size = sub_graph['batch_size']
            ground_truth = sub_graph['labels'][:batch_size].view(-1, 1).to(torch.float)

        pred = model(sub_graph)

        if is_supcon:
            pred, z1, z2 = pred
            bce_loss = bce_loss_fn(pred, ground_truth)
            if lambda_con == 0.0:
                loss = bce_loss
            else:
                features = torch.stack([z1, z2], dim=1)   # (B, 2, proj_dim) — view-pair aware
                con_loss = supcon_criterion(features, ground_truth.view(-1))
                loss = bce_loss + lambda_con * con_loss
        elif is_simclr:
            pred, z1, z2 = pred
            bce_loss = bce_loss_fn(pred, ground_truth)
            cl_loss  = ntxent_criterion(z1, z2)
            loss = bce_loss + lambda_con * cl_loss
        elif is_byol:
            pred, byol_loss = pred
            bce_loss = bce_loss_fn(pred, ground_truth)
            loss = bce_loss + lambda_con * byol_loss
        elif use_contrastive:
            pred, proj_embed = pred
            bce_loss = bce_loss_fn(pred, ground_truth)
            con_loss = supcon_criterion(proj_embed, ground_truth)
            loss = bce_loss + lambda_con * con_loss
        else:
            loss = bce_loss_fn(pred, ground_truth)

        loss.backward()
        optimizer.step()

        # BYOL: momentum update sau mỗi optimizer step
        if is_byol:
            model.momentum_update()

        total_loss += float(loss) * batch_size
        total_examples += batch_size
    _t_train_end = time.time()
    _t_train_elapsed = _t_train_end - _t_train_start
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}, Train Time: {_t_train_elapsed:.2f}s")

    preds = []
    ground_truths = []
    model.eval()
    _t_eval_start = time.time()
    for batch in tqdm(test_loader):
        with torch.no_grad():
            if _no_graph:
                seq_feat_batch, labels_batch = batch
                seq_feat_batch = seq_feat_batch.to(device)
                labels_batch = labels_batch.to(device)
                batch_size = seq_feat_batch.shape[0]
                sub_graph = {
                    'batch_size': batch_size,
                    'seq_feat': seq_feat_batch,
                }
                truth = labels_batch.view(-1, 1)
            else:
                sub_graph = batch
                batch_size = sub_graph['batch_size']
                truth = sub_graph['labels'][:batch_size].view(-1, 1)

            pred = model(sub_graph)
            # SimCLR/BYOL/SupCon trả logits trực tiếp khi eval (model.eval() context)
            # use_contrastive (LGB) trả tuple (pred, embed) — cần unpack
            if use_contrastive and not is_supcon and not is_cl:
                pred, _ = pred  # discard projection during eval
            preds.append(pred)
            ground_truths.append(truth)
    pred = torch.cat(preds, dim=0).cpu()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    pred_scores = torch.sigmoid(pred).numpy()

    # Tìm threshold tối ưu từ ROC curve (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(ground_truth, pred_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    pred_label = pred_scores.copy()
    pred_label[pred_label < optimal_threshold] = 0
    pred_label[pred_label >= optimal_threshold] = 1

    auc = roc_auc_score(ground_truth, pred_scores)
    acc = np.sum(pred_label == ground_truth) / len(ground_truth)
    precision = precision_score(ground_truth, pred_label)
    recall = recall_score(ground_truth, pred_label)
    f1 = f1_score(ground_truth, pred_label)

    _t_eval_end = time.time()
    _t_eval_elapsed = _t_eval_end - _t_eval_start
    _t_epoch_end = time.time()
    _t_epoch_elapsed = _t_epoch_end - _t_epoch_start
    _epoch_times.append(_t_epoch_elapsed)

    print(f"Epoch: {epoch:03d}, Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Epoch: {epoch:03d}, Test ACC: {acc:.4f}")
    print(f"Epoch: {epoch:03d}, Test Precision: {precision:.4f}")
    print(f"Epoch: {epoch:03d}, Test Recall: {recall:.4f}")
    print(f"Epoch: {epoch:03d}, Test AUC: {auc:.4f}")
    print(f"Epoch: {epoch:03d}, Test F1: {f1:.4f}")
    print(f"Epoch: {epoch:03d}, Time: {_t_epoch_elapsed:.2f}s (train={_t_train_elapsed:.2f}s, eval={_t_eval_elapsed:.2f}s)")

    # --- Early Prediction: eval per-week (only if enabled) ---
    if args.early_prediction and getattr(model, 'early_mask', None) is not None:
        _week_count = model.early_mask.week_count
        print(f"\n  === Early Prediction Evaluation (epoch {epoch:03d}) ===")
        for eval_week in range(1, _week_count + 1):
            model.early_mask.set_eval_weeks(eval_week)
            model.eval()  # CRITICAL: tắt dropout, BN dùng running stats

            _ep_preds, _ep_gts = [], []
            with torch.no_grad():  # CRITICAL: không tính gradient
                for batch in test_loader:
                    if _no_graph:
                        seq_feat_batch, labels_batch = batch
                        seq_feat_batch = seq_feat_batch.to(device)
                        labels_batch = labels_batch.to(device)
                        _bs = seq_feat_batch.shape[0]
                        _sg = {'batch_size': _bs, 'seq_feat': seq_feat_batch}
                        _tr = labels_batch.view(-1, 1)
                    else:
                        _sg = batch
                        _bs = _sg['batch_size']
                        _tr = _sg['labels'][:_bs].view(-1, 1)

                    _pred = model(_sg)
                    if use_contrastive and not is_supcon and not is_cl:
                        _pred, _ = _pred
                    _ep_preds.append(_pred)
                    _ep_gts.append(_tr)

            _ep_pred = torch.cat(_ep_preds, dim=0).cpu()
            _ep_gt = torch.cat(_ep_gts, dim=0).cpu().numpy()
            _ep_scores = torch.sigmoid(_ep_pred).numpy()
            _ep_auc = roc_auc_score(_ep_gt, _ep_scores)

            _ep_fpr, _ep_tpr, _ep_thresholds = roc_curve(_ep_gt, _ep_scores)
            _ep_opt_idx = np.argmax(_ep_tpr - _ep_fpr)
            _ep_opt_t = _ep_thresholds[_ep_opt_idx]
            _ep_labels = (_ep_scores >= _ep_opt_t).astype(float)
            _ep_f1 = f1_score(_ep_gt, _ep_labels)
            _ep_acc = np.mean(_ep_labels == _ep_gt)

            print(f"  Week {eval_week}/{_week_count}: AUC={_ep_auc:.4f}, F1={_ep_f1:.4f}, ACC={_ep_acc:.4f}")

        # CRITICAL: Reset eval_weeks về full SAU MỖI EPOCH
        model.early_mask.set_eval_weeks(_week_count)

# === Timing Summary ===
_t_pipeline_end = time.time()
_t_training_total = _t_pipeline_end - _t_training_start
_t_pipeline_total = _t_pipeline_end - _t_pipeline_start
_t_avg_epoch = sum(_epoch_times) / len(_epoch_times) if _epoch_times else 0

print(f"\n{'═'*60}")
print(f"  ⏱ TIMING SUMMARY")
print(f"{'═'*60}")
print(f"  Mode:              {mode}")
print(f"  Dataset:           {ds_config['name']}")
print(f"  Branch:            {'GRAPH' if _uses_graph else 'NO-GRAPH'}")
print(f"  Device:            {device}")
print(f"  Epochs:            {epoch_num}")
print(f"{'─'*60}")
print(f"  Data loading:      {_t_data_elapsed:>8.2f}s")
print(f"  Model creation:    {_t_model_elapsed:>8.2f}s")
print(f"  Training total:    {_t_training_total:>8.2f}s")
print(f"  Avg epoch time:    {_t_avg_epoch:>8.2f}s")
print(f"{'─'*60}")
print(f"  TOTAL PIPELINE:    {_t_pipeline_total:>8.2f}s ({_t_pipeline_total/60:.1f}min)")
print(f"{'═'*60}")
print(f"\n  Final metrics (epoch {epoch_num-1:03d}):")
print(f"    ACC={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")

# === Action Weight Summary (if enabled) ===
if args.action_weight and hasattr(model, 'action_weighting'):
    _aw = model.action_weighting.get_weights().numpy()
    print(f"\n  📊 Learned Action Weights (baseline=1.0, higher=more important):")
    _sorted_idx = np.argsort(_aw)[::-1]
    for i in _sorted_idx:
        _bar = '█' * int(_aw[i] * 10)
        _marker = ' ⬆' if _aw[i] > 1.3 else (' ⬇' if _aw[i] < 0.7 else '')
        print(f"    action[{i:2d}]: {_aw[i]:5.3f} {_bar}{_marker}")

# === Early Prediction Summary (if enabled) ===
if args.early_prediction and getattr(model, 'early_mask', None) is not None:
    print(f"\n  📊 Early Prediction — final epoch per-week AUC (above):")

# ═══════════════════════════════════════════════════════════════
# === Auto-save: Lưu bảng kết quả cuối cùng vào file TXT ===
# Folder: results/{mode}/    File: {dataset}_{timestamp}.txt
# ═══════════════════════════════════════════════════════════════
from datetime import datetime as _dt

_result_timestamp = _dt.now().strftime('%Y%m%d_%H%M%S')
_result_dir = os.path.join(output_dir, 'results', mode)
os.makedirs(_result_dir, exist_ok=True)
_result_file = os.path.join(_result_dir, f'{dataset_name}_{_result_timestamp}.txt')

_lines = []
_lines.append('=' * 60)
_lines.append('  DP-SCL Training Results')
_lines.append('=' * 60)
_lines.append(f'  Mode:            {mode}')
_lines.append(f'  Dataset:         {ds_config["name"]}')
_lines.append(f'  Branch:          {"GRAPH" if _uses_graph else "NO-GRAPH"}')
_lines.append(f'  Epochs:          {epoch_num}')
_lines.append(f'  Learning Rate:   {learning_rate}')
_lines.append(f'  Batch Size:      {args.batch_size}')
_lines.append(f'  Sampling:        {args.sampling}')
if is_supcon or is_cl or use_contrastive:
    _lines.append(f'  λ (contrastive): {lambda_con}')
_lines.append(f'  Device:          {device}')
_lines.append(f'  Timestamp:       {_dt.now().strftime("%Y-%m-%d %H:%M:%S")}')
_lines.append('=' * 60)

# Hyperparameters (framework-specific)
if is_supcon:
    _lines.append('')
    _lines.append('  Hyperparameters (SupCon):')
    _lines.append(f'    hidden_size={param_dict.get("supcon_hidden_size")}, '
                  f'temperature={param_dict.get("supcon_temperature")}, '
                  f'mask_ratio={param_dict.get("supcon_mask_ratio")}')
    _lines.append(f'    noise_std={param_dict.get("supcon_noise_std")}, '
                  f'enc_layers={param_dict.get("supcon_num_layers")}, '
                  f'cls_layers={param_dict.get("supcon_cls_hidden_layers")}')
    if args.action_weight:
        _lines.append(f'    action_weight=ON')
    if args.early_prediction:
        _lines.append(f'    early_prediction=ON (min_weeks={args.early_min_weeks})')
elif is_cl:
    _fw_name = 'SimCLR' if is_simclr else 'BYOL'
    _lines.append('')
    _lines.append(f'  Hyperparameters ({_fw_name}):')
    _lines.append(f'    hidden_size={param_dict.get("cl_hidden_size")}, '
                  f'temperature={param_dict.get("cl_temperature")}, '
                  f'mask_ratio={param_dict.get("cl_mask_ratio")}')
    _lines.append(f'    noise_std={param_dict.get("cl_noise_std")}, '
                  f'enc_layers={param_dict.get("cl_num_layers")}, '
                  f'cls_layers={param_dict.get("cl_cls_hidden_layers")}')
    if is_byol:
        _lines.append(f'    momentum={param_dict.get("cl_momentum")}')
    if args.action_weight:
        _lines.append(f'    action_weight=ON')
    if args.early_prediction:
        _lines.append(f'    early_prediction=ON (min_weeks={args.early_min_weeks})')

# Final Results
_lines.append('')
_lines.append('=' * 60)
_lines.append(f'  FINAL RESULTS (Epoch {epoch_num - 1:03d})')
_lines.append('=' * 60)
_lines.append(f'  ACC:          {acc:.4f}')
_lines.append(f'  AUC:          {auc:.4f}')
_lines.append(f'  F1:           {f1:.4f}')
_lines.append(f'  Precision:    {precision:.4f}')
_lines.append(f'  Recall:       {recall:.4f}')
_lines.append(f'  Threshold:    {optimal_threshold:.4f}')
_lines.append('=' * 60)

# Timing
_lines.append('')
_lines.append('  TIMING')
_lines.append('-' * 60)
_lines.append(f'  Data loading:      {_t_data_elapsed:>8.2f}s')
_lines.append(f'  Model creation:    {_t_model_elapsed:>8.2f}s')
_lines.append(f'  Training total:    {_t_training_total:>8.2f}s')
_lines.append(f'  Avg epoch time:    {_t_avg_epoch:>8.2f}s')
_lines.append(f'  TOTAL PIPELINE:    {_t_pipeline_total:>8.2f}s ({_t_pipeline_total/60:.1f}min)')
_lines.append('=' * 60)

# Action Weights (if enabled)
if args.action_weight and hasattr(model, 'action_weighting'):
    _aw = model.action_weighting.get_weights().numpy()
    _lines.append('')
    _lines.append('  ACTION WEIGHTS (learned):')
    _sorted = np.argsort(_aw)[::-1]
    for i in _sorted:
        _lines.append(f'    action[{i:2d}]: {_aw[i]:5.3f}')

# Write to file
with open(_result_file, 'w', encoding='utf-8') as _f:
    _f.write('\n'.join(_lines) + '\n')

print(f"\n  💾 Results saved to: {_result_file}")
