"""
Phase 1 Baseline: 1D-CNN + BiLSTM + Attention
5-Fold Stratified Cross-Validation Training Script

Supports:
  - 3 attention variants: bahdanau, multihead, cross
  - Ablation experiments (disable CNN/BiLSTM/Attention)
  - AdamW + ReduceLROnPlateau + gradient clipping + early stopping
  - Class weighting for imbalanced data

Usage:
    python3 train_baseline.py                                   # default (bahdanau)
    python3 train_baseline.py --attn multihead                  # multihead attention
    python3 train_baseline.py --attn cross                      # cross attention
    python3 train_baseline.py --no-cnn                          # ablation: no CNN
    python3 train_baseline.py --no-bilstm                       # ablation: no BiLSTM
    python3 train_baseline.py --no-attention                    # ablation: no Attention
    python3 train_baseline.py --unidirectional                  # ablation: LSTM instead of BiLSTM
    python3 train_baseline.py -e 30 -lr 5e-4 -folds 5          # custom training
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import DropoutPredictor

# ======================== Args ========================
parser = argparse.ArgumentParser(description='Baseline 1D-CNN + BiLSTM + Attention — 5-Fold CV')
parser.add_argument('-indir', type=str, default='.', help='project root dir')
parser.add_argument('-e', type=int, default=20, help='epochs per fold (default: 20)')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('-bs', type=int, default=256, help='batch size (default: 256)')
parser.add_argument('-folds', type=int, default=5, help='number of CV folds (default: 5)')
parser.add_argument('-patience', type=int, default=5, help='early stopping patience (default: 5)')
parser.add_argument('-seed', type=int, default=42, help='random seed (default: 42)')
# Attention variant
parser.add_argument('--attn', type=str, default='bahdanau',
                    choices=['bahdanau', 'multihead', 'cross'],
                    help='attention type: bahdanau, multihead, cross (default: bahdanau)')
# Ablation flags
parser.add_argument('--no-cnn', action='store_true', help='Ablation: disable CNN')
parser.add_argument('--no-bilstm', action='store_true', help='Ablation: disable BiLSTM')
parser.add_argument('--no-attention', action='store_true', help='Ablation: disable Attention')
parser.add_argument('--unidirectional', action='store_true',
                    help='Ablation: use unidirectional LSTM instead of BiLSTM')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dir = os.path.abspath(os.path.expanduser(args.indir))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Build experiment name
exp_parts = []
exp_parts.append('CNN' if not args.no_cnn else 'noCNN')
exp_parts.append('BiLSTM' if not (args.no_bilstm or args.unidirectional) else
                 ('LSTM' if args.unidirectional else 'noLSTM'))
exp_parts.append(args.attn if not args.no_attention else 'noAttn')
exp_name = '+'.join(exp_parts)

print(f"\n{'='*60}")
print(f"  Phase 1 Baseline: {exp_name}")
print(f"  5-Fold Stratified Cross-Validation")
print(f"{'='*60}")
print(f"  Device:      {device}")
print(f"  Attention:   {args.attn}")
print(f"  Epochs:      {args.e}")
print(f"  LR:          {args.lr}")
print(f"  Batch size:  {args.bs}")
print(f"  Folds:       {args.folds}")
print(f"  Patience:    {args.patience}")
print(f"  Seed:        {args.seed}")
if args.no_cnn:
    print(f"  ⚠ Ablation:  CNN disabled")
if args.no_bilstm:
    print(f"  ⚠ Ablation:  BiLSTM disabled")
if args.unidirectional:
    print(f"  ⚠ Ablation:  Unidirectional LSTM (not BiLSTM)")
if args.no_attention:
    print(f"  ⚠ Ablation:  Attention disabled (mean pooling)")
print(f"{'='*60}\n")

# ======================== Load Data ========================
npz_path = os.path.join(input_dir, 'datastore', 'all_data_std.npz')
print(f"Loading data from: {npz_path}")
data = np.load(npz_path)

# Merge train + test for k-fold
train_data = data['t_data']    # (N_train, 5, 7, 22)
train_label = data['t_label']  # (N_train,)
test_data = data['v_data']     # (N_test, 5, 7, 22)
test_label = data['v_label']   # (N_test,)

all_data = np.concatenate([train_data, test_data], axis=0)
all_labels = np.concatenate([train_label, test_label], axis=0)

n_pos = int(np.sum(all_labels == 1))
n_neg = int(np.sum(all_labels == 0))
pos_weight = n_neg / max(n_pos, 1)

print(f"  Total samples: {len(all_data)}")
print(f"  Label distribution: neg={n_neg}, pos={n_pos} (ratio={n_neg/max(n_pos,1):.2f})")
print(f"  pos_weight for BCE: {pos_weight:.2f}")
print(f"  Data shape: {all_data.shape}")
print()

# ======================== Model Config ========================
model_config = {
    'num_actions': 22,
    'days_per_week': 7,
    'num_weeks': 5,
    'cnn_out_dim': 128,
    'cnn_dropout': 0.2,
    'lstm_hidden': 64,
    'lstm_layers': 2,
    'lstm_dropout': 0.3,
    'attn_heads': 2,
    'attn_dropout': 0.1,
    'cls_dropout': 0.3,
    'attention_type': args.attn,
    'use_cnn': not args.no_cnn,
    'use_bilstm': not args.no_bilstm,
    'use_attention': not args.no_attention,
}


# ======================== Training Functions ========================
def train_one_epoch(model, loader, optimizer, pw_tensor):
    model.train()
    total_loss, total_samples = 0.0, 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).view(-1, 1)
        optimizer.zero_grad()
        logits, _ = model(batch_x)
        loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pw_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)
    return total_loss / total_samples


def evaluate(model, loader):
    model.eval()
    all_logits, all_truths, all_attn = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits, attn_w = model(batch_x)
            all_logits.append(logits.cpu())
            all_truths.append(batch_y.view(-1, 1))
            all_attn.append(attn_w.cpu())

    logits = torch.cat(all_logits, dim=0)
    truths = torch.cat(all_truths, dim=0).numpy()
    attn = torch.cat(all_attn, dim=0).numpy()

    pred_prob = torch.sigmoid(logits).numpy()
    pred_label = (pred_prob >= 0.5).astype(float)

    acc = np.mean(pred_label == truths)
    precision = precision_score(truths, pred_label, zero_division=0)
    recall = recall_score(truths, pred_label, zero_division=0)
    f1 = f1_score(truths, pred_label, zero_division=0)
    auc = roc_auc_score(truths, logits.numpy())

    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}, attn


# ======================== K-Fold CV ========================
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
fold_results = []
total_start = time.time()

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_data, all_labels)):
    print(f"\n{'─'*60}")
    print(f"  Fold {fold_idx + 1}/{args.folds}  [{exp_name}]")
    print(f"{'─'*60}")

    X_train = torch.from_numpy(all_data[train_idx]).float()
    y_train = torch.from_numpy(all_labels[train_idx]).float()
    X_val = torch.from_numpy(all_data[val_idx]).float()
    y_val = torch.from_numpy(all_labels[val_idx]).float()

    # Class weight for this fold
    fold_pos = y_train.sum().item()
    fold_neg = len(y_train) - fold_pos
    fold_pw = torch.tensor([fold_neg / max(fold_pos, 1)]).to(device)

    print(f"  Train: {len(X_train)} (pos={int(fold_pos)}, neg={int(fold_neg)})")
    print(f"  Val:   {len(X_val)} (pos={int(y_val.sum())}, neg={int(len(y_val)-y_val.sum())})")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.bs, shuffle=False)

    # Build model
    model = DropoutPredictor(model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                      patience=3, min_lr=1e-6)

    if fold_idx == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {total_params:,}")

    # Training with early stopping
    best_auc = 0.0
    best_state = None
    patience_counter = 0

    fold_start = time.time()
    for epoch in range(args.e):
        train_loss = train_one_epoch(model, train_loader, optimizer, fold_pw)
        metrics, attn_weights = evaluate(model, val_loader)
        scheduler.step(metrics['auc'])
        current_lr = optimizer.param_groups[0]['lr']

        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = ' ★'
        else:
            patience_counter += 1
            marker = ''

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(f"  Epoch {epoch+1:3d} | Loss={train_loss:.4f} | "
                  f"ACC={metrics['acc']:.4f} AUC={metrics['auc']:.4f} "
                  f"F1={metrics['f1']:.4f} | lr={current_lr:.1e}{marker}")

        if patience_counter >= args.patience:
            print(f"  ⏹ Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # Restore best & final eval
    model.load_state_dict(best_state)
    final_metrics, final_attn = evaluate(model, val_loader)
    fold_time = time.time() - fold_start

    print(f"\n  ✅ Fold {fold_idx+1} Best Results ({fold_time:.0f}s):")
    print(f"     ACC:       {final_metrics['acc']:.4f}")
    print(f"     Precision: {final_metrics['precision']:.4f}")
    print(f"     Recall:    {final_metrics['recall']:.4f}")
    print(f"     F1:        {final_metrics['f1']:.4f}")
    print(f"     AUC:       {final_metrics['auc']:.4f}")

    # Attention weight analysis
    mean_attn = final_attn.mean(axis=0)
    week_labels = [f'W{i+1}' for i in range(len(mean_attn))]
    attn_str = ', '.join(f'{w}={a:.3f}' for w, a in zip(week_labels, mean_attn))
    print(f"     Attn avg:  [{attn_str}]")

    fold_results.append(final_metrics)

# ======================== Summary ========================
total_time = time.time() - total_start

print(f"\n{'='*60}")
print(f"  FINAL RESULTS: {args.folds}-Fold CV — {exp_name}")
print(f"{'='*60}")
print(f"  {'Metric':<12} {'Mean':>8} {'± Std':>8}   {'Per-fold values'}")
print(f"  {'─'*54}")

for metric_name in ['acc', 'precision', 'recall', 'f1', 'auc']:
    values = [r[metric_name] for r in fold_results]
    mean_val = np.mean(values)
    std_val = np.std(values)
    per_fold = ', '.join(f'{v:.4f}' for v in values)
    print(f"  {metric_name.upper():<12} {mean_val:>8.4f} {f'±{std_val:.4f}':>8}   [{per_fold}]")

print(f"  {'─'*54}")
print(f"  Total time: {total_time:.0f}s ({int(total_time//60)}m{int(total_time%60)}s)")
print(f"  Device: {device}")
print(f"{'='*60}")

# Save results
results_dir = os.path.join(input_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
timestamp = time.strftime('%Y%m%d_%H%M%S')
result_file = os.path.join(results_dir, f'baseline_{exp_name}_{timestamp}.txt')

with open(result_file, 'w') as f:
    f.write(f"Phase 1 Baseline: {exp_name}\n")
    f.write(f"{args.folds}-Fold Stratified Cross-Validation\n")
    f.write(f"{'='*50}\n")
    f.write(f"Attention: {args.attn}\n")
    f.write(f"Epochs: {args.e}, LR: {args.lr}, BS: {args.bs}, Patience: {args.patience}\n")
    f.write(f"Seed: {args.seed}, Device: {device}\n")
    f.write(f"Ablation: use_cnn={not args.no_cnn}, use_bilstm={not args.no_bilstm}, "
            f"use_attention={not args.no_attention}\n\n")
    for metric_name in ['acc', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric_name] for r in fold_results]
        f.write(f"{metric_name.upper():<12} {np.mean(values):.4f} ± {np.std(values):.4f}\n")
    f.write(f"\nTotal time: {total_time:.0f}s\n")
    f.write(f"\nPer-fold detail:\n")
    for i, r in enumerate(fold_results):
        f.write(f"  Fold {i+1}: ACC={r['acc']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
                f"F1={r['f1']:.4f} AUC={r['auc']:.4f}\n")

print(f"\n  📄 Results saved to: {result_file}")
