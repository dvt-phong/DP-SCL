"""
SNAP MOOC Graph Construction: Xây StrongClassmatesGraph cho SNAP dataset.

Logic:
  - "Classmates" = users cùng interact với TARGETID (giống course trong KDD)
  - Cosine similarity trên activity feature vectors → threshold → edges
  - Context features: 7 dims từ per-user statistics (tổng actions, mean features, ...)
  - Temporal seq_feat: load từ snap_data_std.npz (đã preprocess)

Output: datastore/snap_graph/StrongClassmatesGraph.pkl
  - PyG Data object:
    - edge_index, labels, org_context (B, 7), seq_feat (B, W*D*F)
    - enhanced_context (B, 7), train_mask, test_mask, train_truth, test_truth

Usage:
  python src/dataprocess/snap_build_graph.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# ======================== Config ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'datastore', 'snap', 'act-mooc')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'datastore')

SIM_THRESHOLD = 0.95
WEEK_COUNT = 5
DAYS_PER_WEEK = 7
STA_DAY = WEEK_COUNT * DAYS_PER_WEEK  # 35
NUM_FEATURES = 6
TEST_SIZE = 0.2
RANDOM_SEED = 42

print("=" * 60)
print("  SNAP MOOC Graph Construction")
print("=" * 60)

# ======================== 1. Load raw data ========================
print("\n[1/7] Loading SNAP MOOC data...")
actions = pd.read_csv(os.path.join(DATA_DIR, 'mooc_actions.tsv'), sep='\t')
features = pd.read_csv(os.path.join(DATA_DIR, 'mooc_action_features.tsv'), sep='\t')
labels = pd.read_csv(os.path.join(DATA_DIR, 'mooc_action_labels.tsv'), sep='\t')

df = actions.merge(features, on='ACTIONID').merge(labels, on='ACTIONID')
print(f"  Merged: {len(df):,} rows, {df['USERID'].nunique():,} users, {df['TARGETID'].nunique():,} targets")

# ======================== 2. Per-user labels & features ========================
print("\n[2/7] Computing per-user labels and context features...")

user_stats = df.groupby('USERID').agg(
    dropout=('LABEL', 'max'),
    total_actions=('ACTIONID', 'count'),
    unique_targets=('TARGETID', 'nunique'),
    mean_feat0=('FEATURE0', 'mean'),
    mean_feat1=('FEATURE1', 'mean'),
    mean_feat2=('FEATURE2', 'mean'),
    mean_feat3=('FEATURE3', 'mean'),
    std_feat0=('FEATURE0', 'std'),        # variability of behavior
    temporal_span=('TIMESTAMP', lambda x: x.max() - x.min()),  # activity duration
).reset_index()
user_stats['std_feat0'] = user_stats['std_feat0'].fillna(0)
user_stats['temporal_span'] = user_stats['temporal_span'].fillna(0)

users = sorted(df['USERID'].unique())
N = len(users)
user_to_idx = {u: i for i, u in enumerate(users)}

print(f"  Users: {N:,}")
print(f"  Dropout: {int(user_stats['dropout'].sum()):,} / {N:,} ({user_stats['dropout'].mean():.3f})")

# ======================== 3. Build temporal tensor (same as snap_preprocess) ========================
print(f"\n[3/7] Building temporal tensor (N, {STA_DAY}, {NUM_FEATURES})...")

def process_user(group):
    tensor = np.zeros((STA_DAY, NUM_FEATURES), dtype=np.float64)
    ts = group['TIMESTAMP'].values
    ts_min, ts_max = ts.min(), ts.max()
    if ts_max > ts_min:
        normalized = (ts - ts_min) / (ts_max - ts_min + 1e-9)
        bins = np.clip((normalized * STA_DAY).astype(int), 0, STA_DAY - 1)
    else:
        bins = np.zeros(len(ts), dtype=int)
    group = group.copy()
    group['bin'] = bins
    for b in range(STA_DAY):
        subset = group[group['bin'] == b]
        if len(subset) == 0:
            continue
        tensor[b, 0] = len(subset)
        tensor[b, 1] = subset['FEATURE0'].mean()
        tensor[b, 2] = subset['FEATURE1'].mean()
        tensor[b, 3] = subset['FEATURE2'].mean()
        tensor[b, 4] = subset['FEATURE3'].mean()
        tensor[b, 5] = subset['TARGETID'].nunique()
    return tensor

all_tensors = np.zeros((N, STA_DAY, NUM_FEATURES), dtype=np.float32)
for uid, group in tqdm(df.groupby('USERID'), desc="  Processing users", total=N):
    all_tensors[user_to_idx[uid]] = process_user(group)

# ======================== 4. Build user-target matrix for similarity ========================
print("\n[4/7] Building user-target interaction matrix...")

targets = sorted(df['TARGETID'].unique())
target_to_idx = {t: i for i, t in enumerate(targets)}

# User-target matrix: how many actions each user did on each target
user_target_matrix = np.zeros((N, len(targets)), dtype=np.float32)
for uid, group in df.groupby('USERID'):
    user_idx = user_to_idx[uid]
    for tid, count in group['TARGETID'].value_counts().items():
        user_target_matrix[user_idx, target_to_idx[tid]] = count

# Normalize rows (per-user L2 norm)
row_norms = np.linalg.norm(user_target_matrix, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1
user_target_normed = user_target_matrix / row_norms

print(f"  User-target matrix: {user_target_matrix.shape}")
print(f"  Non-zero entries: {(user_target_matrix > 0).sum():,}")

# ======================== 5. Build edges via cosine similarity ========================
print(f"\n[5/7] Building edges (cosine similarity ≥ {SIM_THRESHOLD})...")

# Group users by target (like KDD groups by course)
target_to_users = {}
for uid, group in df.groupby('USERID'):
    for tid in group['TARGETID'].unique():
        if tid not in target_to_users:
            target_to_users[tid] = []
        target_to_users[tid].append(user_to_idx[uid])

source_nodes = []
target_nodes = []

# For each target (="course"), find similar users
for tid in tqdm(sorted(target_to_users.keys()), desc="  Building edges"):
    user_indices = sorted(set(target_to_users[tid]))
    if len(user_indices) < 2:
        continue
    
    # Cosine similarity among users sharing this target
    sub_matrix = user_target_normed[user_indices]
    cos_sim = cosine_similarity(sub_matrix)
    np.fill_diagonal(cos_sim, 0)
    
    row, col = np.where(cos_sim >= SIM_THRESHOLD)
    for i in range(len(row)):
        source_nodes.append(user_indices[row[i]])
        target_nodes.append(user_indices[col[i]])

# Deduplicate edges
edge_set = set()
for s, t in zip(source_nodes, target_nodes):
    if s != t:
        edge_set.add((s, t))

source_nodes = [e[0] for e in edge_set]
target_nodes = [e[1] for e in edge_set]

print(f"  Total edges: {len(source_nodes):,}")
print(f"  Avg degree: {len(source_nodes) / N:.1f}")

# ======================== 6. Train/test split ========================
print(f"\n[6/7] Stratified train/test split ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")

all_labels = np.array([user_stats[user_stats['USERID'] == u]['dropout'].values[0] for u in users], dtype=np.int64)

train_idx, test_idx = train_test_split(
    np.arange(N), test_size=TEST_SIZE,
    stratify=all_labels, random_state=RANDOM_SEED
)

print(f"  Train: {len(train_idx):,}, Test: {len(test_idx):,}")

# ======================== 7. Build PyG Data object ========================
print("\n[7/7] Building PyG Data object...")

# Context features (7 dims): behavioral statistics — NO label leakage!
context_cols = ['total_actions', 'unique_targets', 'mean_feat0', 'mean_feat1', 'mean_feat2', 'mean_feat3', 'std_feat0']
context_array = np.array([
    user_stats[user_stats['USERID'] == u][context_cols].values[0] for u in users
], dtype=np.float32)

# Standardize context
scaler_ctx = StandardScaler()
context_std = scaler_ctx.fit_transform(context_array).astype(np.float32)

# Standardize temporal
scaler_seq = StandardScaler()
seq_flat = all_tensors.reshape(N, -1)
seq_std = scaler_seq.fit_transform(seq_flat).astype(np.float32)

# Build edge_index
if len(source_nodes) > 0:
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
else:
    # Fallback: self-loops if no edges
    edge_index = torch.tensor([list(range(N)), list(range(N))], dtype=torch.long)
    print("  ⚠️ No edges found, using self-loops")

# Build graph
graph = Data(edge_index=edge_index)
graph.labels = torch.tensor(all_labels, dtype=torch.long)
graph.n_id = torch.arange(N, dtype=torch.long)

# org_context: 7 features (used by Context Embedding)
graph.org_context = torch.tensor(context_std, dtype=torch.float)

# enhanced_context: same as org_context for now (no link prediction for SNAP)
graph.enhanced_context = torch.tensor(context_std, dtype=torch.float)

# seq_feat: flattened temporal → (N, sta_day * num_features)
graph.seq_feat = torch.tensor(seq_std, dtype=torch.float)

# Train/test masks
graph.train_mask = torch.tensor(train_idx, dtype=torch.long)
graph.test_mask = torch.tensor(test_idx, dtype=torch.long)
graph.train_truth = torch.tensor(all_labels[train_idx], dtype=torch.long)
graph.test_truth = torch.tensor(all_labels[test_idx], dtype=torch.long)

# Save
output_path = os.path.join(OUTPUT_DIR, 'snap_StrongClassmatesGraph.pkl')
with open(output_path, 'wb') as f:
    pkl.dump(graph, f)

file_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"\n  Saved: {output_path} ({file_size:.1f} MB)")
print(f"  Nodes: {N:,}")
print(f"  Edges: {edge_index.shape[1]:,}")
print(f"  Context features: {graph.org_context.shape}")
print(f"  Seq features: {graph.seq_feat.shape}")
print(f"  Train: {len(train_idx):,}, Test: {len(test_idx):,}")

print("\n" + "=" * 60)
print("  ✅ SNAP graph construction complete!")
print("=" * 60)
