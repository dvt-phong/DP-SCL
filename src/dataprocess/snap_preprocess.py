"""
SNAP MOOC Preprocessing: 3 TSV → (N, 5, 7, 6) tensor → .npz

Pipeline:
  1. Load mooc_actions + mooc_action_features + mooc_action_labels
  2. Per-user label: user có bất kỳ action nào label=1 → dropout
  3. Per-user: bin timestamps into 35 equal bins (matching 5w×7d structure)
  4. Per bin: engineer 6 features:
     [action_count, mean_feat0, mean_feat1, mean_feat2, mean_feat3, unique_targets]
  5. Stratified train/test split (80/20)
  6. StandardScaler → save .npz

Features per time bin (6 dims):
  0: action_count      — Số lượng actions trong bin
  1: mean_feature_0    — Trung bình FEATURE0 trong bin
  2: mean_feature_1    — Trung bình FEATURE1 trong bin
  3: mean_feature_2    — Trung bình FEATURE2 trong bin
  4: mean_feature_3    — Trung bình FEATURE3 trong bin
  5: unique_targets    — Số target activities khác nhau trong bin

Usage:
  python src/dataprocess/snap_preprocess.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======================== Paths ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'datastore', 'snap', 'act-mooc')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'datastore')

STA_DAY = 35
WEEK_COUNT = 5
DAYS_PER_WEEK = 7
NUM_FEATURES = 6
TEST_SIZE = 0.2
RANDOM_SEED = 42

print("=" * 60)
print("  SNAP MOOC Preprocessing Pipeline")
print("=" * 60)

# ======================== 1. Load data ========================
print("\n[1/6] Loading SNAP MOOC TSV files...")

actions = pd.read_csv(os.path.join(DATA_DIR, 'mooc_actions.tsv'), sep='\t')
features = pd.read_csv(os.path.join(DATA_DIR, 'mooc_action_features.tsv'), sep='\t')
labels = pd.read_csv(os.path.join(DATA_DIR, 'mooc_action_labels.tsv'), sep='\t')

print(f"  Actions:  {len(actions):>10,} rows")
print(f"  Features: {len(features):>10,} rows")
print(f"  Labels:   {len(labels):>10,} rows")

# ======================== 2. Merge all tables ========================
print("\n[2/6] Merging action tables...")

df = actions.merge(features, on='ACTIONID').merge(labels, on='ACTIONID')
print(f"  Merged: {len(df):,} rows")
print(f"  Users: {df['USERID'].nunique():,}")
print(f"  Targets: {df['TARGETID'].nunique():,}")
print(f"  Timestamp range: {df['TIMESTAMP'].min():.0f} — {df['TIMESTAMP'].max():.0f} seconds")
print(f"  = {df['TIMESTAMP'].max() / 86400:.1f} days")

# ======================== 3. Per-user labels ========================
print("\n[3/6] Creating per-user dropout labels...")

# User is dropout if ANY of their actions has LABEL=1
user_dropout = df.groupby('USERID')['LABEL'].max().reset_index()
user_dropout.columns = ['USERID', 'dropout']

n_dropout = int(user_dropout['dropout'].sum())
n_total = len(user_dropout)
print(f"  Dropout: {n_dropout:,} / {n_total:,} ({n_dropout/n_total:.3f})")

# ======================== 4. Temporal binning ========================
print(f"\n[4/6] Binning timestamps into {STA_DAY} bins per user...")

# For each user, normalize timestamps to [0, 1] relative to their own time range,
# then map to one of 35 bins. This handles users with different activity durations.
# If user has all actions at same timestamp → all in bin 0.

def process_user(group):
    """Process one user's actions into (35, 6) tensor."""
    tensor = np.zeros((STA_DAY, NUM_FEATURES), dtype=np.float64)

    ts = group['TIMESTAMP'].values
    ts_min, ts_max = ts.min(), ts.max()

    if ts_max > ts_min:
        # Normalize to [0, 1) → bin index
        normalized = (ts - ts_min) / (ts_max - ts_min + 1e-9)
        bins = np.clip((normalized * STA_DAY).astype(int), 0, STA_DAY - 1)
    else:
        bins = np.zeros(len(ts), dtype=int)

    group = group.copy()
    group['bin'] = bins

    for b in range(STA_DAY):
        mask = group['bin'] == b
        subset = group[mask]
        if len(subset) == 0:
            continue
        tensor[b, 0] = len(subset)                           # action count
        tensor[b, 1] = subset['FEATURE0'].mean()             # mean feature 0
        tensor[b, 2] = subset['FEATURE1'].mean()             # mean feature 1
        tensor[b, 3] = subset['FEATURE2'].mean()             # mean feature 2
        tensor[b, 4] = subset['FEATURE3'].mean()             # mean feature 3
        tensor[b, 5] = subset['TARGETID'].nunique()           # unique targets

    return tensor

users = sorted(df['USERID'].unique())
N = len(users)
user_to_idx = {u: i for i, u in enumerate(users)}

all_tensors = np.zeros((N, STA_DAY, NUM_FEATURES), dtype=np.float32)

# Process each user
from tqdm import tqdm
for uid, group in tqdm(df.groupby('USERID'), desc="  Processing users", total=N):
    idx = user_to_idx[uid]
    all_tensors[idx] = process_user(group)

# Reshape to (N, week_count, days_per_week, num_features)
all_tensors = all_tensors.reshape(N, WEEK_COUNT, DAYS_PER_WEEK, NUM_FEATURES)

# Get labels in order
all_labels = np.array([
    user_dropout[user_dropout['USERID'] == u]['dropout'].values[0]
    for u in users
], dtype=np.float32)

print(f"  Tensor shape: {all_tensors.shape}")
print(f"  Labels: {int(all_labels.sum()):,} dropout / {N:,} total ({all_labels.mean():.3f})")

# ======================== 5. Train/test split ========================
print(f"\n[5/6] Stratified train/test split ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")

train_idx, test_idx = train_test_split(
    np.arange(N), test_size=TEST_SIZE,
    stratify=all_labels, random_state=RANDOM_SEED
)

train_data = all_tensors[train_idx]
train_labels = all_labels[train_idx]
test_data = all_tensors[test_idx]
test_labels = all_labels[test_idx]

print(f"  Train: {len(train_idx):,} (dropout rate: {train_labels.mean():.3f})")
print(f"  Test:  {len(test_idx):,} (dropout rate: {test_labels.mean():.3f})")

# ======================== 6. Normalize + Save ========================
print("\n[6/6] StandardScaler normalization + saving .npz...")

train_flat = train_data.reshape(len(train_idx), -1)
test_flat = test_data.reshape(len(test_idx), -1)

scaler = StandardScaler()
train_flat = scaler.fit_transform(train_flat).astype(np.float32)
test_flat = scaler.transform(test_flat).astype(np.float32)

train_data = train_flat.reshape(len(train_idx), WEEK_COUNT, DAYS_PER_WEEK, NUM_FEATURES)
test_data = test_flat.reshape(len(test_idx), WEEK_COUNT, DAYS_PER_WEEK, NUM_FEATURES)

output_path = os.path.join(OUTPUT_DIR, 'snap_data_std.npz')
np.savez(
    output_path,
    t_data=train_data, t_label=train_labels,
    v_data=test_data, v_label=test_labels
)

file_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"  Saved: {output_path} ({file_size:.1f} MB)")
print(f"  Train shape: {train_data.shape}, Test shape: {test_data.shape}")

print("\n" + "=" * 60)
print("  ✅ SNAP MOOC preprocessing complete!")
print(f"  Output: {output_path}")
print(f"  Tensor: (N, {WEEK_COUNT}, {DAYS_PER_WEEK}, {NUM_FEATURES})")
print("=" * 60)
