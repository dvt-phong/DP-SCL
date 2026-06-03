"""
OULAD Preprocessing: 7 CSV → (N, 5, 7, 20) tensor → .npz

Pipeline:
  1. Load studentVle + vle → merge để get activity_type
  2. Map date → day 0..34 (first 35 days from course start)
  3. Aggregate sum_click per (student, module, presentation, day, activity_type)
  4. Pivot → (N_students, 35, 20) tensor
  5. Label: Withdrawn/Fail → 1, Pass/Distinction → 0
  6. Stratified train/test split (80/20)
  7. StandardScaler per feature → save .npz

Usage:
  python src/dataprocess/oulad_preprocess.py
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
DATA_DIR = os.path.join(PROJECT_DIR, 'datastore', 'oulad')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'datastore')

STA_DAY = 35
WEEK_COUNT = 5
DAYS_PER_WEEK = 7
TEST_SIZE = 0.2
RANDOM_SEED = 42

print("=" * 60)
print("  OULAD Preprocessing Pipeline")
print("=" * 60)

# ======================== 1. Load data ========================
print("\n[1/7] Loading OULAD CSV files...")

student_vle = pd.read_csv(os.path.join(DATA_DIR, 'studentVle.csv'))
vle = pd.read_csv(os.path.join(DATA_DIR, 'vle.csv'))
student_info = pd.read_csv(os.path.join(DATA_DIR, 'studentInfo.csv'))
courses = pd.read_csv(os.path.join(DATA_DIR, 'courses.csv'))

print(f"  studentVle:  {len(student_vle):>10,} rows")
print(f"  vle:         {len(vle):>10,} rows")
print(f"  studentInfo: {len(student_info):>10,} rows")
print(f"  courses:     {len(courses):>10,} rows")

# ======================== 2. Merge activity_type ========================
print("\n[2/7] Merging VLE activity types...")

# Keep only columns we need from vle
vle_types = vle[['id_site', 'code_module', 'code_presentation', 'activity_type']].copy()

# Merge to get activity_type for each interaction
svle = student_vle.merge(
    vle_types,
    on=['id_site', 'code_module', 'code_presentation'],
    how='left'
)

# Drop rows without activity type (should be rare)
n_before = len(svle)
svle = svle.dropna(subset=['activity_type'])
print(f"  Merged: {len(svle):,} rows ({n_before - len(svle)} dropped for missing activity_type)")

# ======================== 3. Get activity types (sorted) ========================
ACTIVITY_TYPES = sorted(svle['activity_type'].unique())
NUM_ACTIVITIES = len(ACTIVITY_TYPES)
activity_to_idx = {a: i for i, a in enumerate(ACTIVITY_TYPES)}
print(f"  Activity types ({NUM_ACTIVITIES}): {ACTIVITY_TYPES}")

# ======================== 4. Filter to first 35 days ========================
print(f"\n[3/7] Filtering to first {STA_DAY} days (day 0..{STA_DAY-1})...")

# date in OULAD is relative to course start (can be negative for pre-course)
# We take day 0 to 34 (first 5 weeks of course)
svle = svle[(svle['date'] >= 0) & (svle['date'] < STA_DAY)].copy()
print(f"  After filter: {len(svle):,} interactions")

# ======================== 5. Create labels ========================
print("\n[4/7] Creating dropout labels...")

# Each student-module-presentation is one "enrollment"
# Label: Withdrawn/Fail → 1 (dropout), Pass/Distinction → 0
student_info['dropout'] = student_info['final_result'].isin(['Withdrawn', 'Fail']).astype(int)

# Create unique enrollment key
student_info['enroll_key'] = (
    student_info['id_student'].astype(str) + '_' +
    student_info['code_module'] + '_' +
    student_info['code_presentation']
)
svle['enroll_key'] = (
    svle['id_student'].astype(str) + '_' +
    svle['code_module'] + '_' +
    svle['code_presentation']
)

label_map = student_info.set_index('enroll_key')['dropout'].to_dict()

# Get all enrollments that have VLE interactions
valid_enrolls = svle['enroll_key'].unique()
print(f"  Enrollments with VLE data: {len(valid_enrolls):,}")

# ======================== 6. Build tensor ========================
print(f"\n[5/7] Building tensor (N, {WEEK_COUNT}, {DAYS_PER_WEEK}, {NUM_ACTIVITIES})...")

# Aggregate: sum_click per (enroll_key, date, activity_type)
agg = svle.groupby(['enroll_key', 'date', 'activity_type'])['sum_click'].sum().reset_index()

# Build the tensor
enroll_to_idx = {e: i for i, e in enumerate(valid_enrolls)}
N = len(valid_enrolls)
tensor = np.zeros((N, STA_DAY, NUM_ACTIVITIES), dtype=np.float32)

for _, row in agg.iterrows():
    ei = enroll_to_idx.get(row['enroll_key'])
    if ei is None:
        continue
    day = int(row['date'])
    ai = activity_to_idx.get(row['activity_type'])
    if ai is not None and 0 <= day < STA_DAY:
        tensor[ei, day, ai] += row['sum_click']

# Reshape to (N, week_count, days_per_week, activity_num)
tensor = tensor.reshape(N, WEEK_COUNT, DAYS_PER_WEEK, NUM_ACTIVITIES)

# Get labels
labels = np.array([label_map.get(e, 0) for e in valid_enrolls], dtype=np.float32)

print(f"  Tensor shape: {tensor.shape}")
print(f"  Labels: {int(labels.sum())} dropout / {N} total ({labels.mean():.3f})")

# ======================== 7. Train/test split ========================
print(f"\n[6/7] Stratified train/test split ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")

train_idx, test_idx = train_test_split(
    np.arange(N), test_size=TEST_SIZE,
    stratify=labels, random_state=RANDOM_SEED
)

train_data = tensor[train_idx]
train_labels = labels[train_idx]
test_data = tensor[test_idx]
test_labels = labels[test_idx]

print(f"  Train: {len(train_idx):,} (dropout rate: {train_labels.mean():.3f})")
print(f"  Test:  {len(test_idx):,} (dropout rate: {test_labels.mean():.3f})")

# ======================== 8. Normalize ========================
print("\n[7/7] StandardScaler normalization + saving .npz...")

# Flatten for scaling → scale per feature across all timesteps
train_flat = train_data.reshape(len(train_idx), -1)
test_flat = test_data.reshape(len(test_idx), -1)

scaler = StandardScaler()
train_flat = scaler.fit_transform(train_flat).astype(np.float32)
test_flat = scaler.transform(test_flat).astype(np.float32)

# Reshape back
train_data = train_flat.reshape(len(train_idx), WEEK_COUNT, DAYS_PER_WEEK, NUM_ACTIVITIES)
test_data = test_flat.reshape(len(test_idx), WEEK_COUNT, DAYS_PER_WEEK, NUM_ACTIVITIES)

# Save
output_path = os.path.join(OUTPUT_DIR, 'oulad_data_std.npz')
np.savez(
    output_path,
    t_data=train_data, t_label=train_labels,
    v_data=test_data, v_label=test_labels
)

file_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"  Saved: {output_path} ({file_size:.1f} MB)")
print(f"  Train shape: {train_data.shape}, Test shape: {test_data.shape}")

print("\n" + "=" * 60)
print("  ✅ OULAD preprocessing complete!")
print(f"  Output: {output_path}")
print(f"  Tensor: (N, {WEEK_COUNT}, {DAYS_PER_WEEK}, {NUM_ACTIVITIES})")
print("=" * 60)
