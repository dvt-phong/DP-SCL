"""
Origin and attribution:
  Project: DP-SCL.
  Purpose: Dataset metadata for temporal NPZ files used by DP-SCL.

Reference sources:
  The XuetangX/KDD Cup entry is compatible with the adapted CA-TFHN
  preprocessing scripts, which use 35 days = 5 weeks x 7 days and 22 action
  types:
  https://github.com/codeds27/CA-TFHN
  OULAD and SNAP entries are DP-SCL additions for cross-dataset experiments.
"""

DATASET_CONFIGS = {
    "xuetangx": {
        "name": "XuetangX (KDD Cup 2015)",
        "npz_filename": "all_data_std.npz",
        "activity_num": 22,
        "sta_day": 35,
        "week_count": 5,
        "days_per_week": 7,
        "description": "22 activity types x 35 days from Chinese MOOC platform XuetangX",
    },
    "oulad": {
        "name": "OULAD (Open University)",
        "npz_filename": "oulad_data_std.npz",
        "activity_num": 20,
        "sta_day": 35,
        "week_count": 5,
        "days_per_week": 7,
        "description": "20 VLE activity types x 35 days from UK Open University",
    },
    "snap": {
        "name": "SNAP MOOC (Stanford ACT-MOOC)",
        "npz_filename": "snap_data_std.npz",
        "activity_num": 6,
        "sta_day": 35,
        "week_count": 5,
        "days_per_week": 7,
        "description": "6 engineered features x 35 time bins from Stanford MOOC",
    },
}


def get_dataset_config(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        valid = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid: {valid}")
    return DATASET_CONFIGS[dataset_name]
