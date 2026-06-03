"""
Dataset Configuration for DP-SCL Multi-Dataset Support.

Mỗi dataset được tiền xử lý thành cùng 1 định dạng tensor:
    (N, week_count * days_per_week, activity_num) = (N, T, F)
với T=35 cho tất cả datasets, chỉ khác activity_num (F).

Output .npz format:
    t_data:  (N_train, week_count, days_per_week, activity_num)
    t_label: (N_train,)
    v_data:  (N_test, week_count, days_per_week, activity_num)
    v_label: (N_test,)
"""

DATASET_CONFIGS = {
    'xuetangx': {
        'name': 'XuetangX (KDD Cup 2015)',
        'npz_filename': 'all_data_std.npz',
        'graph_filename': 'StrongClassmatesGraph.pkl',
        'activity_num': 22,
        'sta_day': 35,
        'week_count': 5,
        'days_per_week': 7,
        'description': '22 activity types × 35 days from Chinese MOOC platform XuetangX',
    },
    'oulad': {
        'name': 'OULAD (Open University)',
        'npz_filename': 'oulad_data_std.npz',
        'graph_filename': 'oulad_StrongClassmatesGraph.pkl',
        'activity_num': 20,
        'sta_day': 35,
        'week_count': 5,
        'days_per_week': 7,
        'description': '20 VLE activity types × 35 days from UK Open University',
    },
    'snap': {
        'name': 'SNAP MOOC (Stanford ACT-MOOC)',
        'npz_filename': 'snap_data_std.npz',
        'graph_filename': 'snap_StrongClassmatesGraph.pkl',
        'activity_num': 6,
        'sta_day': 35,
        'week_count': 5,
        'days_per_week': 7,
        'description': '6 engineered features × 35 time bins from Stanford MOOC',
    },
}


def get_dataset_config(dataset_name):
    """Get config dict for a dataset. Raises ValueError if not found."""
    if dataset_name not in DATASET_CONFIGS:
        valid = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid: {valid}")
    return DATASET_CONFIGS[dataset_name]
