"""
Data Validator — Kiểm tra data files tồn tại và đúng cấu trúc.

Logic:
    1. Kiểm tra file tồn tại
    2. Kiểm tra cấu trúc (shape, keys, attributes)
    3. Nếu OK → skip (return True)
    4. Nếu FAIL → báo lỗi + hướng dẫn chạy lại (return False)

Dùng bởi: train.py, run_mode.sh (qua CLI)

Sử dụng CLI:
    python -m src.data_validator --mode default --dataset xuetangx
    python -m src.data_validator --mode siamese_lstm --dataset snap
    python -m src.data_validator --mode mba_cnn --dataset oulad --datastore ./datastore
"""
import os
import sys
import pickle
import numpy as np


def validate_npz(npz_path, dataset_config):
    """Kiểm tra file .npz tồn tại và có đúng cấu trúc.

    Cấu trúc yêu cầu:
        t_data:  (N_train, week_count, days_per_week, activity_num)
        t_label: (N_train,)
        v_data:  (N_test, week_count, days_per_week, activity_num)
        v_label: (N_test,)

    Returns:
        (is_valid: bool, message: str)
    """
    if not os.path.exists(npz_path):
        return False, f"File không tồn tại: {npz_path}"

    try:
        data = np.load(npz_path)
    except Exception as e:
        return False, f"Không thể đọc file: {e}"

    # Check required keys
    required_keys = {'t_data', 't_label', 'v_data', 'v_label'}
    actual_keys = set(data.files)
    missing = required_keys - actual_keys
    if missing:
        return False, f"Thiếu keys: {missing}. Có: {actual_keys}"

    # Check shapes
    wc = dataset_config['week_count']
    dpw = dataset_config['days_per_week']
    act = dataset_config['activity_num']
    expected_shape_suffix = (wc, dpw, act)

    t_data = data['t_data']
    v_data = data['v_data']
    t_label = data['t_label']
    v_label = data['v_label']

    if t_data.ndim != 4:
        return False, f"t_data phải là 4D, got {t_data.ndim}D shape={t_data.shape}"
    if t_data.shape[1:] != expected_shape_suffix:
        return False, f"t_data shape[1:]={t_data.shape[1:]} ≠ expected ({wc},{dpw},{act})"
    if v_data.ndim != 4:
        return False, f"v_data phải là 4D, got {v_data.ndim}D shape={v_data.shape}"
    if v_data.shape[1:] != expected_shape_suffix:
        return False, f"v_data shape[1:]={v_data.shape[1:]} ≠ expected ({wc},{dpw},{act})"
    if t_label.ndim != 1:
        return False, f"t_label phải là 1D, got {t_label.ndim}D"
    if v_label.ndim != 1:
        return False, f"v_label phải là 1D, got {v_label.ndim}D"
    if t_data.shape[0] != t_label.shape[0]:
        return False, f"t_data N={t_data.shape[0]} ≠ t_label N={t_label.shape[0]}"
    if v_data.shape[0] != v_label.shape[0]:
        return False, f"v_data N={v_data.shape[0]} ≠ v_label N={v_label.shape[0]}"

    N_train = t_data.shape[0]
    N_test = v_data.shape[0]
    n_pos_train = int(t_label.sum())
    n_pos_test = int(v_label.sum())

    msg = (f"✅ NPZ OK: train={N_train} (pos={n_pos_train}), "
           f"test={N_test} (pos={n_pos_test}), "
           f"shape=({wc},{dpw},{act})")
    return True, msg


def validate_graph(graph_path, dataset_config):
    """Kiểm tra file graph .pkl tồn tại và có đúng cấu trúc.

    Cấu trúc yêu cầu (PyG Data object):
        - edge_index:        (2, num_edges) LongTensor
        - labels:            (N,) LongTensor
        - seq_feat:          (N, seq_len) FloatTensor
        - org_context:       (N, context_dim) FloatTensor     (optional nhưng nên có)
        - enhanced_context:  (N, context_dim) FloatTensor     (optional nhưng nên có)
        - train_mask:        LongTensor indices
        - test_mask:         LongTensor indices

    Returns:
        (is_valid: bool, message: str)
    """
    if not os.path.exists(graph_path):
        return False, f"File không tồn tại: {graph_path}"

    try:
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
    except Exception as e:
        return False, f"Không thể đọc file: {e}"

    # Check required attributes
    required_attrs = ['edge_index', 'labels', 'seq_feat', 'train_mask', 'test_mask']
    missing = [a for a in required_attrs if not hasattr(graph, a)]
    if missing:
        return False, f"Graph thiếu attributes: {missing}"

    # Check edge_index shape
    import torch
    ei = graph.edge_index
    if not isinstance(ei, torch.Tensor):
        return False, f"edge_index phải là Tensor, got {type(ei)}"
    if ei.dim() != 2 or ei.shape[0] != 2:
        return False, f"edge_index shape phải là (2, E), got {tuple(ei.shape)}"

    # Check labels
    N = graph.labels.shape[0]
    n_edges = ei.shape[1]

    # Check seq_feat
    sf = graph.seq_feat
    if sf.shape[0] != N:
        return False, f"seq_feat N={sf.shape[0]} ≠ labels N={N}"

    # Check masks
    n_train = len(graph.train_mask)
    n_test = len(graph.test_mask)

    # Check context (optional but recommended)
    has_org = hasattr(graph, 'org_context')
    has_enh = hasattr(graph, 'enhanced_context')
    ctx_msg = ""
    if has_org:
        ctx_msg += f", org_ctx={tuple(graph.org_context.shape)}"
    if has_enh:
        ctx_msg += f", enh_ctx={tuple(graph.enhanced_context.shape)}"

    n_pos = int(graph.labels.sum())
    msg = (f"✅ Graph OK: N={N}, edges={n_edges}, "
           f"train={n_train}, test={n_test}, "
           f"pos={n_pos} ({n_pos/N:.3f}), "
           f"seq_feat={tuple(sf.shape)}{ctx_msg}")
    return True, msg


def validate_data_for_mode(mode, dataset_name, datastore_dir):
    """Kiểm tra tất cả data files cần cho mode + dataset.

    Args:
        mode: training mode (ví dụ: 'default', 'siamese_lstm', ...)
        dataset_name: 'xuetangx', 'oulad', 'snap'
        datastore_dir: path đến thư mục datastore/

    Returns:
        (all_valid: bool, results: dict)
        results = {
            'npz': (is_valid, message),
            'graph': (is_valid, message),  # chỉ có nếu graph mode
        }
    """
    from src.dataset_config import get_dataset_config
    from src.mode_registry import is_graph_mode, get_required_data_files

    ds_config = get_dataset_config(dataset_name)
    required = get_required_data_files(mode, dataset_name)

    results = {}
    all_valid = True

    # Always check NPZ
    npz_path = os.path.join(datastore_dir, required['npz'])
    valid, msg = validate_npz(npz_path, ds_config)
    results['npz'] = (valid, msg)
    if not valid:
        all_valid = False

    # Check graph only if graph mode
    if 'graph' in required:
        graph_path = os.path.join(datastore_dir, required['graph'])
        valid, msg = validate_graph(graph_path, ds_config)
        results['graph'] = (valid, msg)
        if not valid:
            all_valid = False

    return all_valid, results


def get_regeneration_command(mode, dataset_name, datastore_dir, results):
    """Trả hướng dẫn chạy lại data nếu validation fail.

    Returns:
        list of (description, command) tuples
    """
    from src.mode_registry import is_graph_mode

    commands = []

    npz_valid = results.get('npz', (True, ''))[0]
    graph_valid = results.get('graph', (True, ''))[0]

    if not npz_valid:
        if dataset_name == 'xuetangx':
            commands.append((
                "Chạy XuetangX feature extraction pipeline:",
                "bash run_mode.sh default --clean --dataset xuetangx"
            ))
        elif dataset_name == 'oulad':
            commands.append((
                "Chạy OULAD preprocessing:",
                "python src/dataprocess/oulad_preprocess.py"
            ))
        elif dataset_name == 'snap':
            commands.append((
                "Chạy SNAP preprocessing:",
                "python src/dataprocess/snap_preprocess.py"
            ))

    if not graph_valid and is_graph_mode(mode):
        if dataset_name == 'xuetangx':
            commands.append((
                "Chạy XuetangX graph generation:",
                "cd src/graphgeneration && python 0_create_strong_classmates_graph.py && python 1_graph_addition_information.py"
            ))
        elif dataset_name == 'snap':
            commands.append((
                "Chạy SNAP graph construction:",
                "python src/dataprocess/snap_build_graph.py"
            ))
        elif dataset_name == 'oulad':
            commands.append((
                "Chạy OULAD graph construction (nếu đã implement):",
                "python src/dataprocess/oulad_build_graph.py"
            ))

    return commands


# ============================================================
# CLI Interface — cho shell scripts gọi
# ============================================================
def main():
    """CLI entry point: python -m src.data_validator --mode X --dataset Y"""
    import argparse
    from src.mode_registry import ALL_MODES, describe_mode, is_graph_mode

    parser = argparse.ArgumentParser(description='Validate data files for DP-SCL training')
    parser.add_argument('--mode', type=str, required=True, choices=sorted(ALL_MODES),
                        help='Training mode')
    parser.add_argument('--dataset', type=str, default='xuetangx', choices=['xuetangx', 'oulad', 'snap'],
                        help='Dataset name')
    parser.add_argument('--datastore', type=str, default=None,
                        help='Path to datastore directory (default: ./datastore)')
    parser.add_argument('--quiet', action='store_true', help='Only output exit code')
    args = parser.parse_args()

    if args.datastore is None:
        # Auto-detect: script may be called from project root or elsewhere
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        args.datastore = os.path.join(project_dir, 'datastore')

    if not args.quiet:
        branch = "GRAPH" if is_graph_mode(args.mode) else "NO-GRAPH"
        print(f"\n{'─'*60}")
        print(f"  Data Validation: mode={args.mode}, dataset={args.dataset}")
        print(f"  Branch: {branch}")
        print(f"  Datastore: {args.datastore}")
        print(f"{'─'*60}")

    all_valid, results = validate_data_for_mode(args.mode, args.dataset, args.datastore)

    if not args.quiet:
        for key, (valid, msg) in results.items():
            status = "✅" if valid else "❌"
            print(f"  [{key.upper():>5}] {status} {msg}")

        if all_valid:
            print(f"\n  ✅ Tất cả data đã sẵn sàng! Có thể bắt đầu training.")
        else:
            print(f"\n  ❌ Data chưa sẵn sàng. Cần chạy:")
            commands = get_regeneration_command(args.mode, args.dataset, args.datastore, results)
            for desc, cmd in commands:
                print(f"    {desc}")
                print(f"      $ {cmd}")

        print(f"{'─'*60}\n")

    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
