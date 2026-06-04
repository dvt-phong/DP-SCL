DP_SCL_MODE = "dp_scl"
DP_SCL_BACKEND_MODE = "supcon_lstm_attn"
ALL_MODES = frozenset({DP_SCL_MODE})


def get_framework(mode):
    if mode == DP_SCL_MODE:
        return "dp_scl"
    raise ValueError(f"Unknown mode: {mode}. Valid modes: {sorted(ALL_MODES)}")


def resolve_backend_mode(mode):
    if mode == DP_SCL_MODE:
        return DP_SCL_BACKEND_MODE
    raise ValueError(f"Unknown mode: {mode}. Valid modes: {sorted(ALL_MODES)}")


def get_required_data_files(mode, dataset_name):
    if mode not in ALL_MODES:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {sorted(ALL_MODES)}")
    from src.dataset_config import get_dataset_config

    ds = get_dataset_config(dataset_name)
    return {"npz": ds["npz_filename"]}


def describe_mode(mode):
    if mode != DP_SCL_MODE:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {sorted(ALL_MODES)}")
    return "[TEMPORAL] DP-SCL"


def print_mode_table():
    print("DP-SCL modes:")
    for mode in sorted(ALL_MODES):
        print(f"  {mode:<12} {describe_mode(mode)}")
