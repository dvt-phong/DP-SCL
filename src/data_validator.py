import argparse
import os
import sys

import numpy as np


def validate_npz(npz_path, dataset_config):
    if not os.path.exists(npz_path):
        return False, f"File does not exist: {npz_path}"

    try:
        data = np.load(npz_path)
    except Exception as exc:
        return False, f"Cannot read file: {exc}"

    required_keys = {"t_data", "t_label", "v_data", "v_label"}
    missing = required_keys - set(data.files)
    if missing:
        return False, f"Missing keys: {sorted(missing)}. Found: {sorted(data.files)}"

    wc = dataset_config["week_count"]
    dpw = dataset_config["days_per_week"]
    act = dataset_config["activity_num"]
    expected_suffix = (wc, dpw, act)

    t_data = data["t_data"]
    v_data = data["v_data"]
    t_label = data["t_label"]
    v_label = data["v_label"]

    if t_data.ndim != 4 or t_data.shape[1:] != expected_suffix:
        return False, f"t_data must have shape (N,{wc},{dpw},{act}), got {t_data.shape}"
    if v_data.ndim != 4 or v_data.shape[1:] != expected_suffix:
        return False, f"v_data must have shape (N,{wc},{dpw},{act}), got {v_data.shape}"
    if t_label.ndim != 1 or len(t_label) != len(t_data):
        return False, f"t_label must be 1D and match t_data length, got {t_label.shape}"
    if v_label.ndim != 1 or len(v_label) != len(v_data):
        return False, f"v_label must be 1D and match v_data length, got {v_label.shape}"

    return (
        True,
        f"NPZ OK: train={len(t_label)} test={len(v_label)} shape=({wc},{dpw},{act})",
    )


def validate_data_for_mode(mode, dataset_name, datastore_dir):
    from src.dataset_config import get_dataset_config
    from src.mode_registry import ALL_MODES, get_required_data_files

    if mode not in ALL_MODES:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {sorted(ALL_MODES)}")

    ds_config = get_dataset_config(dataset_name)
    required = get_required_data_files(mode, dataset_name)
    npz_path = os.path.join(datastore_dir, required["npz"])
    valid, message = validate_npz(npz_path, ds_config)
    return valid, {"npz": (valid, message)}


def get_regeneration_command(mode, dataset_name, datastore_dir, results):
    if results.get("npz", (True, ""))[0]:
        return []
    if dataset_name == "oulad":
        return [("Run OULAD preprocessing:", "python src/dataprocess/oulad_preprocess.py")]
    if dataset_name == "snap":
        return [("Run SNAP preprocessing:", "python src/dataprocess/snap_preprocess.py")]
    return [("Prepare XuetangX temporal NPZ:", "place all_data_std.npz under datastore/")]


def main():
    from src.mode_registry import ALL_MODES, describe_mode

    parser = argparse.ArgumentParser(description="Validate temporal data for DP-SCL")
    parser.add_argument("--mode", type=str, default="dp_scl", choices=sorted(ALL_MODES))
    parser.add_argument("--dataset", type=str, default="xuetangx", choices=["xuetangx", "oulad", "snap"])
    parser.add_argument("--datastore", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.datastore is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.datastore = os.path.join(project_dir, "datastore")

    valid, results = validate_data_for_mode(args.mode, args.dataset, args.datastore)
    if not args.quiet:
        print(f"Data validation: {describe_mode(args.mode)} | dataset={args.dataset}")
        print(f"Datastore: {args.datastore}")
        for key, (ok, msg) in results.items():
            print(f"  [{key.upper()}] {'OK' if ok else 'FAIL'} {msg}")
        if not valid:
            for desc, cmd in get_regeneration_command(args.mode, args.dataset, args.datastore, results):
                print(f"  {desc}\n    {cmd}")
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
