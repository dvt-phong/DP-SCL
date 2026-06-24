"""
Inspect a XuetangX temporal NPZ sample for the DP-SCL pipeline.

This script is read-only. It prints real examples from datastore/all_data_std.npz
when the dataset is available locally.
"""

import argparse
import os
import sys

import numpy as np


ACTIONS = [
    "seek_video",
    "play_video",
    "pause_video",
    "stop_video",
    "load_video",
    "problem_get",
    "problem_check",
    "problem_save",
    "reset_problem",
    "problem_check_correct",
    "problem_check_incorrect",
    "create_thread",
    "create_comment",
    "delete_thread",
    "delete_comment",
    "close_forum",
    "click_info",
    "click_courseware",
    "click_about",
    "click_forum",
    "click_progress",
    "close_courseware",
]


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def label_summary(labels):
    labels = np.asarray(labels).astype(int)
    total = len(labels)
    pos = int(labels.sum())
    neg = total - pos
    ratio = pos / total if total else 0.0
    return f"total={total} dropout/1={pos} non_dropout/0={neg} dropout_ratio={ratio:.4f}"


def print_npz_summary(data):
    print("NPZ keys:", ", ".join(data.files))
    for key in ["t_data", "t_label", "v_data", "v_label", "t_context", "v_context"]:
        if key in data.files:
            arr = data[key]
            print(f"{key}: shape={arr.shape} dtype={arr.dtype}")
    print("train labels:", label_summary(data["t_label"]))
    print("test labels: ", label_summary(data["v_label"]))


def select_split(data, split):
    if split == "train":
        return data["t_data"], data["t_label"], "train"
    if split == "test":
        return data["v_data"], data["v_label"], "test"
    x = np.concatenate([data["t_data"], data["v_data"]], axis=0)
    y = np.concatenate([data["t_label"], data["v_label"]], axis=0)
    return x, y, "all"


def describe_sample(x, y, split_name, sample_index, top_k):
    if x.ndim != 4 or x.shape[1:] != (5, 7, 22):
        raise ValueError(f"Expected sample shape suffix (5, 7, 22), got {x.shape}")
    if not 0 <= sample_index < len(x):
        raise IndexError(f"sample-index must be in [0, {len(x) - 1}], got {sample_index}")

    sample = x[sample_index]
    label = int(y[sample_index])
    days = sample.reshape(35, 22)
    flat = sample.reshape(-1)

    print()
    print(f"Sample split={split_name} index={sample_index} label={label}")
    print(f"sample tensor shape: {sample.shape}")
    print(f"flattened model input shape: {flat.shape}")
    print(f"model sequence view shape: {days.shape}")
    print(f"total events in first 35 days: {float(days.sum()):.4g}")

    action_totals = days.sum(axis=0)
    ranked_actions = sorted(
        [(ACTIONS[i], float(value)) for i, value in enumerate(action_totals) if value != 0],
        key=lambda item: item[1],
        reverse=True,
    )
    print()
    print("Top action totals across 35 days:")
    if ranked_actions:
        for name, value in ranked_actions[:top_k]:
            print(f"  {name:<25} {value:.4g}")
    else:
        print("  no non-zero actions")

    nonzero_events = []
    for day_idx in range(35):
        for action_idx, value in enumerate(days[day_idx]):
            if value != 0:
                week = day_idx // 7
                day_in_week = day_idx % 7
                nonzero_events.append((float(value), day_idx, week, day_in_week, ACTIONS[action_idx]))
    nonzero_events.sort(reverse=True)

    print()
    print(f"Top {top_k} non-zero day/action cells:")
    if nonzero_events:
        for value, day_idx, week, day_in_week, action in nonzero_events[:top_k]:
            print(f"  day={day_idx:02d} week={week} day_in_week={day_in_week} action={action:<25} count={value:.4g}")
    else:
        print("  no non-zero day/action cells")

    print()
    print("Daily activity summary:")
    for day_idx, row in enumerate(days):
        total = float(row.sum())
        if total == 0:
            continue
        active = [f"{ACTIONS[i]}={float(v):.4g}" for i, v in enumerate(row) if v != 0]
        print(f"  day={day_idx:02d} total={total:.4g} | " + ", ".join(active))


def parse_args():
    default_npz = os.path.join(project_root(), "datastore", "all_data_std.npz")
    parser = argparse.ArgumentParser(description="Inspect real XuetangX samples for DP-SCL.")
    parser.add_argument("--npz", default=default_npz, help="Path to all_data_std.npz")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.npz):
        print(f"Missing NPZ file: {args.npz}")
        print("Place XuetangX temporal data at datastore/all_data_std.npz or pass --npz PATH.")
        return 1

    data = np.load(args.npz, allow_pickle=False)
    required = {"t_data", "t_label", "v_data", "v_label"}
    missing = required - set(data.files)
    if missing:
        print(f"Missing required keys: {sorted(missing)}")
        print(f"Found keys: {sorted(data.files)}")
        return 1

    print_npz_summary(data)
    x, y, split_name = select_split(data, args.split)
    describe_sample(x, y, split_name, args.sample_index, args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
