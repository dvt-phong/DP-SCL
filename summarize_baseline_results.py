import argparse
import csv
import glob
import os
import re
from datetime import datetime


METRIC_KEYS = ("auc", "acc", "f1", "precision", "recall", "threshold")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ML and DL baseline result files")
    parser.add_argument("-outdir", type=str, default=".", help="output dir containing results/")
    parser.add_argument("--dataset", type=str, default=None, help="dataset prefix to filter")
    return parser.parse_args()


def parse_ml_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) != 9 or parts[0] == "method":
                continue
            metrics = dict(zip(METRIC_KEYS, [_to_float(value) for value in parts[1:7]]))
            rows.append({
                "method": parts[0],
                "source": path,
                "status": parts[8],
                **metrics,
            })
    return rows


def parse_dl_file(path):
    text = _read(path)
    mode = _match(text, r"Mode:\s+([^\n]+)")
    if not mode:
        mode = os.path.basename(os.path.dirname(path))

    return [{
        "method": mode.strip(),
        "source": path,
        "status": "ok",
        "auc": _match_float(text, r"AUC:\s+([0-9.]+)"),
        "acc": _match_float(text, r"ACC:\s+([0-9.]+)"),
        "f1": _match_float(text, r"F1:\s+([0-9.]+)"),
        "precision": _match_float(text, r"Precision:\s+([0-9.]+)"),
        "recall": _match_float(text, r"Recall:\s+([0-9.]+)"),
        "threshold": _match_float(text, r"Threshold:\s+([0-9.]+)"),
    }]


def main():
    args = parse_args()
    output_dir = os.path.abspath(os.path.expanduser(args.outdir))
    results_dir = os.path.join(output_dir, "results")
    dataset_glob = f"{args.dataset}_*.txt" if args.dataset else "*.txt"

    rows = []
    for path in glob.glob(os.path.join(results_dir, "ml_baselines", dataset_glob)):
        rows.extend(parse_ml_file(path))

    for path in glob.glob(os.path.join(results_dir, "dl_*", dataset_glob)):
        rows.extend(parse_dl_file(path))

    rows.sort(key=lambda row: _sort_value(row.get("auc")), reverse=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, "result_write")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"baseline_summary_{timestamp}.csv")

    with open(result_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["method", "auc", "acc", "f1", "precision", "recall", "threshold", "status", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary: {result_path}")
    print(f"Rows: {len(rows)}")


def _read(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def _match(text, pattern):
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _match_float(text, pattern):
    value = _match(text, pattern)
    return _to_float(value)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sort_value(value):
    return -1.0 if value != value else value


if __name__ == "__main__":
    main()

