import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


DEFAULT_SEEDS = [1, 11, 111, 1111, 11111]


@dataclass(frozen=True)
class ProtocolRun:
    run_name: str
    model_name: str
    mode: str
    lambda_con: float
    temperature: float


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-indir", type=str, default=".")
    parser.add_argument("-outdir", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="xuetangx", choices=["xuetangx", "oulad", "snap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--cls-layers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="rerun even when result CSV already exists")
    parser.add_argument("--dry-run", action="store_true", help="print commands without launching training")


def result_dir(outdir: str, run_name: str) -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(outdir)), "results", f"experiment_{run_name}")


def run_protocol(protocol: ProtocolRun, args: argparse.Namespace) -> str:
    run_dir = result_dir(args.outdir, protocol.run_name)
    summary_csv = os.path.join(run_dir, "summary_results.csv")
    epoch_history_csv = os.path.join(run_dir, "epoch_history.csv")
    if os.path.exists(summary_csv) and not args.force:
        if not os.path.exists(epoch_history_csv):
            print(f"Existing result missing epoch_history.csv; rerunning: {run_dir}")
        else:
            config_path = os.path.join(run_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as handle:
                    config = json.load(handle)
                expected = {
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "hidden_size": args.hidden_size,
                    "lambda_con": protocol.lambda_con,
                    "temperature": protocol.temperature,
                    "proposed_name": protocol.model_name,
                    "proposed_mode": protocol.mode,
                    "seeds": args.seeds,
                    "mask_ratio": args.mask_ratio,
                    "noise_std": args.noise_std,
                    "num_layers": args.num_layers,
                    "cls_layers": args.cls_layers,
                }
                mismatches = [
                    key for key, value in expected.items()
                    if config.get(key) != value
                ]
                if not mismatches:
                    print(f"Reuse existing: {run_dir}")
                    return run_dir
                print(f"Existing result config mismatch ({', '.join(mismatches)}); rerunning: {run_dir}")
            else:
                print(f"Existing result missing config.json; rerunning: {run_dir}")
    command = [
        sys.executable,
        "train_experiment.py",
        "-indir", args.indir,
        "-outdir", args.outdir,
        "--dataset", args.dataset,
        "--models", "proposed",
        "--proposed-name", protocol.model_name,
        "--proposed-mode", protocol.mode,
        "--seeds", *[str(seed) for seed in args.seeds],
        "--split", "0.60", "0.10", "0.30",
        "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--hidden-size", str(args.hidden_size),
        "--lambda-con", str(protocol.lambda_con),
        "--temperature", str(protocol.temperature),
        "--mask-ratio", str(args.mask_ratio),
        "--noise-std", str(args.noise_std),
        "--num-layers", str(args.num_layers),
        "--cls-layers", str(args.cls_layers),
        "--num-workers", str(args.num_workers),
        "--run-name", protocol.run_name,
    ]
    print("CMD:", " ".join(command))
    if args.dry_run:
        return run_dir
    subprocess.run(command, check=True)
    return run_dir


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def get_model_rows(run_dir: str, model_name: str) -> List[Dict[str, str]]:
    rows = read_csv_rows(os.path.join(run_dir, "per_seed_results.csv"))
    return [row for row in rows if row["model"] == model_name and row["status"] == "ok"]


def get_model_summary(run_dir: str, model_name: str) -> Dict[str, str]:
    rows = read_csv_rows(os.path.join(run_dir, "summary_results.csv"))
    for row in rows:
        if row["model"] == model_name:
            return row
    raise RuntimeError(f"No summary row for {model_name} in {run_dir}")


def f4(value: str) -> str:
    return f"{float(value):.4f}"


def mean_std(summary: Dict[str, str], metric: str) -> str:
    return f"{f4(summary[f'{metric}_mean'])} ± {f4(summary[f'{metric}_std'])}"


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} ± {std_value:.4f}"


def ensure_result_write(outdir: str) -> str:
    path = os.path.join(os.path.abspath(os.path.expanduser(outdir)), "result_write")
    os.makedirs(path, exist_ok=True)
    return path


def write_combined_epoch_history(path: str, run_dirs_by_value: Dict[float, str], symbol: str) -> None:
    rows = []
    for value, run_dir in run_dirs_by_value.items():
        history_path = os.path.join(run_dir, "epoch_history.csv")
        if not os.path.exists(history_path):
            raise RuntimeError(f"Missing epoch history: {history_path}")
        for row in read_csv_rows(history_path):
            row = dict(row)
            row[symbol] = value
            row["run_dir"] = run_dir
            rows.append(row)

    fieldnames = [symbol, "run_dir"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_values(rows: List[Dict[str, str]], key: str) -> List[float]:
    values = []
    for row in rows:
        raw_value = row.get(key, "")
        if raw_value == "":
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value == value:
            values.append(value)
    return values


def summarize_epoch_history(run_dir: str, model_name: str) -> Dict[str, float]:
    rows = [
        row for row in read_csv_rows(os.path.join(run_dir, "epoch_history.csv"))
        if row["model"] == model_name
    ]
    if not rows:
        raise RuntimeError(f"No epoch history rows for {model_name} in {run_dir}")

    summary: Dict[str, float] = {"epoch_rows": float(len(rows))}
    for metric in (
        "train_loss",
        "train_bce_loss",
        "train_supcon_loss",
        "val_auc",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
    ):
        values = _float_values(rows, metric)
        if not values:
            continue
        mean_value = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
            std_value = variance ** 0.5
        else:
            std_value = 0.0
        summary[f"{metric}_mean"] = mean_value
        summary[f"{metric}_std"] = std_value
    return summary


def write_ablation_report(path: str, model_name: str, rows: List[Dict[str, str]], summary: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"MODEL: {model_name}\n")
        for row in rows:
            handle.write(
                f"  Seed {int(row['seed']):>6} | "
                f"best_epoch={row['best_epoch']} stopped={row['stopped_epoch']} | "
                f"AUC={f4(row['test_auc'])} F1={f4(row['test_f1'])}\n"
            )
        handle.write(
            f"  FINAL | AUC={mean_std(summary, 'auc')} | "
            f"F1={mean_std(summary, 'f1')}\n"
        )


def write_sensitivity_report(
    path: str,
    title: str,
    symbol: str,
    summaries: Dict[float, Dict[str, str]],
    order: Iterable[float],
    reference: Optional[Dict[str, str]] = None,
    best_value: float = 0.1,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        for value in order:
            if reference and abs(value - float(reference["value"])) < 1e-12:
                auc_text = reference["auc"]
                f1_text = reference["f1"]
            else:
                summary = summaries[value]
                auc_text = mean_std(summary, "auc")
                f1_text = mean_std(summary, "f1")
            suffix = "  <- best" if abs(value - best_value) < 1e-12 else ""
            handle.write(f"{symbol}={value:.2f} | AUC={auc_text} | F1={f1_text}{suffix}\n")


def write_epoch_average_sensitivity_report(
    path: str,
    title: str,
    symbol: str,
    summaries: Dict[float, Dict[str, float]],
    order: Iterable[float],
    best_value: float,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("Result: mean ± std over all epochs and seeds from epoch_history.csv\n")
        for value in order:
            summary = summaries[value]
            auc_text = format_mean_std(summary["val_auc_mean"], summary["val_auc_std"])
            f1_text = format_mean_std(summary["val_f1_mean"], summary["val_f1_std"])
            acc_text = format_mean_std(summary["val_acc_mean"], summary["val_acc_std"])
            precision_text = format_mean_std(summary["val_precision_mean"], summary["val_precision_std"])
            recall_text = format_mean_std(summary["val_recall_mean"], summary["val_recall_std"])
            loss_text = format_mean_std(summary["train_loss_mean"], summary["train_loss_std"])
            suffix = "  <- best mean AUC" if abs(value - best_value) < 1e-12 else ""
            handle.write(
                f"{symbol}={value:.2f} | "
                f"Val AUC={auc_text} | Val F1={f1_text} | Val ACC={acc_text} | "
                f"Val Precision={precision_text} | Val Recall={recall_text} | "
                f"Train Loss={loss_text} | epoch_rows={int(summary['epoch_rows'])}{suffix}\n"
            )


def best_value_by_metric(summaries: Dict[float, Dict[str, str]], metric: str = "auc") -> float:
    return max(summaries, key=lambda value: float(summaries[value][f"{metric}_mean"]))


def best_epoch_average_by_metric(summaries: Dict[float, Dict[str, float]], metric: str = "val_auc") -> float:
    return max(summaries, key=lambda value: summaries[value][f"{metric}_mean"])
