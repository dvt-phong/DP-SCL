"""Multi-seed DP-SCL experiment orchestration."""

import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from .config import MODEL_NAME
from .data import load_full_temporal_data, set_seed
from .datasets import get_dataset_config
from .metrics import fmt, summarize
from .modes import DP_SCL_MODE
from .reporting import write_csv, write_report
from .splits import class_ratio, make_split_indices, save_split_indices
from .trainer import train_dp_scl


def _build_config(args, ds_config, npz_path, sample_count, device):
    return {
        "dataset": args.dataset,
        "dataset_name": ds_config["name"],
        "npz_path": npz_path,
        "samples": int(sample_count),
        "seeds": args.seeds,
        "split": {"train": args.split[0], "val": args.split[1], "test": args.split[2]},
        "split_strategy": "stratified",
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "lambda_con": args.lambda_con,
        "temperature": args.temperature,
        "mask_ratio": args.mask_ratio,
        "noise_std": args.noise_std,
        "num_layers": args.num_layers,
        "cls_layers": args.cls_layers,
        "num_workers": args.num_workers,
        "model": MODEL_NAME,
        "mode": DP_SCL_MODE,
        "device": str(device),
    }


def run_experiment(args):
    input_dir = os.path.abspath(os.path.expanduser(args.indir))
    output_dir = os.path.abspath(os.path.expanduser(args.outdir))
    ds_config = get_dataset_config(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, "results", f"dp_scl_{timestamp}")
    split_dir = os.path.join(run_dir, "splits")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    x, y, npz_path = load_full_temporal_data(input_dir, ds_config)
    config = _build_config(args, ds_config, npz_path, len(y), device)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print(f"=== DP-SCL Experiment: {ds_config['name']} ===")
    print(f"Data: {npz_path} | X={x.shape} y={y.shape}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {run_dir}")
    print(f"Device: {device}")

    rows = []
    epoch_history_rows = []
    for seed in args.seeds:
        set_seed(seed)
        train_idx, val_idx, test_idx = make_split_indices(y, seed, args.split)
        save_split_indices(split_dir, seed, train_idx, val_idx, test_idx)
        print(
            f"\nSeed {seed}: train={class_ratio(y, train_idx)} "
            f"val={class_ratio(y, val_idx)} test={class_ratio(y, test_idx)}"
        )

        start = time.time()
        args.current_seed = seed
        try:
            result = train_dp_scl(x, y, train_idx, val_idx, test_idx, args, ds_config, device, checkpoint_dir)
        except Exception as exc:
            result = {
                "auc": np.nan,
                "acc": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "threshold": np.nan,
                "best_epoch": "",
                "stopped_epoch": "",
                "best_val_auc": np.nan,
                "status": f"failed: {type(exc).__name__}: {exc}",
                "epoch_history": [],
            }
            print(f"    FAILED: {result['status']}")

        elapsed = time.time() - start
        row = {
            "model": MODEL_NAME,
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "stopped_epoch": result["stopped_epoch"],
            "best_val_auc": result["best_val_auc"],
            "threshold": result["threshold"],
            "test_auc": result["auc"],
            "test_acc": result["acc"],
            "test_precision": result["precision"],
            "test_recall": result["recall"],
            "test_f1": result["f1"],
            "elapsed_sec": elapsed,
            "status": result["status"],
        }
        rows.append(row)
        for history_row in result.get("epoch_history", []):
            epoch_history_rows.append({
                "model": MODEL_NAME,
                "seed": seed,
                "lambda_con": args.lambda_con,
                "temperature": args.temperature,
                **history_row,
            })

        print(
            f"    status={row['status']} AUC={fmt(row['test_auc'])} "
            f"ACC={fmt(row['test_acc'])} Precision={fmt(row['test_precision'])} "
            f"Recall={fmt(row['test_recall'])} F1={fmt(row['test_f1'])} time={elapsed:.1f}s"
        )

        write_csv(
            os.path.join(run_dir, "per_seed_results.csv"),
            rows,
            [
                "model", "seed", "best_epoch", "stopped_epoch", "best_val_auc",
                "threshold", "test_auc", "test_acc", "test_precision",
                "test_recall", "test_f1", "elapsed_sec", "status",
            ],
        )
        if epoch_history_rows:
            write_csv(
                os.path.join(run_dir, "epoch_history.csv"),
                epoch_history_rows,
                [
                    "model", "seed", "lambda_con", "temperature",
                    "epoch", "train_loss", "train_bce_loss", "train_supcon_loss",
                    "val_threshold", "val_auc", "val_acc", "val_precision",
                    "val_recall", "val_f1", "best_val_auc_so_far",
                    "best_val_f1_so_far", "best_epoch_so_far",
                    "patience_count", "is_best",
                ],
            )

    summary_rows = summarize(rows)
    write_csv(
        os.path.join(run_dir, "summary_results.csv"),
        summary_rows,
        [
            "model", "auc_mean", "auc_std", "acc_mean", "acc_std",
            "precision_mean", "precision_std", "recall_mean", "recall_std",
            "f1_mean", "f1_std", "avg_best_epoch", "avg_stopped_epoch",
        ],
    )
    write_report(os.path.join(run_dir, "report.txt"), config, rows, summary_rows)

    print("\nSaved:")
    print(f"  {os.path.join(run_dir, 'per_seed_results.csv')}")
    if epoch_history_rows:
        print(f"  {os.path.join(run_dir, 'epoch_history.csv')}")
    print(f"  {os.path.join(run_dir, 'summary_results.csv')}")
    print(f"  {os.path.join(run_dir, 'report.txt')}")
    return run_dir
