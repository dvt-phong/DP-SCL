import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from train_experiment import (
    SEED_LIST,
    class_ratio,
    compute_metrics_with_threshold,
    eval_torch_model,
    fmt,
    fmt_mean_std,
    load_full_temporal_data,
    make_loaders,
    make_split_indices,
    make_dp_scl_param_dict,
    save_split_indices,
    select_threshold_by_f1,
    set_seed,
    summarize,
    write_csv,
)
from src.dataset_config import get_dataset_config
from baselines.dl import MLPBaseline
from src.models import SiameseLGB, SupConLoss


LOSS_CONFIGS = [
    {
        "key": "bce_only",
        "model_name": "MLP_BCE",
        "display_name": "MLP (BCE only)",
        "loss_text": "L_BCE",
        "lambda_con": 0.1,
    },
    {
        "key": "supcon_only",
        "model_name": "DP_SCL_SupCon_only",
        "display_name": "DP-SCL encoder (SupCon only)",
        "loss_text": "L_SupCon",
        "lambda_con": 0.1,
    },
    {
        "key": "combined",
        "model_name": "DP_SCL",
        "display_name": "DP-SCL",
        "loss_text": "L_BCE + lambda * L_SupCon",
        "lambda_con": 0.1,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DP-SCL loss ablation: BCE only, SupCon only, and BCE + SupCon."
    )
    parser.add_argument("-indir", type=str, default=".", help="input dir")
    parser.add_argument("-outdir", type=str, default=".", help="output dir")
    parser.add_argument("--dataset", type=str, default="xuetangx", choices=["xuetangx", "oulad", "snap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEED_LIST)
    parser.add_argument("--split", nargs=3, type=float, default=[0.60, 0.10, 0.30],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lambda-con", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--proposed-mode", type=str, default="dp_scl")
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--cls-layers", type=int, default=1)
    parser.add_argument("--monitor", type=str, default="auc", choices=["auc", "f1"],
                        help="validation metric for early stopping; auc is the paper default")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="dp_scl_loss_ablation")
    parser.add_argument("--force", action="store_true", help="rerun even when compatible results exist")
    parser.add_argument("--dry-run", action="store_true", help="print planned runs without training")
    return parser.parse_args()


def compute_dp_scl_loss(logits, y_batch, z1, z2, bce, supcon, loss_mode, lambda_con):
    if loss_mode == "bce_only":
        bce_loss = bce(logits, y_batch)
        return loss_mode, bce_loss

    features = torch.stack([z1, z2], dim=1)

    if loss_mode == "supcon_only":
        supcon_loss = supcon(features, y_batch.view(-1))
        return loss_mode, supcon_loss

    bce_loss = bce(logits, y_batch)
    supcon_loss = supcon(features, y_batch.view(-1))
    return loss_mode, bce_loss + lambda_con * supcon_loss


def train_dp_scl_loss_mode(model_name, loss_mode, lambda_con, X, y, train_idx, val_idx,
                        test_idx, args, ds_config, device, checkpoint_dir):
    set_seed(args.current_seed)
    train_loader, val_loader, test_loader = make_loaders(
        X, y, train_idx, val_idx, test_idx, args.batch_size, args.num_workers
    )
    if loss_mode == "bce_only":
        mlp_params = {
            "activity_num": ds_config["activity_num"],
            "week_count": ds_config["week_count"],
            "days_per_week": ds_config["days_per_week"],
            "sta_day": ds_config["sta_day"],
            "mlp_hidden_dim": 64,
            "mlp_dropout": 0.3,
        }
        model = MLPBaseline(mlp_params).to(device)
    else:
        model = SiameseLGB(mode=args.proposed_mode, param_dict=make_dp_scl_param_dict(args, ds_config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    supcon = None if loss_mode == "bce_only" else SupConLoss(temperature=args.temperature).to(device)

    safe_model_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)
    checkpoint_path = os.path.join(checkpoint_dir, f"{safe_model_name}_seed_{args.current_seed}.pt")

    best_val_auc = -np.inf
    best_val_f1 = -np.inf
    best_epoch = 0
    patience_count = 0
    stopped_epoch = args.max_epochs

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for seq_feat, y_batch in train_loader:
            seq_feat = seq_feat.to(device)
            y_batch = y_batch.to(device).view(-1, 1)
            optimizer.zero_grad()
            batch = {"batch_size": seq_feat.shape[0], "seq_feat": seq_feat}
            if loss_mode == "bce_only":
                logits = model(batch)
                _, loss = compute_dp_scl_loss(
                    logits, y_batch, None, None, bce, supcon, loss_mode, lambda_con
                )
            else:
                logits, z1, z2 = model(batch)
                _, loss = compute_dp_scl_loss(
                    logits, y_batch, z1, z2, bce, supcon, loss_mode, lambda_con
                )
            loss.backward()
            optimizer.step()

        val_y, val_score = eval_torch_model(model, val_loader, device)
        val_threshold = select_threshold_by_f1(val_y, val_score)
        val_metrics = compute_metrics_with_threshold(val_y, val_score, val_threshold)
        val_auc = val_metrics["auc"]
        val_f1 = val_metrics["f1"]

        if args.monitor == "f1":
            improved = (
                val_f1 > best_val_f1 + 1e-6 or
                (abs(val_f1 - best_val_f1) <= 1e-6 and val_auc > best_val_auc)
            )
        else:
            improved = (
                val_auc > best_val_auc + 1e-6 or
                (abs(val_auc - best_val_auc) <= 1e-6 and val_f1 > best_val_f1)
            )
        if improved:
            best_val_auc = val_auc
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1

        print(
            f"    epoch={epoch:03d} loss_mode={loss_mode} seed={args.current_seed} "
            f"monitor={args.monitor} "
            f"val_auc={val_auc:.4f} val_f1={val_f1:.4f} "
            f"best_epoch={best_epoch} patience={patience_count}/{args.patience}",
            flush=True,
        )

        if patience_count >= args.patience:
            stopped_epoch = epoch
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    val_y, val_score = eval_torch_model(model, val_loader, device)
    threshold = select_threshold_by_f1(val_y, val_score)
    test_y, test_score = eval_torch_model(model, test_loader, device)
    test_metrics = compute_metrics_with_threshold(test_y, test_score, threshold)
    return {
        **test_metrics,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_auc": best_val_auc,
        "status": "ok",
    }


def compatible_result_exists(run_dir, expected_config):
    summary_csv = os.path.join(run_dir, "summary_results.csv")
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(summary_csv) or not os.path.exists(config_path):
        return False
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    return all(config.get(key) == value for key, value in expected_config.items())


def write_report(path, config, rows, summary_rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("DP-SCL LOSS ABLATION RESULTS\n")
        handle.write(f"Dataset: {config['dataset']}\n")
        handle.write("Split: 60/10/30 stratified\n")
        handle.write(f"Seeds: {config['seeds']}\n")
        handle.write(f"Max epochs: {config['max_epochs']}\n")
        handle.write(f"Early stopping: Val {config['monitor'].upper()}, patience={config['patience']}\n")
        handle.write(f"Mode: {config['proposed_mode']}\n")
        handle.write(f"Loss mode: {config['loss_mode']}\n\n")

        for row in rows:
            if row["status"] != "ok":
                handle.write(f"Seed {row['seed']:>6} | status={row['status']}\n")
                continue
            handle.write(
                f"Seed {row['seed']:>6} | "
                f"best_epoch={row['best_epoch']} stopped={row['stopped_epoch']} | "
                f"threshold={fmt(row['threshold'])} | "
                f"AUC={fmt(row['test_auc'])} "
                f"ACC={fmt(row['test_acc'])} "
                f"Precision={fmt(row['test_precision'])} "
                f"Recall={fmt(row['test_recall'])} "
                f"F1={fmt(row['test_f1'])}\n"
            )

        if summary_rows:
            summary = summary_rows[0]
            handle.write("-" * 56 + "\n")
            handle.write(
                "FINAL | "
                f"AUC={fmt_mean_std(summary, 'auc')} | "
                f"ACC={fmt_mean_std(summary, 'acc')} | "
                f"Precision={fmt_mean_std(summary, 'precision')} | "
                f"Recall={fmt_mean_std(summary, 'recall')} | "
                f"F1={fmt_mean_std(summary, 'f1')}\n"
            )


def run_one_config(config_item, args, X, y, ds_config, device, root_run_dir):
    loss_mode = config_item["key"]
    model_name = config_item["model_name"]
    lambda_con = args.lambda_con if loss_mode == "combined" else config_item["lambda_con"]
    run_dir = os.path.join(root_run_dir, loss_mode)
    split_dir = os.path.join(run_dir, "splits")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = {
        "dataset": args.dataset,
        "dataset_name": ds_config["name"],
        "samples": int(len(y)),
        "seeds": args.seeds,
        "split": {"train": args.split[0], "val": args.split[1], "test": args.split[2]},
        "split_strategy": "stratified",
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "lambda_con": lambda_con,
        "temperature": args.temperature,
        "mask_ratio": args.mask_ratio,
        "noise_std": args.noise_std,
        "num_layers": args.num_layers,
        "cls_layers": args.cls_layers,
        "proposed_mode": args.proposed_mode,
        "loss_mode": loss_mode,
        "monitor": args.monitor,
        "models": [model_name],
        "device": str(device),
    }

    if compatible_result_exists(run_dir, config) and not args.force:
        print(f"Reuse existing: {run_dir}")
        return run_dir

    print(f"\n=== {model_name}: loss_mode={loss_mode}, lambda={lambda_con} ===")
    print(f"Output: {run_dir}")
    if args.dry_run:
        return run_dir

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        train_idx, val_idx, test_idx = make_split_indices(y, seed, args.split)
        save_split_indices(split_dir, seed, train_idx, val_idx, test_idx)
        print(
            f"\nSeed {seed}: "
            f"train={class_ratio(y, train_idx)} "
            f"val={class_ratio(y, val_idx)} "
            f"test={class_ratio(y, test_idx)}"
        )

        start = time.time()
        args.current_seed = seed
        try:
            result = train_dp_scl_loss_mode(
                model_name, loss_mode, lambda_con, X, y, train_idx, val_idx, test_idx,
                args, ds_config, device, checkpoint_dir
            )
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
            }
            print(f"    FAILED: {result['status']}")

        elapsed = time.time() - start
        row = {
            "group": "PROPOSED",
            "model": model_name,
            "seed": seed,
            "loss_mode": loss_mode,
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
        print(
            f"    status={row['status']} AUC={fmt(row['test_auc'])} "
            f"ACC={fmt(row['test_acc'])} "
            f"Precision={fmt(row['test_precision'])} "
            f"Recall={fmt(row['test_recall'])} F1={fmt(row['test_f1'])} "
            f"time={elapsed:.1f}s"
        )

        write_csv(
            os.path.join(run_dir, "per_seed_results.csv"),
            rows,
            [
                "group", "model", "seed", "loss_mode", "best_epoch", "stopped_epoch",
                "best_val_auc", "threshold", "test_auc", "test_acc", "test_precision",
                "test_recall", "test_f1", "elapsed_sec", "status",
            ],
        )

    summary_rows = summarize(rows)
    write_csv(
        os.path.join(run_dir, "summary_results.csv"),
        summary_rows,
        [
            "group", "model",
            "auc_mean", "auc_std",
            "acc_mean", "acc_std",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std",
            "avg_best_epoch", "avg_stopped_epoch",
        ],
    )
    write_report(os.path.join(run_dir, "report.txt"), config, rows, summary_rows)
    return run_dir


def read_summary(run_dir):
    path = os.path.join(run_dir, "summary_results.csv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip().split(",") for line in handle if line.strip()]
    header = lines[0]
    values = lines[1]
    return dict(zip(header, values))


def write_combined_report(path, run_dirs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("DP-SCL LOSS ABLATION\n")
        handle.write(f"{'Configuration':<22} {'Loss':<26} {'AUC':<17} {'ACC':<17} {'F1':<17}\n")
        handle.write("-" * 102 + "\n")
        for config_item, run_dir in zip(LOSS_CONFIGS, run_dirs):
            summary = read_summary(run_dir)
            if summary is None:
                auc_text = "pending"
                acc_text = "pending"
                f1_text = "pending"
            else:
                auc_text = f"{float(summary['auc_mean']):.4f} +/- {float(summary['auc_std']):.4f}"
                acc_text = f"{float(summary['acc_mean']):.4f} +/- {float(summary['acc_std']):.4f}"
                f1_text = f"{float(summary['f1_mean']):.4f} +/- {float(summary['f1_std']):.4f}"
            handle.write(
                f"{config_item['display_name']:<22} "
                f"{config_item['loss_text']:<26} "
                f"{auc_text:<17} {acc_text:<17} {f1_text:<17}\n"
            )


def main():
    args = parse_args()
    input_dir = os.path.abspath(os.path.expanduser(args.indir))
    output_dir = os.path.abspath(os.path.expanduser(args.outdir))
    ds_config = get_dataset_config(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    root_run_dir = os.path.join(output_dir, "results", f"experiment_{timestamp}")

    if args.dry_run:
        print(f"=== DP-SCL Loss Ablation: {ds_config['name']} ===")
        print(f"Input data directory: {os.path.join(input_dir, 'datastore')}")
        print(f"Root output: {root_run_dir}")
        print(f"Early stopping monitor: Val {args.monitor.upper()}")
        print(f"Device: {device}")
        for config_item in LOSS_CONFIGS:
            loss_mode = config_item["key"]
            lambda_con = args.lambda_con if loss_mode == "combined" else config_item["lambda_con"]
            run_dir = os.path.join(root_run_dir, loss_mode)
            print(
                f"DRY RUN | model={config_item['model_name']} "
                f"loss_mode={loss_mode} lambda={lambda_con} output={run_dir}"
            )
        return

    X, y, npz_path = load_full_temporal_data(input_dir, ds_config)
    print(f"=== DP-SCL Loss Ablation: {ds_config['name']} ===")
    print(f"Data: {npz_path} | X={X.shape} y={y.shape}")
    print(f"Root output: {root_run_dir}")
    print(f"Early stopping monitor: Val {args.monitor.upper()}")
    print(f"Device: {device}")

    run_dirs = []
    for config_item in LOSS_CONFIGS:
        run_dir = run_one_config(config_item, args, X, y, ds_config, device, root_run_dir)
        run_dirs.append(run_dir)

    report_path = os.path.join(output_dir, "result_write", f"{timestamp}.txt")
    if not args.dry_run:
        write_combined_report(report_path, run_dirs)
        print(f"\nSaved combined report: {report_path}")


if __name__ == "__main__":
    main()
