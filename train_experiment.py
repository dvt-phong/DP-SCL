import argparse
import csv
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.dataset_config import get_dataset_config
from src.models import DPSCLModel, SupConLoss
from src.mode_registry import DP_SCL_MODE, resolve_backend_mode


SEED_LIST = [1, 11, 111, 1111, 11111]
MODEL_NAME = "DP-SCL"
METRIC_NAMES = ["auc", "acc", "precision", "recall", "f1"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run DP-SCL experiments with one or more seeds.")
    parser.add_argument("-indir", type=str, default=".", help="input directory")
    parser.add_argument("-outdir", type=str, default=".", help="output directory")
    parser.add_argument("--dataset", type=str, default="xuetangx", choices=["xuetangx", "oulad", "snap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEED_LIST)
    parser.add_argument("--split", nargs=3, type=float, default=[0.60, 0.10, 0.30], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lambda-con", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--cls-layers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_full_temporal_data(input_dir, ds_config):
    npz_path = os.path.join(input_dir, "datastore", ds_config["npz_filename"])
    data = np.load(npz_path)
    x = np.concatenate([data["t_data"], data["v_data"]], axis=0).astype(np.float32)
    y = np.concatenate([data["t_label"], data["v_label"]], axis=0).astype(np.int64)
    expected_suffix = (ds_config["week_count"], ds_config["days_per_week"], ds_config["activity_num"])
    if x.ndim != 4 or x.shape[1:] != expected_suffix:
        raise ValueError(f"Expected data shape (N,{expected_suffix[0]},{expected_suffix[1]},{expected_suffix[2]}), got {x.shape}")
    return x, y, npz_path


def make_split_indices(y, seed, split):
    train_ratio, val_ratio, test_ratio = split
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {split}")

    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=y,
        random_state=seed,
    )
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        stratify=y[train_val_idx],
        random_state=seed,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def save_split_indices(split_dir, seed, train_idx, val_idx, test_idx):
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, f"seed_{seed}_train.npy"), train_idx)
    np.save(os.path.join(split_dir, f"seed_{seed}_val.npy"), val_idx)
    np.save(os.path.join(split_dir, f"seed_{seed}_test.npy"), test_idx)


def class_ratio(y, idx):
    labels = y[idx]
    pos = int(labels.sum())
    total = len(labels)
    return {"total": total, "pos": pos, "neg": total - pos, "pos_ratio": pos / max(total, 1)}


def select_threshold_by_f1(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    candidates = np.unique(y_score)
    if len(candidates) == 0:
        return 0.5
    if len(candidates) > 1000:
        candidates = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 1000)))

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        score = f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics_with_threshold(y_true, y_score, threshold):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    y_pred = (y_score >= threshold).astype(int)
    auc = float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_score))
    return {
        "auc": auc,
        "acc": float(np.mean(y_pred == y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def make_dp_scl_param_dict(args, ds_config):
    return {
        "activity_num": ds_config["activity_num"],
        "sta_day": ds_config["sta_day"],
        "week_count": ds_config["week_count"],
        "select_count": ds_config["week_count"],
        "cnn_in_channels": ds_config["days_per_week"],
        "supcon_hidden_size": args.hidden_size,
        "supcon_proj_dim": args.hidden_size,
        "supcon_temperature": args.temperature,
        "supcon_mask_ratio": args.mask_ratio,
        "supcon_noise_std": args.noise_std,
        "supcon_attn_heads": 4,
        "supcon_cls_dropout": 0.3,
        "supcon_num_layers": args.num_layers,
        "supcon_cls_hidden_layers": args.cls_layers,
        "use_action_weight": False,
        "use_early_prediction": False,
        "early_min_weeks": 2,
    }


def make_loaders(x, y, train_idx, val_idx, test_idx, batch_size, num_workers):
    x_flat = x.reshape(x.shape[0], -1)
    tensors = {
        "train": (torch.from_numpy(x_flat[train_idx]).float(), torch.from_numpy(y[train_idx]).float()),
        "val": (torch.from_numpy(x_flat[val_idx]).float(), torch.from_numpy(y[val_idx]).float()),
        "test": (torch.from_numpy(x_flat[test_idx]).float(), torch.from_numpy(y[test_idx]).float()),
    }
    train_loader = DataLoader(TensorDataset(*tensors["train"]), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(TensorDataset(*tensors["val"]), batch_size=max(1, batch_size // 2), shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(TensorDataset(*tensors["test"]), batch_size=max(1, batch_size // 2), shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def eval_dp_scl(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for seq_feat, y_batch in loader:
            seq_feat = seq_feat.to(device)
            logits = model({"batch_size": seq_feat.shape[0], "seq_feat": seq_feat})
            scores.append(torch.sigmoid(logits).detach().cpu().view(-1))
            labels.append(y_batch.detach().cpu().view(-1))
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def train_dp_scl(x, y, train_idx, val_idx, test_idx, args, ds_config, device, checkpoint_dir):
    set_seed(args.current_seed)
    train_loader, val_loader, test_loader = make_loaders(
        x, y, train_idx, val_idx, test_idx, args.batch_size, args.num_workers
    )
    backend_mode = resolve_backend_mode(DP_SCL_MODE)
    model = DPSCLModel(mode=backend_mode, param_dict=make_dp_scl_param_dict(args, ds_config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    supcon = SupConLoss(temperature=args.temperature).to(device)
    checkpoint_path = os.path.join(checkpoint_dir, f"dp_scl_seed_{args.current_seed}.pt")

    best_val_auc = -np.inf
    best_val_f1 = -np.inf
    best_epoch = 0
    patience_count = 0
    stopped_epoch = args.max_epochs
    epoch_history = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_bce_sum = 0.0
        train_supcon_sum = 0.0
        train_sample_count = 0

        for seq_feat, y_batch in train_loader:
            seq_feat = seq_feat.to(device)
            y_batch = y_batch.to(device).view(-1, 1)
            optimizer.zero_grad()
            logits, z1, z2 = model({"batch_size": seq_feat.shape[0], "seq_feat": seq_feat})
            bce_loss = bce(logits, y_batch)
            supcon_loss = supcon(torch.stack([z1, z2], dim=1), y_batch.view(-1))
            loss = bce_loss + args.lambda_con * supcon_loss
            loss.backward()
            optimizer.step()

            batch_count = int(y_batch.size(0))
            train_loss_sum += float(loss.detach().cpu()) * batch_count
            train_bce_sum += float(bce_loss.detach().cpu()) * batch_count
            train_supcon_sum += float(supcon_loss.detach().cpu()) * batch_count
            train_sample_count += batch_count

        val_y, val_score = eval_dp_scl(model, val_loader, device)
        val_threshold = select_threshold_by_f1(val_y, val_score)
        val_metrics = compute_metrics_with_threshold(val_y, val_score, val_threshold)
        val_auc = val_metrics["auc"]
        val_f1 = val_metrics["f1"]

        improved = val_auc > best_val_auc + 1e-6 or (
            abs(val_auc - best_val_auc) <= 1e-6 and val_f1 > best_val_f1
        )
        if improved:
            best_val_auc = val_auc
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1

        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_sample_count, 1),
            "train_bce_loss": train_bce_sum / max(train_sample_count, 1),
            "train_supcon_loss": train_supcon_sum / max(train_sample_count, 1),
            "val_threshold": val_threshold,
            "val_auc": val_metrics["auc"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "best_val_auc_so_far": best_val_auc,
            "best_val_f1_so_far": best_val_f1,
            "best_epoch_so_far": best_epoch,
            "patience_count": patience_count,
            "is_best": int(best_epoch == epoch),
        })

        if patience_count >= args.patience:
            stopped_epoch = epoch
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    val_y, val_score = eval_dp_scl(model, val_loader, device)
    threshold = select_threshold_by_f1(val_y, val_score)
    test_y, test_score = eval_dp_scl(model, test_loader, device)
    test_metrics = compute_metrics_with_threshold(test_y, test_score, threshold)
    return {
        **test_metrics,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_auc": best_val_auc,
        "epoch_history": epoch_history,
        "status": "ok",
    }


def summarize(rows):
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return []
    out = {"model": MODEL_NAME}
    for metric in METRIC_NAMES:
        values = np.array([float(row[f"test_{metric}"]) for row in ok_rows], dtype=float)
        out[f"{metric}_mean"] = float(np.nanmean(values))
        out[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    out["avg_best_epoch"] = float(np.mean([int(row["best_epoch"]) for row in ok_rows]))
    out["avg_stopped_epoch"] = float(np.mean([int(row["stopped_epoch"]) for row in ok_rows]))
    return [out]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value == "":
        return ""
    value = float(value)
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def fmt_mean_std(row, metric):
    return f"{fmt(row[f'{metric}_mean'])} +/- {fmt(row[f'{metric}_std'])}"


def write_report(path, config, rows, summary_rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("DP-SCL EXPERIMENT RESULTS\n")
        handle.write(f"Dataset: {config['dataset']} ({config['dataset_name']})\n")
        handle.write(f"Split: {config['split']}\n")
        handle.write(f"Seeds: {config['seeds']}\n")
        handle.write(f"Max epochs: {config['max_epochs']}\n")
        handle.write(f"Early stopping: Val AUC, patience={config['patience']}\n")
        handle.write(f"lambda_con: {config['lambda_con']}\n")
        handle.write(f"temperature: {config['temperature']}\n\n")

        for row in rows:
            if row["status"] != "ok":
                handle.write(f"Seed {row['seed']:>6} | status={row['status']}\n")
                continue
            handle.write(
                f"Seed {row['seed']:>6} | best_epoch={row['best_epoch']} "
                f"stopped={row['stopped_epoch']} threshold={fmt(row['threshold'])} | "
                f"AUC={fmt(row['test_auc'])} ACC={fmt(row['test_acc'])} "
                f"Precision={fmt(row['test_precision'])} Recall={fmt(row['test_recall'])} "
                f"F1={fmt(row['test_f1'])}\n"
            )

        handle.write("\nSUMMARY TABLE\n")
        handle.write("Model, AUC, ACC, Precision, Recall, F1\n")
        for row in summary_rows:
            handle.write(
                f"{row['model']}, {fmt_mean_std(row, 'auc')}, "
                f"{fmt_mean_std(row, 'acc')}, {fmt_mean_std(row, 'precision')}, "
                f"{fmt_mean_std(row, 'recall')}, {fmt_mean_std(row, 'f1')}\n"
            )


def main():
    args = parse_args()
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
    config = {
        "dataset": args.dataset,
        "dataset_name": ds_config["name"],
        "npz_path": npz_path,
        "samples": int(len(y)),
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
        "model": MODEL_NAME,
        "mode": DP_SCL_MODE,
        "device": str(device),
    }
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


if __name__ == "__main__":
    main()
