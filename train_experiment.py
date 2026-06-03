import argparse
import csv
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from baselines.dl import DL_BASELINE_REGISTRY, build_dl_baseline
from baselines.ml import DEFAULT_ML_ORDER, ML_BASELINE_REGISTRY
from baselines.utils.metrics import get_estimator_scores
from src.dataset_config import get_dataset_config
from src.models import SupConLGB, SupConLoss
from src.mode_registry import resolve_backend_mode


SEED_LIST = [1, 11, 111, 1111, 11111]
DL_MODES = [
    "dl_cnn",
    "dl_lstm",
    "dl_gru",
    "dl_cnn_lstm",
    "dl_cnn_gru",
    "dl_cnn_rnn",
    "dl_cnn_lstm_at1",
    "dl_cnn_lstm_at2",
    "dl_lstm_mha",
    "dl_lstm_mha_lq",
]
PROPOSED_NAME = "DP-SCL"
PROPOSED_MODE = "dp_scl"
METRIC_NAMES = ["auc", "acc", "precision", "recall", "f1"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run protocol experiments: ML + DL + DP-SCL with 5 seeds."
    )
    parser.add_argument("-indir", type=str, default=".", help="input dir")
    parser.add_argument("-outdir", type=str, default=".", help="output dir")
    parser.add_argument("--dataset", type=str, default="xuetangx", choices=["xuetangx", "oulad", "snap"])
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="all, ml, dl, proposed, or explicit model names")
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
    parser.add_argument("--proposed-name", type=str, default=PROPOSED_NAME)
    parser.add_argument("--proposed-mode", type=str, default=PROPOSED_MODE)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--cls-layers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-slow", action="store_true", help="skip SVM and kNN")
    parser.add_argument("--strict-missing", action="store_true",
                        help="fail if optional packages such as xgboost are missing")
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
    X = np.concatenate([data["t_data"], data["v_data"]], axis=0).astype(np.float32)
    y = np.concatenate([data["t_label"], data["v_label"]], axis=0).astype(np.int64)
    if X.ndim != 4:
        raise ValueError(f"Expected temporal data shape (N,W,D,F), got {X.shape}")
    return X, y, npz_path


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
        candidates = np.quantile(y_score, np.linspace(0.0, 1.0, 1000))
        candidates = np.unique(candidates)

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
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


def resolve_models(requested, proposed_name=PROPOSED_NAME):
    requested = list(requested)
    selected = []
    if "all" in requested:
        selected.extend(("ml", "dl", "proposed"))
    else:
        selected.extend(requested)

    final = []
    if "ml" in selected:
        final.extend(DEFAULT_ML_ORDER)
    if "dl" in selected:
        final.extend(DL_MODES)
    if "proposed" in selected:
        final.append(proposed_name)

    explicit = [
        item for item in selected
        if item not in {"all", "ml", "dl", "proposed"}
    ]
    final.extend(explicit)

    deduped = []
    seen = set()
    for model in final:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def model_group(model_name, proposed_name=PROPOSED_NAME):
    if model_name.startswith("ml_"):
        return "ML"
    if model_name.startswith("dl_"):
        return "DL"
    if model_name == proposed_name:
        return "PROPOSED"
    raise ValueError(f"Unknown model: {model_name}")


def make_dl_param_dict(args, ds_config):
    return {
        "activity_num": ds_config["activity_num"],
        "week_count": ds_config["week_count"],
        "days_per_week": ds_config["days_per_week"],
        "sta_day": ds_config["sta_day"],
        "dl_hidden_size": args.hidden_size,
        "dl_num_layers": args.num_layers,
        "dl_dropout": 0.3,
        "dl_attn_heads": 4,
        "dl_attn_dropout": 0.1,
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


def make_loaders(X, y, train_idx, val_idx, test_idx, batch_size, num_workers):
    X_flat = X.reshape(X.shape[0], -1)
    tensors = {
        "train": (
            torch.from_numpy(X_flat[train_idx]).float(),
            torch.from_numpy(y[train_idx]).float(),
        ),
        "val": (
            torch.from_numpy(X_flat[val_idx]).float(),
            torch.from_numpy(y[val_idx]).float(),
        ),
        "test": (
            torch.from_numpy(X_flat[test_idx]).float(),
            torch.from_numpy(y[test_idx]).float(),
        ),
    }
    train_loader = DataLoader(
        TensorDataset(*tensors["train"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        TensorDataset(*tensors["val"]),
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        TensorDataset(*tensors["test"]),
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def eval_torch_model(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for seq_feat, y_batch in loader:
            seq_feat = seq_feat.to(device)
            y_batch = y_batch.to(device)
            sub_graph = {"batch_size": seq_feat.shape[0], "seq_feat": seq_feat}
            logits = model(sub_graph)
            if isinstance(logits, tuple):
                logits = logits[0]
            scores.append(torch.sigmoid(logits).detach().cpu().view(-1))
            labels.append(y_batch.detach().cpu().view(-1))
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def train_dl_model(model_name, X, y, train_idx, val_idx, test_idx, args,
                   ds_config, device, checkpoint_dir):
    set_seed(args.current_seed)
    train_loader, val_loader, test_loader = make_loaders(
        X, y, train_idx, val_idx, test_idx, args.batch_size, args.num_workers
    )
    model = build_dl_baseline(model_name, make_dl_param_dict(args, ds_config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_seed_{args.current_seed}.pt")

    best_val_auc = -np.inf
    best_val_f1 = -np.inf
    best_epoch = 0
    patience_count = 0
    stopped_epoch = args.max_epochs
    epoch_history = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for seq_feat, y_batch in train_loader:
            seq_feat = seq_feat.to(device)
            y_batch = y_batch.to(device).view(-1, 1)
            optimizer.zero_grad()
            logits = model({"batch_size": seq_feat.shape[0], "seq_feat": seq_feat})
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            batch_count = int(y_batch.size(0))
            train_loss_sum += float(loss.detach().cpu()) * batch_count
            train_sample_count += batch_count

        val_y, val_score = eval_torch_model(model, val_loader, device)
        val_threshold = select_threshold_by_f1(val_y, val_score)
        val_metrics = compute_metrics_with_threshold(val_y, val_score, val_threshold)
        val_auc = val_metrics["auc"]
        val_f1 = val_metrics["f1"]

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

        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_sample_count, 1),
            "train_bce_loss": train_loss_sum / max(train_sample_count, 1),
            "train_supcon_loss": "",
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
        "epoch_history": epoch_history,
        "status": "ok",
    }


def train_dp_scl(model_name, X, y, train_idx, val_idx, test_idx, args, ds_config, device, checkpoint_dir):
    set_seed(args.current_seed)
    train_loader, val_loader, test_loader = make_loaders(
        X, y, train_idx, val_idx, test_idx, args.batch_size, args.num_workers
    )
    backend_mode = resolve_backend_mode(args.proposed_mode)
    model = SupConLGB(mode=backend_mode, param_dict=make_dp_scl_param_dict(args, ds_config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    supcon = SupConLoss(temperature=args.temperature).to(device)
    safe_model_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)
    checkpoint_path = os.path.join(checkpoint_dir, f"{safe_model_name}_seed_{args.current_seed}.pt")

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
            features = torch.stack([z1, z2], dim=1)
            bce_loss = bce(logits, y_batch)
            if args.lambda_con == 0.0:
                supcon_loss = torch.zeros((), device=device)
                loss = bce_loss
            else:
                supcon_loss = supcon(features, y_batch.view(-1))
                loss = bce_loss + args.lambda_con * supcon_loss
            loss.backward()
            optimizer.step()
            batch_count = int(y_batch.size(0))
            train_loss_sum += float(loss.detach().cpu()) * batch_count
            train_bce_sum += float(bce_loss.detach().cpu()) * batch_count
            train_supcon_sum += float(supcon_loss.detach().cpu()) * batch_count
            train_sample_count += batch_count

        val_y, val_score = eval_torch_model(model, val_loader, device)
        val_threshold = select_threshold_by_f1(val_y, val_score)
        val_metrics = compute_metrics_with_threshold(val_y, val_score, val_threshold)
        val_auc = val_metrics["auc"]
        val_f1 = val_metrics["f1"]

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
        "epoch_history": epoch_history,
        "status": "ok",
    }


def run_ml_model(model_name, X, y, train_idx, val_idx, test_idx, seed, strict_missing):
    X_flat = X.reshape(X.shape[0], -1)
    try:
        model = ML_BASELINE_REGISTRY[model_name](seed)
        model.fit(X_flat[train_idx], y[train_idx])
        val_score = get_estimator_scores(model, X_flat[val_idx])
        threshold = select_threshold_by_f1(y[val_idx], val_score)
        val_metrics = compute_metrics_with_threshold(y[val_idx], val_score, threshold)
        test_score = get_estimator_scores(model, X_flat[test_idx])
        test_metrics = compute_metrics_with_threshold(y[test_idx], test_score, threshold)
        return {
            **test_metrics,
            "threshold": threshold,
            "best_epoch": "",
            "stopped_epoch": "",
            "best_val_auc": val_metrics["auc"],
            "status": "ok",
        }
    except ModuleNotFoundError:
        if strict_missing:
            raise
        return {
            "auc": np.nan,
            "acc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "threshold": np.nan,
            "best_epoch": "",
            "stopped_epoch": "",
            "best_val_auc": np.nan,
            "status": "skipped_missing_dependency",
        }


def summarize(rows):
    summary_rows = []
    by_model = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        by_model.setdefault(row["model"], []).append(row)

    for model, model_rows in by_model.items():
        out = {"group": model_rows[0]["group"], "model": model}
        for metric in METRIC_NAMES:
            values = np.array([float(r[f"test_{metric}"]) for r in model_rows], dtype=float)
            out[f"{metric}_mean"] = float(np.nanmean(values))
            out[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        best_epochs = [r["best_epoch"] for r in model_rows if r["best_epoch"] != ""]
        stopped_epochs = [r["stopped_epoch"] for r in model_rows if r["stopped_epoch"] != ""]
        out["avg_best_epoch"] = float(np.mean([int(v) for v in best_epochs])) if best_epochs else ""
        out["avg_stopped_epoch"] = float(np.mean([int(v) for v in stopped_epochs])) if stopped_epochs else ""
        summary_rows.append(out)
    return summary_rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value):
    if value == "":
        return ""
    value = float(value)
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def fmt_mean_std(row, metric):
    return f"{fmt(row[f'{metric}_mean'])} ± {fmt(row[f'{metric}_std'])}"


def write_report(path, config, rows, summary_rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("EXPERIMENT RESULTS\n")
        handle.write(f"Dataset: {config['dataset']}\n")
        handle.write("Split: 60/10/30 stratified\n")
        handle.write(f"Seeds: {config['seeds']}\n")
        handle.write(f"Max epochs: {config['max_epochs']}\n")
        handle.write(f"Early stopping: Val AUC, patience={config['patience']}\n")
        handle.write(f"Proposed alias: {config['proposed_name']} -> {config['proposed_mode']}\n\n")

        for group in ["ML", "DL", "PROPOSED"]:
            group_rows = [row for row in rows if row["group"] == group]
            if not group_rows:
                continue
            handle.write("=" * 56 + "\n")
            handle.write(f"GROUP: {group}\n")
            handle.write("=" * 56 + "\n")
            for model in dict.fromkeys(row["model"] for row in group_rows):
                handle.write(f"\nMODEL: {model}\n")
                handle.write("-" * 56 + "\n")
                for row in [r for r in group_rows if r["model"] == model]:
                    if row["status"] != "ok":
                        handle.write(f"Seed {row['seed']:>6} | status={row['status']}\n")
                        continue
                    epoch_text = ""
                    if row["best_epoch"] != "":
                        epoch_text = f"best_epoch={row['best_epoch']} stopped={row['stopped_epoch']} "
                    handle.write(
                        f"Seed {row['seed']:>6} | {epoch_text}"
                        f"threshold={fmt(row['threshold'])} | "
                        f"AUC={fmt(row['test_auc'])} "
                        f"ACC={fmt(row['test_acc'])} "
                        f"Precision={fmt(row['test_precision'])} "
                        f"Recall={fmt(row['test_recall'])} "
                        f"F1={fmt(row['test_f1'])}\n"
                    )
                summary = next((s for s in summary_rows if s["model"] == model), None)
                if summary:
                    handle.write("-" * 56 + "\n")
                    handle.write(
                        "FINAL | "
                        f"AUC={fmt_mean_std(summary, 'auc')} | "
                        f"ACC={fmt_mean_std(summary, 'acc')} | "
                        f"Precision={fmt_mean_std(summary, 'precision')} | "
                        f"Recall={fmt_mean_std(summary, 'recall')} | "
                        f"F1={fmt_mean_std(summary, 'f1')}\n"
                    )

        handle.write("\nSUMMARY TABLE\n")
        handle.write("Model, AUC, ACC, Precision, Recall, F1\n")
        for row in summary_rows:
            handle.write(
                f"{row['model']}, "
                f"{fmt_mean_std(row, 'auc')}, "
                f"{fmt_mean_std(row, 'acc')}, "
                f"{fmt_mean_std(row, 'precision')}, "
                f"{fmt_mean_std(row, 'recall')}, "
                f"{fmt_mean_std(row, 'f1')}\n"
            )


def main():
    args = parse_args()
    if args.proposed_mode == "supcon_lstm_attn_lambda0":
        args.lambda_con = 0.0
    elif args.proposed_mode in {"dp_scl", "tsn_supcon"}:
        args.lambda_con = 0.1
        args.temperature = 0.07
    input_dir = os.path.abspath(os.path.expanduser(args.indir))
    output_dir = os.path.abspath(os.path.expanduser(args.outdir))
    ds_config = get_dataset_config(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_models = resolve_models(args.models, args.proposed_name)
    if args.skip_slow:
        selected_models = [m for m in selected_models if m not in {"ml_svm", "ml_knn"}]

    timestamp = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, "results", f"experiment_{timestamp}")
    split_dir = os.path.join(run_dir, "splits")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    X, y, npz_path = load_full_temporal_data(input_dir, ds_config)
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
        "proposed_name": args.proposed_name,
        "proposed_mode": args.proposed_mode,
        "proposed_alias": {args.proposed_name: args.proposed_mode},
        "models": selected_models,
        "device": str(device),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print(f"=== Protocol Experiment: {ds_config['name']} ===")
    print(f"Data: {npz_path} | X={X.shape} y={y.shape}")
    print(f"Models: {selected_models}")
    print(f"Output: {run_dir}")
    print(f"Device: {device}")

    rows = []
    epoch_history_rows = []
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

        for model_name in selected_models:
            group = model_group(model_name, args.proposed_name)
            print(f"  [{group}] {model_name} ...", flush=True)
            start = time.time()
            args.current_seed = seed
            try:
                if group == "ML":
                    result = run_ml_model(
                        model_name, X, y, train_idx, val_idx, test_idx,
                        seed, args.strict_missing
                    )
                elif group == "DL":
                    result = train_dl_model(
                        model_name, X, y, train_idx, val_idx, test_idx,
                        args, ds_config, device, checkpoint_dir
                    )
                else:
                    result = train_dp_scl(
                        model_name, X, y, train_idx, val_idx, test_idx,
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
                "group": group,
                "model": model_name,
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
                    "group": group,
                    "model": model_name,
                    "seed": seed,
                    "lambda_con": args.lambda_con if group == "PROPOSED" else "",
                    "temperature": args.temperature if group == "PROPOSED" else "",
                    **history_row,
                })
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
                    "group", "model", "seed", "best_epoch", "stopped_epoch",
                    "best_val_auc", "threshold", "test_auc", "test_acc", "test_precision",
                    "test_recall", "test_f1", "elapsed_sec", "status",
                ],
            )
            if epoch_history_rows:
                write_csv(
                    os.path.join(run_dir, "epoch_history.csv"),
                    epoch_history_rows,
                    [
                        "group", "model", "seed", "lambda_con", "temperature",
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

    print(f"\nSaved:")
    print(f"  {os.path.join(run_dir, 'per_seed_results.csv')}")
    if epoch_history_rows:
        print(f"  {os.path.join(run_dir, 'epoch_history.csv')}")
    print(f"  {os.path.join(run_dir, 'summary_results.csv')}")
    print(f"  {os.path.join(run_dir, 'report.txt')}")


if __name__ == "__main__":
    main()
