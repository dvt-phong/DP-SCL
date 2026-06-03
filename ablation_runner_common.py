import argparse
import json
import os
import subprocess
import sys

from experiment_sensitivity_runner import (
    DEFAULT_SEEDS,
    ensure_result_write,
    f4,
    get_model_rows,
    get_model_summary,
    mean_std,
)


def add_ablation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-indir", type=str, default=".")
    parser.add_argument("-outdir", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="xuetangx", choices=["xuetangx", "oulad", "snap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--cls-layers", type=int, default=1)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="rerun even when result CSV already exists")
    parser.add_argument("--dry-run", action="store_true", help="print command without launching training")


def experiment_dir(outdir, run_name):
    return os.path.join(os.path.abspath(os.path.expanduser(outdir)), "results", f"experiment_{run_name}")


def run_train_experiment(command, run_dir, args, expected_config):
    summary_csv = os.path.join(run_dir, "summary_results.csv")
    if os.path.exists(summary_csv) and not args.force:
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
            mismatches = [
                key for key, value in expected_config.items()
                if config.get(key) != value
            ]
            if not mismatches:
                print(f"Reuse existing: {run_dir}")
                return
            print(f"Existing result config mismatch ({', '.join(mismatches)}); rerunning: {run_dir}")
        else:
            print(f"Existing result missing config.json; rerunning: {run_dir}")
    print("CMD:", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)


def build_common_command(args, run_name):
    return [
        sys.executable,
        "train_experiment.py",
        "-indir", args.indir,
        "-outdir", args.outdir,
        "--dataset", args.dataset,
        "--seeds", *[str(seed) for seed in args.seeds],
        "--split", "0.60", "0.10", "0.30",
        "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--hidden-size", str(args.hidden_size),
        "--num-layers", str(args.num_layers),
        "--num-workers", str(args.num_workers),
        "--run-name", run_name,
    ]


def run_dl_mode(args, run_name, mode):
    run_dir = experiment_dir(args.outdir, run_name)
    command = build_common_command(args, run_name)
    command[command.index("--seeds"):command.index("--split")] = [
        "--models", mode,
        "--seeds", *[str(seed) for seed in args.seeds],
    ]
    expected_config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "models": [mode],
        "seeds": args.seeds,
    }
    run_train_experiment(command, run_dir, args, expected_config)
    return run_dir


def run_siamese_mode(args, run_name, model_name, mode, lambda_con=0.1, temperature=0.07):
    run_dir = experiment_dir(args.outdir, run_name)
    command = build_common_command(args, run_name)
    command[command.index("--seeds"):command.index("--split")] = [
        "--models", "proposed",
        "--proposed-name", model_name,
        "--proposed-mode", mode,
        "--seeds", *[str(seed) for seed in args.seeds],
    ]
    command.extend([
        "--cls-layers", str(args.cls_layers),
        "--lambda-con", str(lambda_con),
        "--temperature", str(temperature),
        "--mask-ratio", str(args.mask_ratio),
        "--noise-std", str(args.noise_std),
    ])
    expected_config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "lambda_con": lambda_con,
        "temperature": temperature,
        "proposed_name": model_name,
        "proposed_mode": mode,
        "models": [model_name],
        "seeds": args.seeds,
    }
    run_train_experiment(command, run_dir, args, expected_config)
    return run_dir


def write_ablation_model_report(path, display_name, rows, summary):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"MODEL: {display_name}\n")
        for row in rows:
            handle.write(
                f"  Seed {int(row['seed']):>6} | "
                f"best_epoch={row['best_epoch']} stopped={row['stopped_epoch']} | "
                f"threshold={f4(row['threshold'])} | "
                f"AUC={f4(row['test_auc'])} F1={f4(row['test_f1'])}\n"
            )
        handle.write(
            f"  FINAL | AUC={mean_std(summary, 'auc')} | "
            f"Precision={mean_std(summary, 'precision')} | "
            f"Recall={mean_std(summary, 'recall')} | "
            f"F1={mean_std(summary, 'f1')}\n"
        )


def save_single_model_report(args, run_dir, csv_model_name, display_name, output_filename):
    if args.dry_run:
        return
    rows = get_model_rows(run_dir, csv_model_name)
    summary = get_model_summary(run_dir, csv_model_name)
    output_path = os.path.join(ensure_result_write(args.outdir), output_filename)
    write_ablation_model_report(output_path, display_name, rows, summary)
    print(f"Saved paper-ready report: {output_path}")
