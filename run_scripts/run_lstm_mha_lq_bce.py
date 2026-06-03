import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run LSTM+MHA+LQ+BCE ablation, 5 seeds.")
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="rerun even when result CSV already exists")
    parser.add_argument("--dry-run", action="store_true", help="print command without launching training")
    return parser.parse_args()


def write_report(path, model_name, rows, summary):
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


def main():
    args = parse_args()
    model_name = "LSTM_MHA_LQ_BCE"
    run_name = "lstm_mha_lq_bce_hidden128"
    run_dir = os.path.join(os.path.abspath(os.path.expanduser(args.outdir)), "results", f"experiment_{run_name}")
    summary_csv = os.path.join(run_dir, "summary_results.csv")

    if os.path.exists(summary_csv) and not args.force:
        print(f"Reuse existing: {run_dir}")
    else:
        command = [
            sys.executable,
            "train_experiment.py",
            "-indir", args.indir,
            "-outdir", args.outdir,
            "--dataset", args.dataset,
            "--models", "dl_lstm_mha_lq",
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
        print("CMD:", " ".join(command))
        if args.dry_run:
            return
        subprocess.run(command, check=True)

    rows = get_model_rows(run_dir, "dl_lstm_mha_lq")
    summary = get_model_summary(run_dir, "dl_lstm_mha_lq")
    output_path = os.path.join(ensure_result_write(args.outdir), "lstm_mha_lq_bce_hidden128.txt")
    write_report(output_path, model_name, rows, summary)
    print(f"Saved paper-ready report: {output_path}")


if __name__ == "__main__":
    main()
