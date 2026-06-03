import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import argparse
import os

from experiment_sensitivity_runner import (
    ProtocolRun,
    add_common_args,
    best_epoch_average_by_metric,
    best_value_by_metric,
    ensure_result_write,
    get_model_summary,
    run_protocol,
    summarize_epoch_history,
    write_combined_epoch_history,
    write_epoch_average_sensitivity_report,
    write_sensitivity_report,
)


TAU_VALUES = [0.03, 0.05, 0.07, 0.10, 0.20]
RUN_VALUES = TAU_VALUES


def value_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tau sensitivity for DP-SCL.")
    add_common_args(parser)
    args = parser.parse_args()

    summaries = {}
    epoch_summaries = {}
    run_dirs = {}
    for tau in RUN_VALUES:
        model_name = f"DP-SCL_tau_{value_tag(tau)}"
        protocol = ProtocolRun(
            run_name=f"tau_sens_{value_tag(tau)}",
            model_name=model_name,
            mode="siamese_lstm_attn",
            lambda_con=0.1,
            temperature=tau,
        )
        run_dir = run_protocol(protocol, args)
        if not args.dry_run:
            run_dirs[tau] = run_dir
            summaries[tau] = get_model_summary(run_dir, model_name)
            epoch_summaries[tau] = summarize_epoch_history(run_dir, model_name)

    if args.dry_run:
        return

    result_write_dir = ensure_result_write(args.outdir)
    output_path = os.path.join(result_write_dir, "tau_sensitivity.txt")
    epoch_output_path = os.path.join(result_write_dir, "tau_sensitivity_epoch_avg.txt")
    write_sensitivity_report(
        output_path,
        "TAU SENSITIVITY BEST-CHECKPOINT SUMMARY (lambda=0.1 fixed)",
        "tau",
        summaries,
        TAU_VALUES,
        best_value=best_value_by_metric(summaries, "auc"),
    )
    write_epoch_average_sensitivity_report(
        epoch_output_path,
        f"TAU SENSITIVITY EPOCH-AVERAGE SUMMARY (lambda=0.1 fixed, max_epochs={args.max_epochs})",
        "tau",
        epoch_summaries,
        TAU_VALUES,
        best_value=best_epoch_average_by_metric(epoch_summaries, "val_auc"),
    )
    history_path = os.path.join(result_write_dir, "tau_epoch_history.csv")
    write_combined_epoch_history(history_path, run_dirs, "tau")
    print(f"Saved paper-ready report: {output_path}")
    print(f"Saved epoch-average report: {epoch_output_path}")
    print(f"Saved per-epoch history: {history_path}")


if __name__ == "__main__":
    main()
