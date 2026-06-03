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


LAMBDA_VALUES = [0.01, 0.05, 0.10, 0.20, 0.50]
RUN_VALUES = LAMBDA_VALUES


def value_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lambda sensitivity for DP-SCL.")
    add_common_args(parser)
    args = parser.parse_args()

    summaries = {}
    epoch_summaries = {}
    run_dirs = {}
    for lambda_con in RUN_VALUES:
        model_name = f"DP-SCL_lambda_{value_tag(lambda_con)}"
        protocol = ProtocolRun(
            run_name=f"lambda_sens_{value_tag(lambda_con)}",
            model_name=model_name,
            mode="siamese_lstm_attn",
            lambda_con=lambda_con,
            temperature=0.07,
        )
        run_dir = run_protocol(protocol, args)
        if not args.dry_run:
            run_dirs[lambda_con] = run_dir
            summaries[lambda_con] = get_model_summary(run_dir, model_name)
            epoch_summaries[lambda_con] = summarize_epoch_history(run_dir, model_name)

    if args.dry_run:
        return

    result_write_dir = ensure_result_write(args.outdir)
    output_path = os.path.join(result_write_dir, "lambda_sensitivity.txt")
    epoch_output_path = os.path.join(result_write_dir, "lambda_sensitivity_epoch_avg.txt")
    write_sensitivity_report(
        output_path,
        "LAMBDA SENSITIVITY BEST-CHECKPOINT SUMMARY (tau=0.07 fixed)",
        "lambda",
        summaries,
        LAMBDA_VALUES,
        best_value=best_value_by_metric(summaries, "auc"),
    )
    write_epoch_average_sensitivity_report(
        epoch_output_path,
        f"LAMBDA SENSITIVITY EPOCH-AVERAGE SUMMARY (tau=0.07 fixed, max_epochs={args.max_epochs})",
        "lambda",
        epoch_summaries,
        LAMBDA_VALUES,
        best_value=best_epoch_average_by_metric(epoch_summaries, "val_auc"),
    )
    history_path = os.path.join(result_write_dir, "lambda_epoch_history.csv")
    write_combined_epoch_history(history_path, run_dirs, "lambda")
    print(f"Saved paper-ready report: {output_path}")
    print(f"Saved epoch-average report: {epoch_output_path}")
    print(f"Saved per-epoch history: {history_path}")


if __name__ == "__main__":
    main()
