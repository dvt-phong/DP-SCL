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
    ensure_result_write,
    get_model_rows,
    get_model_summary,
    run_protocol,
    write_ablation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation: Siamese + LSTM, 5 seeds.")
    add_common_args(parser)
    args = parser.parse_args()

    model_name = "Siamese_LSTM"
    protocol = ProtocolRun(
        run_name="ablation_siamese_lstm",
        model_name=model_name,
        mode="siamese_lstm",
        lambda_con=0.1,
        temperature=0.07,
    )
    run_dir = run_protocol(protocol, args)
    if args.dry_run:
        return

    rows = get_model_rows(run_dir, model_name)
    summary = get_model_summary(run_dir, model_name)
    output_path = os.path.join(ensure_result_write(args.outdir), "ablation_siamese_lstm.txt")
    write_ablation_report(output_path, model_name, rows, summary)
    print(f"Saved paper-ready report: {output_path}")


if __name__ == "__main__":
    main()
