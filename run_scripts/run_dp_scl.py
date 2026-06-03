import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import argparse

from ablation_runner_common import (
    add_ablation_args,
    run_siamese_mode,
    save_single_model_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run A7: full DP-SCL.")
    add_ablation_args(parser)
    args = parser.parse_args()

    model_name = "A7_DP_SCL"
    run_dir = run_siamese_mode(
        args,
        "ablation_a7_dp_scl",
        model_name,
        "siamese_lstm_attn",
        lambda_con=0.1,
        temperature=0.07,
    )
    save_single_model_report(args, run_dir, model_name, "A7_DP_SCL", "ablation_a7_dp_scl.txt")


if __name__ == "__main__":
    main()
