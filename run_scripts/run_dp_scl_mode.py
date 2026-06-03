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
    parser = argparse.ArgumentParser(
        description="Run DP-SCL standalone mode with lambda=0.1, tau=0.07, 5 seeds."
    )
    add_ablation_args(parser)
    args = parser.parse_args()

    model_name = "DP-SCL"
    run_dir = run_siamese_mode(
        args,
        "dp_scl_mode",
        model_name,
        "dp_scl",
        lambda_con=0.1,
        temperature=0.07,
    )
    save_single_model_report(
        args,
        run_dir,
        model_name,
        "DP-SCL",
        "dp_scl_mode.txt",
    )


if __name__ == "__main__":
    main()
