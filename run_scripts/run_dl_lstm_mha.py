import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import argparse

from ablation_runner_common import (
    add_ablation_args,
    run_dl_mode,
    save_single_model_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run A2: LSTM+MHA+mean pooling+BCE baseline.")
    add_ablation_args(parser)
    args = parser.parse_args()

    run_dir = run_dl_mode(args, "ablation_a2_dl_lstm_mha", "dl_lstm_mha")
    save_single_model_report(args, run_dir, "dl_lstm_mha", "A2_LSTM_MHA_BCE", "ablation_a2_dl_lstm_mha.txt")


if __name__ == "__main__":
    main()
