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
    parser = argparse.ArgumentParser(description="Run A5: Siamese+LSTM+MHA+SupCon without LQ.")
    add_ablation_args(parser)
    args = parser.parse_args()

    model_name = "A5_Siamese_LSTM_MHA_SupCon"
    run_dir = run_siamese_mode(
        args,
        "ablation_a5_siamese_lstm_mha",
        model_name,
        "siamese_lstm_mha",
        lambda_con=0.1,
        temperature=0.07,
    )
    save_single_model_report(
        args,
        run_dir,
        model_name,
        "A5_Siamese_LSTM_MHA_SupCon",
        "ablation_a5_siamese_lstm_mha.txt",
    )


if __name__ == "__main__":
    main()
