import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from train_experiment import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        *sys.argv[1:],
        "--models",
        "proposed",
        "--proposed-name",
        "DP-SCL",
        "--proposed-mode",
        "dp_scl",
    ]
    main()
