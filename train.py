"""
Origin and attribution:
  Project: DP-SCL.
  Purpose: Compatibility wrapper for running the DP-SCL experiment runner with
  command-line flags similar to the original CA-TFHN entry point.

Reference source:
  CA-TFHN GitHub repository by codeds27:
  https://github.com/codeds27/CA-TFHN

  Original CA-TFHN train entry point:
  https://github.com/codeds27/CA-TFHN/blob/main/train.py
"""

import sys

from train_experiment import main


if __name__ == "__main__":
    argv = [sys.argv[0]]
    args = sys.argv[1:]
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "-e":
            argv.append("--max-epochs")
            idx += 1
            argv.append(args[idx])
        elif arg == "-r":
            argv.append("--seeds")
            idx += 1
            argv.append(args[idx])
        elif arg == "-mode":
            idx += 1
            if args[idx] != "dp_scl":
                raise SystemExit("This repository now supports only -mode dp_scl.")
        elif arg == "--contrastive":
            pass
        else:
            argv.append(arg)
        idx += 1
    sys.argv = argv
    main()
