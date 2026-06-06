"""
Origin and attribution:
  Project: DP-SCL.
  Purpose: Convenience launcher that runs train_experiment.py from the project
  root.

Reference source:
  This launcher is project utility code for DP-SCL and is not copied from an
  external project.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from train_experiment import main


if __name__ == "__main__":
    main()
