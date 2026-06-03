"""
src.trainers — Training framework package.

    LGBTrainer          — Legacy graph-temporal baselines gốc (18 modes)
    SupConTrainer      — Framework 1: SupCon Network (6 modes)
    ContrastiveTrainer  — Framework 2: SimCLR + BYOL (8 modes)
"""
from .lgb_trainer import LGBTrainer
from .sup_con_trainer import SupConTrainer
from .contrastive_trainer import ContrastiveTrainer
