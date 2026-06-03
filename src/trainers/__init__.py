"""
src.trainers — Training framework package.

    LGBTrainer          — Legacy graph-temporal baselines gốc (18 modes)
    SiameseTrainer      — Framework 1: Siamese Network (6 modes)
    ContrastiveTrainer  — Framework 2: SimCLR + BYOL (8 modes)
"""
from .lgb_trainer import LGBTrainer
from .siamese_trainer import SiameseTrainer
from .contrastive_trainer import ContrastiveTrainer
