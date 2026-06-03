"""
src.models — DP-SCL model package.

Re-exports tất cả model classes để backward compatible:
    from src.models import LGB, SiameseLGB, CLSimCLR, CLBYOL, ...
    from src.models import *

Cấu trúc:
    common.py      — Shared building blocks (Context, GraphSage, GAT, CNN, LSTM, Attention, etc.)
    lgb.py         — LGB class (Framework 0: 18 modes)
    siamese.py     — SiameseLGB + helpers (Framework 1: 6 modes)
    contrastive.py — CLSimCLR + CLBYOL + helpers (Framework 2: 8 modes)
    baseline.py    — DropoutPredictor + helpers (Baseline)
    baselines/dl/  — Standalone DL baseline methods
"""

# === Shared Building Blocks ===
from .common import (
    Context, GraphSage, GATNetwork,
    CNNFeatureExtractor, CNN2DFeatureExtractor, MBACNNFeatureExtractor,
    MyLSTM, MyBiLSTM,
    MySelfAttention, MyMHAttention, MyCrossAttention,
    LearnableQueryPool, BahdanauAttention,
    CrossAttentionFusion,
    Classifier,
    SupConLoss, ProjectionHead,
    ActionWeightedInput, EarlyPredictionMask,
    AugmentationModule,
)

# === Legacy graph-temporal baselines (LGB) ===
from .lgb import LGB

# === Framework 1: Siamese Network ===
from .siamese import (
    SiameseLGB, SiameseEncoder, SiameseProjectionHead, SiameseClassifier,
)

# === Framework 2: Contrastive Learning (SimCLR + BYOL) ===
from .contrastive import (
    CLSimCLR, CLBYOL, CLEncoder, CLProjectionHead, CLPredictorHead,
    NTXentLoss, CLClassifier,
)

# === Graph-Enhanced Wrapper (2-stream: Temporal + Graph) ===
from .graph_wrapper import GraphEnhancedWrapper

# === Baseline ===
from .baseline import (
    DropoutPredictor, WeeklyCNN, TemporalBiLSTM,
    MultiHeadAttentionPool, CrossAttentionPool, DropoutClassifier,
)

# === Standalone DL Baselines ===
from baselines.dl import (
    DLCNN, DLCNNGRU, DLCNNLSTM, DLCNNLSTMAT1, DLCNNLSTMAT2, DLCNNRNN,
    DLGRU, DLLSTM, DLLSTMMHA, DLLSTMMHALQ, MLPBaseline, build_dl_baseline,
)
