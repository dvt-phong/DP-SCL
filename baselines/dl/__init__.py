from .cnn import DLCNN
from .cnn_gru import DLCNNGRU
from .cnn_lstm import DLCNNLSTM
from .cnn_lstm_at1 import DLCNNLSTMAT1
from .cnn_lstm_at2 import DLCNNLSTMAT2
from .cnn_rnn import DLCNNRNN
from .gru import DLGRU
from .lstm import DLLSTM
from .lstm_mha import DLLSTMMHA
from .lstm_mha_lq import DLLSTMMHALQ
from .mlp import MLPBaseline


DL_BASELINE_REGISTRY = {
    "dl_cnn": DLCNN,
    "dl_lstm": DLLSTM,
    "dl_gru": DLGRU,
    "dl_cnn_lstm": DLCNNLSTM,
    "dl_cnn_gru": DLCNNGRU,
    "dl_cnn_rnn": DLCNNRNN,
    "dl_cnn_lstm_at1": DLCNNLSTMAT1,
    "dl_cnn_lstm_at2": DLCNNLSTMAT2,
    "dl_lstm_mha": DLLSTMMHA,
    "dl_lstm_mha_lq": DLLSTMMHALQ,
    "dl_mlp_flat": MLPBaseline,
}


def build_dl_baseline(mode, param_dict):
    if mode not in DL_BASELINE_REGISTRY:
        raise ValueError(f"Unknown DL baseline mode: {mode}")
    return DL_BASELINE_REGISTRY[mode](param_dict)


__all__ = [
    "DLCNN",
    "DLCNNGRU",
    "DLCNNLSTM",
    "DLCNNLSTMAT1",
    "DLCNNLSTMAT2",
    "DLCNNRNN",
    "DLGRU",
    "DLLSTM",
    "DLLSTMMHA",
    "DLLSTMMHALQ",
    "MLPBaseline",
    "DL_BASELINE_REGISTRY",
    "build_dl_baseline",
]
