from torch import nn

from .base import CNNRecurrentBaseline


class DLCNNLSTM(CNNRecurrentBaseline):
    recurrent_cls = nn.LSTM

