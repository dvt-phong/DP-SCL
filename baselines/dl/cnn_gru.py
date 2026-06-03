from torch import nn

from .base import CNNRecurrentBaseline


class DLCNNGRU(CNNRecurrentBaseline):
    recurrent_cls = nn.GRU

