from torch import nn

from .base import CNNRecurrentBaseline


class DLCNNRNN(CNNRecurrentBaseline):
    recurrent_cls = nn.RNN

