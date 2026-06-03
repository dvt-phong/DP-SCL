import torch

from .base import BinaryClassifierHead, DLBaselineBase, TemporalConvEncoder


class DLCNN(DLBaselineBase):
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.cnn = TemporalConvEncoder(self.weekly_input_dim, self.hidden_size, self.dropout)
        self.classifier = BinaryClassifierHead(self.hidden_size, self.hidden_size, self.dropout)

    def forward(self, sub_graph):
        weekly_seq = self.reshape_weekly(sub_graph)
        conv_seq = self.cnn(weekly_seq)
        pooled = torch.mean(conv_seq, dim=1)
        return self.classifier(pooled)

