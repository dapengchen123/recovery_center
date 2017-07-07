from __future__ import absolute_import
import torch
from torch import nn

class LSTMCC(nn.Module):
    def __init__(self, num_features, hiddensize, LSTMlayer=1, drop=0):
        super(LSTMCC, self).__init__()
        self.inputsize = num_features
        self.hiddensize = hiddensize
        self.lstm = nn.LSTM(self.inputsize, self.hiddensize, num_layers=LSTMlayer, batch_first=True, dropout=drop)

    def forward(self, input):
        x, _ = self.lstm(input)
        x = torch.squeeze(torch.mean(x, 1))
        x = x / x.norm(2, 1).expand_as(x)
        return x



