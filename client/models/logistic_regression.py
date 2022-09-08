import torch
from torch import nn


class LogisticRegression(nn.Module):
  def __init__(self, n_features, num_classes):
    super().__init__()
    self.linear = nn.Linear(n_features, num_classes, bias=False)
    self.initialize()

  def initialize(self):
    nn.init.xavier_uniform_(self.linear.weight.data)
    # self.linear.bias.data.zero_()

  def forward(self, x):
    return torch.sigmoid(self.linear(x))
