import torch
import torch.nn as nn
import torch.nn.functional as F


class ComEffFlPaperCnnModel(nn.Module):
    """The CNN model used in https://arxiv.org/abs/1602.05629 for MNIST"""

    def __init__(self):
        super(ComEffFlPaperCnnModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out
