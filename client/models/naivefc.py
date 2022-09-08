'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class NaiveFC(nn.Module):
    def __init__(self, n_features, num_classes):
        super(NaiveFC, self).__init__()
        self.fc1   = nn.Linear(n_features, 128)
        #self.bn1 = nn.BatchNorm1d(128)
        self.fc2   = nn.Linear(128, 64)
        #self.bn2 = nn.BatchNorm1d(64)
        self.fc3   = nn.Linear(64, 32)
        #self.bn3 = nn.BatchNorm1d(32)
        self.fc4   = nn.Linear(32, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    def predict(self, x):
        out = self.forward(x)
        output = F.softmax(out, dim=1).cpu().detach().numpy()
        return output
