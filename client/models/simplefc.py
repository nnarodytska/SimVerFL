'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class SimpleFC(nn.Module):
    def __init__(self, n_features, num_classes):
        super(SimpleFC, self).__init__()
        self.fc1   = nn.Linear(n_features, 10)
        self.fc2   = nn.Linear(10, 10)
        self.fc3   = nn.Linear(10, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))        
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
    def predict(self, x):
        out = self.forward(x)
        output = F.softmax(out, dim=1).cpu().detach().numpy()
        return output
