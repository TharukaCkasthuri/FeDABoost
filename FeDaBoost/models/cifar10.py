import torch
import torch.nn.functional as F
from torch import nn

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch_size, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [batch_size, 64, 8, 8]

        x = x.view(-1, 64 * 8 * 8)  
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
