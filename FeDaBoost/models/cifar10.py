import torch

class CIFAR10Net(torch.nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.fc1 = torch.nn.Linear(3 * 32 * 32, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x