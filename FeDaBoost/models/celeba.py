import torch
from torch import nn
import torch.nn.functional as F

class CELEBANet(nn.Module):
    def __init__(self, image_size=64):
        super(CELEBANet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(8 * (image_size // 2) * (image_size // 2), 1)
        self.track_layers = {"conv1": self.conv1, "fc1": self.fc1}

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  
        x = self.dropout(x)
        x = self.fc1(x)
        return x

    def process_x(self, raw_x_batch):
        """
        Converts a batch of raw images to a torch.Tensor.
        Expects raw_x_batch to be a numpy array or list of images in NHWC format.
        """
        if not isinstance(raw_x_batch, torch.Tensor):
            x = torch.tensor(raw_x_batch, dtype=torch.float32)
        else:
            x = raw_x_batch.clone().detach()
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        return x

    def process_y(self, raw_y_batch):
        """
        Converts a batch of binary labels to a torch.Tensor.
        For binary classification, BCEWithLogitsLoss expects targets as floats
        of shape (N, 1) with values 0 or 1.
        """
        return torch.tensor(raw_y_batch, dtype=torch.float32).view(-1, 1)
