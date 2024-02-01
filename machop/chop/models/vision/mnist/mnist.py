import torch
from torch import nn
from typing import Any, Dict


class MNISTLab4ReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class MNISTLab4LeakyReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 16)
        self.relu1 = nn.LeakyReLU(negative_slope=(2**-2))
        self.fc2 = nn.Linear(16, 10)
        self.relu2 = nn.LeakyReLU(negative_slope=(2**-2))
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def get_mnist_lab4_relu(info: Dict, pretrained: bool = False, **kwargs: Any):
    return MNISTLab4ReLU()

def get_mnist_lab4_leakyrelu(info: Dict, pretrained: bool = False, **kwargs: Any):
    return MNISTLab4LeakyReLU()
