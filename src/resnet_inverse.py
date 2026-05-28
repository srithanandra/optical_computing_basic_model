import torch
import torch.nn as nn
from pathlib import Path

class Res1DBlock(nn.Module):
    """A single 1D Residual block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv_path(x) + self.shortcut(x))

class InverseResNet1D(nn.Module):
    """1D ResNet tailored for Spectrum [1, 500] -> Geometry [8]."""
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layer1 = Res1DBlock(32, 64, stride=2)
        self.layer2 = Res1DBlock(64, 128, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 8) # Output matches your 8 parameters (r1-4, theta1-4)

    def forward(self, x):
        if x.dim() == 2: # Convert [Batch, 500] to [Batch, 1, 500]
            x = x.unsqueeze(1)
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x).squeeze(-1)
        return self.fc(x)

def load_inverse_model(device='cpu'):
    model = InverseResNet1D()
    model.to(device)
    return model