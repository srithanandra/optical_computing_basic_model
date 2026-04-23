from pathlib import Path

import torch
from torch import nn


class InverseModelTandem(nn.Module):
    """
    1D CNN that maps a spectrum/response (100,) -> geometry vector (8,).

    This architecture is inferred from `data/other/inverse_model_tandem.pth`.
    """

    def __init__(self):
        super().__init__()
        # padding=2 keeps the length unchanged for kernel_size=5
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 25, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        # Accept either (N, 100) or (N, 1, 100)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))  # 100 -> 50
        x = self.pool(self.relu(self.conv2(x)))  # 50 -> 25
        x = self.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def load_inverse_model(checkpoint_path=Path('data') / 'other' / 'inverse_model_tandem.pth', *, device='cpu'):
    model = InverseModelTandem()
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

