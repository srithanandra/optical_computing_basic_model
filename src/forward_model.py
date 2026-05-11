from pathlib import Path

import torch
from torch import nn


class ForwardModelMLP(nn.Module):
    """
    MLP that maps geometry vectors (8,) -> spectrum/response (100,).

    This architecture is inferred from `data/other/forward_model_mlp.pth`.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
        )

    def forward(self, x):
        return self.net(x)


def load_forward_model(checkpoint_path=Path('data') / 'other' / 'forward_model_mlp.pth', *, device='cpu'):
    model = ForwardModelMLP()
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

