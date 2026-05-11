import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.model import MLP
except ModuleNotFoundError:
    from model import MLP


def main() -> None:
    start = time.time()

    data_path = Path('data') / 'geometry_compiled.npy'
    batch_size = 64
    epochs = 3
    lr = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: USING {device} DEVICE\n')

    x_np = np.load(data_path)
    if x_np.ndim != 2 or x_np.shape[1] != 8:
        raise ValueError(f'Expected .npy with shape (N, 8); got {x_np.shape}')

    # For now: use the full dataset as BOTH train and test.
    # Targets are set to X so the model learns an identity mapping (sanity-check training loop).
    X = torch.tensor(x_np, dtype=torch.float32)
    y = X.clone()
    dataset = TensorDataset(X, y)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print('-------------- DATA SHAPE ----------------')
    for X, y in test_dataloader:
        print(f'Shape of X [N, 8]: {X.shape}')
        print(f'Shape of y [N, 8]: {y.shape} {y.dtype}')
        break
    print('------------------------------------------\n')

    model = MLP().to(device)
    print('---------------------- MODEL ---------------------')
    print(model)
    print('-------------------------------------------------\n')

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_epoch() -> None:
        model.train()
        size = len(train_dataloader.dataset)
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 200 == 0:
                current = (batch + 1) * len(X)
                print(f'loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]')

    def evaluate() -> None:
        model.eval()
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches
        print(f'Test Error:\n Avg loss: {test_loss:>8f}\n')

    for t in range(epochs):
        print(f'Epoch {t + 1}:\n-------------------------------')
        train_epoch()
        evaluate()

    model.eval()
    x, y = dataset[0]
    with torch.no_grad():
        x = x.to(device)
        y_hat = model(x)
        print('Single example (first 4 values):')
        print(f'  x     = {x[:4].tolist()}')
        print(f'  y_hat = {y_hat[:4].tolist()}')

    print(f'\nTime spent: {time.time() - start:.2f} seconds')


if __name__ == '__main__':
    main()

