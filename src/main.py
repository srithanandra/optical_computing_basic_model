import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from src.model import MLP
except ModuleNotFoundError:
    from model import MLP


def regression_accuracy(pred, target, *, rtol=0.05, atol=1e-8):
    return torch.isclose(pred, target, rtol=rtol, atol=atol).float().mean().item() * 100.0


def plot_accuracy_history(accuracy_history, plot_path):
    if plt is None:
        print('Skipping accuracy plot because matplotlib is not installed.')
        print('Install it with: pip install matplotlib')
        return

    epochs = np.arange(1, len(accuracy_history['train']) + 1)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracy_history['train'], label='Train accuracy')
    plt.plot(epochs, accuracy_history['test'], label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (% within tolerance)')
    plt.title('MLP geometry reconstruction accuracy over time')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def main() -> None:
    start = time.time()

    data_path = Path('data') / 'geometry_compiled.npy'
    accuracy_plot_path = Path('data') / 'plot_images' / 'main_accuracy.png'
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
    accuracy_history = {
        'train': [],
        'test': [],
    }

    def train_epoch():
        model.train()
        size = len(train_dataloader.dataset)
        total_loss = 0.0
        total_accuracy = 0.0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            accuracy = regression_accuracy(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(X)
            total_accuracy += accuracy * len(X)

            if batch % 200 == 0:
                current = (batch + 1) * len(X)
                print(f'loss: {loss.item():>7f}  accuracy: {accuracy:>6.2f}%  [{current:>5d}/{size:>5d}]')

        return total_loss / size, total_accuracy / size

    def evaluate():
        model.eval()
        size = len(test_dataloader.dataset)
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                accuracy = regression_accuracy(pred, y)

                test_loss += loss.item() * len(X)
                test_accuracy += accuracy * len(X)

        test_loss /= size
        test_accuracy /= size
        print(f'Test Error:\n Avg loss: {test_loss:>8f}\n Accuracy: {test_accuracy:>6.2f}%\n')
        return test_loss, test_accuracy

    for t in range(epochs):
        print(f'Epoch {t + 1}:\n-------------------------------')
        _, train_accuracy = train_epoch()
        _, test_accuracy = evaluate()
        accuracy_history['train'].append(train_accuracy)
        accuracy_history['test'].append(test_accuracy)

    plot_accuracy_history(accuracy_history, accuracy_plot_path)
    print(f'Saved accuracy plot to {accuracy_plot_path}')

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

