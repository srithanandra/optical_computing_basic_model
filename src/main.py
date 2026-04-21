import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def main() -> None:
    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE: USING {device} DEVICE\n")

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    print("-------------- DATA SHAPE ----------------")
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    print("------------------------------------------\n")

    class NeuralNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.flatten(x)
            return self.linear_relu_stack(x)

    model = NeuralNetwork().to(device)
    print("---------------------- MODEL ---------------------")
    print(model)
    print("-------------------------------------------------\n")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    def evaluate() -> None:
        model.eval()
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        test_loss, correct = 0.0, 0.0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error:\n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

    epochs = 3
    for t in range(epochs):
        print(f"Epoch {t + 1}:\n-------------------------------")
        train_epoch()
        evaluate()

    model.eval()
    x, y = test_data[0]
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)
        predicted = int(logits.argmax(1).item())
        print(f"Single example prediction: predicted={predicted}, actual={int(y)}")

    print(f"\nTime spent: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()

