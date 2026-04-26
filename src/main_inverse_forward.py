import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from src.forward_model import load_forward_model
    from src.model import MLP
except ModuleNotFoundError:
    from forward_model import load_forward_model
    from model import MLP


class TandemInverseForward(nn.Module):
    """
    Tandem network: desired spectrum -> inverse MLP -> geometry -> frozen forward MLP -> spectrum.
    """

    def __init__(self, inverse_model_mlp: nn.Module, forward_model_mlp: nn.Module):
        super().__init__()
        self.inverse_model_mlp = inverse_model_mlp
        self.forward_model_mlp = forward_model_mlp

        for param in self.forward_model_mlp.parameters():
            param.requires_grad = False
        self.forward_model_mlp.eval()

    def forward(self, spectrum):
        geometry = self.inverse_model_mlp(spectrum)
        reconstructed_spectrum = self.forward_model_mlp(geometry)
        return reconstructed_spectrum, geometry


def build_spectrum_dataset(geometry, forward_model_mlp, *, batch_size, device):
    spectra = []
    loader = DataLoader(TensorDataset(geometry), batch_size=batch_size, shuffle=False)

    forward_model_mlp.eval()
    with torch.no_grad():
        for (batch_geometry,) in loader:
            batch_geometry = batch_geometry.to(device)
            spectra.append(forward_model_mlp(batch_geometry).cpu())

    return torch.cat(spectra, dim=0)


def export_tandem_data(model, dataset, export_path, *, batch_size, device):
    all_spectra = []
    all_geometry = []
    all_predicted_geometry = []
    all_reconstructed_spectra = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        for spectrum, geometry_target in loader:
            spectrum = spectrum.to(device)
            geometry_target = geometry_target.to(device)

            reconstructed_spectrum, predicted_geometry = model(spectrum)

            all_spectra.append(spectrum.cpu())
            all_geometry.append(geometry_target.cpu())
            all_predicted_geometry.append(predicted_geometry.cpu())
            all_reconstructed_spectra.append(reconstructed_spectrum.cpu())

    export_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        export_path,
        spectra=torch.cat(all_spectra).numpy(),
        geometry=torch.cat(all_geometry).numpy(),
        predicted_geometry=torch.cat(all_predicted_geometry).numpy(),
        reconstructed_spectra=torch.cat(all_reconstructed_spectra).numpy(),
    )


def main() -> None:
    start = time.time()

    data_path = Path('data') / 'geometry_compiled.npy'
    inverse_checkpoint_path = Path('data') / 'other' / 'inverse_model_mlp.pth'
    export_path = Path('data') / 'inverse_forward_export.npz'
    batch_size = 64
    epochs = 10
    lr = 1e-3
    train_fraction = 0.8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: USING {device} DEVICE\n')

    geometry_np = np.load(data_path)
    if geometry_np.ndim != 2 or geometry_np.shape[1] != 8:
        raise ValueError(f'Expected geometry .npy with shape (N, 8); got {geometry_np.shape}')

    geometry = torch.tensor(geometry_np, dtype=torch.float32)

    forward_model_mlp = load_forward_model(device=device)
    spectra = build_spectrum_dataset(
        geometry,
        forward_model_mlp,
        batch_size=batch_size,
        device=device,
    )

    dataset = TensorDataset(spectra, geometry)
    train_size = int(train_fraction * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('-------------- DATA SHAPE ----------------')
    for spectrum, geometry_target in test_dataloader:
        print(f'Shape of spectrum X [N, 100]: {spectrum.shape}')
        print(f'Shape of geometry reference [N, 8]: {geometry_target.shape} {geometry_target.dtype}')
        break
    print('------------------------------------------\n')

    inverse_model_mlp = MLP(input_dim=100, hidden_dims=(256, 128, 64), output_dim=8)
    model = TandemInverseForward(inverse_model_mlp, forward_model_mlp).to(device)
    print('---------------------- MODEL ---------------------')
    print(model)
    print('-------------------------------------------------\n')

    spectrum_loss_fn = nn.MSELoss()
    geometry_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.inverse_model_mlp.parameters(), lr=lr)

    def train_epoch() -> None:
        model.train()
        model.forward_model_mlp.eval()
        size = len(train_dataloader.dataset)
        for batch, (spectrum, geometry_target) in enumerate(train_dataloader):
            spectrum = spectrum.to(device)
            geometry_target = geometry_target.to(device)

            reconstructed_spectrum, predicted_geometry = model(spectrum)
            spectrum_loss = spectrum_loss_fn(reconstructed_spectrum, spectrum)
            geometry_loss = geometry_loss_fn(predicted_geometry, geometry_target)

            spectrum_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 20 == 0:
                current = min((batch + 1) * len(spectrum), size)
                print(
                    f'spectrum loss: {spectrum_loss.item():>7f}  '
                    f'geometry ref loss: {geometry_loss.item():>7f}  '
                    f'[{current:>5d}/{size:>5d}]'
                )

    def evaluate() -> None:
        model.eval()
        num_batches = len(test_dataloader)
        test_spectrum_loss = 0.0
        test_geometry_loss = 0.0
        with torch.no_grad():
            for spectrum, geometry_target in test_dataloader:
                spectrum = spectrum.to(device)
                geometry_target = geometry_target.to(device)

                reconstructed_spectrum, predicted_geometry = model(spectrum)
                test_spectrum_loss += spectrum_loss_fn(reconstructed_spectrum, spectrum).item()
                test_geometry_loss += geometry_loss_fn(predicted_geometry, geometry_target).item()

        test_spectrum_loss /= num_batches
        test_geometry_loss /= num_batches
        print(
            'Test Error:\n'
            f' Avg spectrum loss: {test_spectrum_loss:>8f}\n'
            f' Avg geometry ref loss: {test_geometry_loss:>8f}\n'
        )

    for t in range(epochs):
        print(f'Epoch {t + 1}:\n-------------------------------')
        train_epoch()
        evaluate()

    torch.save(model.inverse_model_mlp.state_dict(), inverse_checkpoint_path)
    print(f'Saved inverse MLP checkpoint to {inverse_checkpoint_path}')

    export_tandem_data(model, dataset, export_path, batch_size=batch_size, device=device)
    print(f'Exported tandem data to {export_path}')

    model.eval()
    spectrum, geometry_target = dataset[0]
    with torch.no_grad():
        spectrum = spectrum.to(device)
        geometry_target = geometry_target.to(device)
        reconstructed_spectrum, predicted_geometry = model(spectrum.unsqueeze(0))

        print('Single example (first 4 values):')
        print(f'  target spectrum       = {spectrum[:4].tolist()}')
        print(f'  reconstructed spectrum = {reconstructed_spectrum[0, :4].tolist()}')
        print(f'  reference geometry    = {geometry_target[:4].tolist()}')
        print(f'  predicted geometry    = {predicted_geometry[0, :4].tolist()}')

    print(f'\nTime spent: {time.time() - start:.2f} seconds')


if __name__ == '__main__':
    main()
