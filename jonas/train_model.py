import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms

import xarray as xr

from codecarbon import track_emissions


def stack_channels(ds):
    channels = ["131A", "1600A", "1700A", "171A", "193A", "211A", "304A", "335A", "94A"]
    data_list = []
    for ch in channels:
        da = ds.sel(channel=ch)["DN"]  # shape: (time, x, y)
        data_list.append(da.values)  # (time, x, y)
    data_array = np.stack(data_list, axis=1)  # shape: (time, channel, x, y)
    return data_array


def train_model(model, train_loader, test_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")

        evaluate_model(model, test_loader, criterion)


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    print(f"  -> Test Loss: {total_loss / len(test_loader):.4f}")


# Dataset-class
class ChannelPredictionDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs  # shape: (N, 1, 128, 128)
        self.targets = targets  # shape: (N, 8, 128, 128)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 128, 128)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # (B, 32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),  # (B, 8, 128, 128)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main():
    data_path = "/cephfs/users/jhackfeld/dl_data/all128.nc"
    ds = xr.open_dataset(data_path)

    full_data = stack_channels(ds)  # shape: (6130, 9, 128, 128)
    del ds

    # Input: Channel 94A (index 8)
    # Output:  Channels (indices 0–7)
    inputs = full_data[:, 8, :, :]  # shape: (6130, 128, 128)
    targets = full_data[:, 0:8, :, :]  # shape: (6130, 8, 128, 128)

    # Transformations
    augmentation = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=0),
            transforms.RandomRotation(degrees=90),
            transforms.RandomRotation(degrees=180),
            transforms.RandomRotation(degrees=270),
        ]
    )
    # convert to torch
    inputs_tensor = torch.from_numpy(inputs).float()
    targets_tensor = torch.from_numpy(targets).float()
    del inputs, targets

    augmented_inputs = torch.stack(
        [augmentation(im) for im in inputs_tensor.unsqueeze(1)]
    )
    augmented_targets = torch.stack([augmentation(im) for im in targets_tensor])
    del inputs_tensor, targets_tensor

    # create dataset
    full_dataset = ChannelPredictionDataset(augmented_inputs, augmented_targets)
    del augmented_inputs, augmented_targets

    # Training/Test split (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = CNN()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader, test_loader, optimizer, criterion, epochs=5)

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
