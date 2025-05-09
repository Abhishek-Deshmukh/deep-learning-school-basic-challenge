import numpy as np
import pickle

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


def normalize(data, mean=None, std=None):
    if data.ndim == 3:
        if mean is None:
            mean = data.mean()
        if std is None:
            std = data.std()
        normed = (data - mean) / std

    elif data.ndim == 4:
        # channelwise
        if mean is None:
            mean = data.mean(axis=(0, 2, 3), keepdims=True)
        if std is None:
            std = data.std(axis=(0, 2, 3), keepdims=True)
        normed = (data - mean) / std

    else:
        raise ValueError("Unsupported data shape. Expected 3D or 4D tensor.")

    return normed, mean, std


def denormalize(data, mean, std):
    return data * std + mean


def save_mean_std(mean, std, file_path):
    with open(file_path, "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)


def train_model(
    model, train_loader, test_loader, optimizer, criterion, device, epochs=10
):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")

        evaluate_model(model, test_loader, criterion, device)


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Residual connection
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # (B, 64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.resblock3 = ResidualBlock(64)

        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # (B, 32, 128, 128)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),  # (B, 8, 128, 128)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.output_conv(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Encoder-Teil
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, padding=1
            ),  # (B, 128, 128, 128) (zusätzlicher Layer)
            nn.ReLU(),
        )

        # Decoder-Teil
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # (B, 64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # (B, 32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),  # (B, 8, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(
                8, 8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (B, 8, 128, 128) (Upsampling Layer)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNN_3(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super(CNN_3, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)  # 128 -> 64

        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 64 -> 32

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 32 -> 64
        self.dec1 = self.conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 64 -> 128
        self.dec2 = self.conv_block(64, 32)

        # Output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # (B, 32, 128, 128)
        p1 = self.pool1(e1)  # (B, 32, 64, 64)

        e2 = self.enc2(p1)  # (B, 64, 64, 64)
        p2 = self.pool2(e2)  # (B, 64, 32, 32)

        # Bottleneck
        b = self.bottleneck(p2)  # (B, 128, 32, 32)

        # Decoder
        u1 = self.up1(b)  # (B, 64, 64, 64)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))  # Skip conn.

        u2 = self.up2(d1)  # (B, 32, 128, 128)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))  # Skip conn.

        out = self.out_conv(d2)  # (B, 8, 128, 128)
        return out


class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
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


@track_emissions
def main():
    data_path = "/cephfs/users/jhackfeld/dl_data/all128.nc"
    ds = xr.open_dataset(data_path)

    full_data = stack_channels(ds)  # shape: (6130, 9, 128, 128)
    del ds

    # Input: Channel 94A (index 8)
    # Output:  Channels (indices 0–7)
    inputs = full_data[:, 8, :, :]  # shape: (6130, 128, 128)
    # norm_inputs, mean_inputs, std_inputs = normalize(inputs)
    # save_mean_std(mean_inputs, std_inputs, "inputs_mean_std.pkl")
    targets = full_data[:, 0:8, :, :]  # shape: (6130, 8, 128, 128)
    # norm_targets, mean_targets, std_targets = normalize(targets)
    # save_mean_std(mean_targets, std_targets, "targets_mean_std.pkl")
    # del inputs, targets

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
    # del norm_inputs, norm_targets

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_2().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(
        model, train_loader, test_loader, optimizer, criterion, device, epochs=15
    )

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
