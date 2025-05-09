#!/usr/bin/env -S submit -M 4000 -m 7500 -f python -u
import numpy as np
from natsort import natsorted
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt
import sunpy.visualization.colormaps as cm
from codecarbon import track_emissions

# ds = xr.open_dataset("../../../all128.nc")
# print(f"{ds.channel=}")

# selected_channel = ds['DN'].sel(channel='171A')
# print(f"{selected_channel=}")

# img = selected_channel.isel(time=0)
# cmap = plt.get_cmap('sdoaia171')
# img.plot(cmap=cmap)
# plt.savefig("single.png")
# plt.close()

# keys = natsorted(ds['channel'].data)
# fig, axes = plt.subplots(3, 3, figsize=(8, 8))

# for ax, key in zip(axes.ravel(), keys):
#     data = ds['DN'].sel(channel=key).isel(time=0)
#     cmap = plt.get_cmap(f'sdoaia{key[:-1]}')
#     im = data.plot(cmap=cmap, ax=ax, add_colorbar=False)
#     ax.set_title(key)
#     ax.axis('off')

# plt.savefig("multiple.png")
# plt.close()

# now solve it ;)

# you have many different DeepLearning library available


#!/usr/bin/env python3


import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xarray as xr

# 1. Load data
ds = xr.open_dataset("../../all128.nc")

input_channel = '171A'
target_channels = ['131A', '193A', '211A', '304A', '335A', '94A', '1600A', '1700A']
time_steps = 6000

# 2. Prepare input and output
inputs, targets = [], []

for t in range(time_steps):
    input_img = ds['DN'].sel(channel=input_channel).isel(time=t).values
    target_imgs = [ds['DN'].sel(channel=ch).isel(time=t).values for ch in target_channels]

    input_img = np.nan_to_num(input_img)
    target_imgs = [np.nan_to_num(img) for img in target_imgs]

    inputs.append(input_img[np.newaxis, :, :])  # shape: (1, H, W)
    targets.append(np.stack(target_imgs, axis=0))  # shape: (8, H, W)

X = np.stack(inputs)  # shape: (time_steps, 1, H, W)
Y = np.stack(targets)  # shape: (time_steps, 8, H, W)

# Normalize
X = X / X.max()
Y = Y / Y.max()

# 3. Dataset class
class SolarPredictionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 4. Train/val/test split
X_train, X_remaining, y_train, y_remaining = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = SolarPredictionDataset(X_train, y_train)
val_dataset = SolarPredictionDataset(X_val, y_val)
test_dataset = SolarPredictionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# 5. U-Net-style model (final output = 8 channels now)
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # down to 64x64
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # back to 128x128
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 1),  # final output: 8 channels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# 6. Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_losses = []
val_losses = []
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_Y).item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.6f} — Val Loss: {avg_val_loss:.6f}")

# 7. Plot loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()


import matplotlib.pyplot as plt

test_index = 0
input_cmap = plt.get_cmap('sdoaia94')
output_cmap = plt.get_cmap('sdoaia131')

# Get prediction
model.eval()
with torch.no_grad():
    input_img = X_test[test_index].unsqueeze(0).to(device)  # add batch dimension
    Y_test_prediction = model(input_img).cpu().squeeze().numpy()  # shape: (8, H, W)

# Ground truth
Y_true = y_test[test_index].cpu().numpy()  # shape: (8, H, W)
input_image = X_test[test_index][0].cpu().numpy()  # shape: (H, W)

# Plotting
n_channels = 8
fig, axes = plt.subplots(3, n_channels, figsize=(3 * n_channels, 9))

# First row: input image duplicated for each column
for i in range(n_channels):
    axes[0, i].imshow(input_image, cmap=input_cmap)
    axes[0, i].set_title(f'Input 171A')
    axes[0, i].axis('off')

# Second row: model outputs
for i in range(n_channels):
    axes[1, i].imshow(Y_test_prediction[i], cmap=output_cmap)
    axes[1, i].set_title(f'Predicted: {target_channels[i]}')
    axes[1, i].axis('off')

# Third row: ground-truth outputs
for i in range(n_channels):
    axes[2, i].imshow(Y_true[i], cmap=output_cmap)
    axes[2, i].set_title(f'Target: {target_channels[i]}')
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig("comparison2000.png")
plt.show()