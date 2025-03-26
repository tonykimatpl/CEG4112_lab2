import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from datasets import load_dataset
from PIL import Image
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 10
batch_size = 4
learning_rate = 0.001

# Load dataset
dataset = load_dataset("keremberke/satellite-building-segmentation", name="full")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Custom Dataset class
class SatelliteBuildingDataset(Dataset):
    def __init__(self, dataset, transform=None, mask_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        mask = self.dataset[idx]['label']

        image = image.convert("RGB")
        mask = mask.convert("L") # Convert to grayscale

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask.squeeze(0).long() # Remove channel dimension and convert to Long

# Create DataLoader
train_dataset = SatelliteBuildingDataset(dataset['train'], transform=transform, mask_transform=mask_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = SatelliteBuildingDataset(dataset['validation'], transform=transform, mask_transform=mask_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load DeepLabV3+ model
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)) # 2 classes: building and background
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader)}")

# Save the trained model
torch.save(model.state_dict(), "deeplabv3_satellite_buildings.pth")

print("Training finished!")