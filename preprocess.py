from dotenv import load_dotenv
from datasets import load_dataset
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import subprocess
from IPython.display import display
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from scipy.spatial import distance

load_dotenv()  # Load secrets from .env file

api_key = os.getenv('API_KEY')
db_password = os.getenv('DB_PASSWORD')

def make_mask(labelled_bbox, image):
    x_min_ones, y_min_ones, width_ones, height_ones = labelled_bbox
    x_min_ones, y_min_ones, width_ones, height_ones = int(x_min_ones), int(y_min_ones), int(width_ones), int(height_ones)
    mask_instance = np.zeros((image.width, image.height))
    last_x = x_min_ones + width_ones
    last_y = y_min_ones + height_ones
    mask_instance[x_min_ones:last_x, y_min_ones:last_y] = np.ones((int(width_ones), int(height_ones)))
    return mask_instance.T

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

if __name__ == '__main__':
    print(f"API Key: {api_key}, DB Password: {db_password}")

    # Load the dataset
    ds = load_dataset("keremberke/satellite-building-segmentation", name="mini")

    # Split the dataset
    train_ds = ds['train']
    validation_ds = ds['validation']
    test_ds = ds['test']

    # SAM setup
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # Replace with the path to your SAM checkpoint
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    def process_data(example):
        image = example["image"].copy()
        image_np = np.array(image)
        masks = mask_generator.generate(image_np)

        sam_masks = []
        for sam_box in masks:
            sam_seg = sam_box['segmentation'].astype(int)
            iou_found = False
            for label_box in example["objects"]["bbox"]:
                label_seg = make_mask(label_box, image)
                iou = np.sum(np.logical_and(sam_seg, label_seg)) / np.sum(np.logical_or(sam_seg, label_seg))
                if iou > 0.3:
                    sam_masks.append(sam_seg)
                    iou_found = True
                    break
            if not iou_found:
                sam_masks.append(np.zeros_like(sam_seg)) # append a zero mask if no iou is found.

        example["sam_masks"] = sam_masks
        return example

    # Process each split
    train_ds = train_ds.map(process_data)
    validation_ds = validation_ds.map(process_data)
    test_ds = test_ds.map(process_data)

    # Example visualization (using the first item in the training set)
    example = train_ds[0]
    image = example["image"]
    sam_masks = example["sam_masks"]

    fig, axes = plt.subplots(1, len(sam_masks) + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis('off')

    for i, mask in enumerate(sam_masks):
        axes[i + 1].imshow(mask, cmap='gray')
        axes[i + 1].set_title(f"SAM Mask {i + 1}")
        axes[i + 1].axis('off')

    plt.show()

    # Now train_ds, validation_ds, and test_ds contain the images and corresponding SAM-generated masks.
    print("Dataset processing complete.")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm

# UNet Implementation
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        return torch.sigmoid(self.conv(dec1))


# PyTorch Dataset Class
class SAMSegmentationDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = np.array(example["image"])  # Convert PIL image to numpy array
        masks = np.array(example["sam_masks"]).sum(axis=0)  # Combine all masks into one
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            masks = self.transform(torch.tensor(masks, dtype=torch.float32))

        return image, masks


# Dataset Transforms (Normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataloader
train_dataset = SAMSegmentationDataset(train_ds, transform=transform)
val_dataset = SAMSegmentationDataset(validation_ds, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    # Validation Loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dataloader)}, Val Loss: {val_loss/len(val_dataloader)}")

# Save the model
torch.save(model.state_dict(), "unet_sam_segmentation.pth")
