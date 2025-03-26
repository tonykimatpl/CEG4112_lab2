from datasets import load_dataset
ds = load_dataset("keremberke/satellite-building-segmentation", name="full")
import numpy as np

def make_mask(labelled_bbox, image):
    """Create a binary mask for the bounding box."""
    x_min, y_min, width, height = labelled_bbox
    x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)
    mask_instance = np.zeros((image.height, image.width))  # Initialize mask
    mask_instance[y_min:y_min + height, x_min:x_min + width] = 1  # Fill bbox
    return mask_instance

for i in range(len(ds["train"])):
    example = ds["train"][i]
    image = example["image"]
    masks = [make_mask(bbox, image) for bbox in example["objects"]["bbox"]]

from sklearn.model_selection import train_test_split

def split_dataset(dataset):
    """Split dataset into training, validation, and test sets."""
    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, validation_data, test_data

train_set, val_set, test_set = split_dataset(ds)

import os
import cv2

def save_labeled_dataset(dataset, output_dir):
    """Save images and masks to the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    for i, example in enumerate(dataset):
        image = np.array(example["image"])
        masks = [make_mask(bbox, image) for bbox in example["objects"]["bbox"]]
        combined_mask = np.any(masks, axis=0).astype(np.uint8)  # Combine all masks

        # Save image and mask
        image_path = os.path.join(output_dir, f"image_{i}.png")
        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, combined_mask * 255)  # Save mask as binary image

save_labeled_dataset(train_set, "output/train")
save_labeled_dataset(val_set, "output/val")
save_labeled_dataset(test_set, "output/test")

