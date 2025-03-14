from dotenv import load_dotenv
from datasets import load_dataset
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import subprocess
from IPython.display import display

load_dotenv()  # Load secrets from .env file

api_key = os.getenv('API_KEY')
db_password = os.getenv('DB_PASSWORD')


def make_mask(labelled_bbox, image):
  x_min_ones, y_min_ones, width_ones, height_ones = labelled_bbox
  x_min_ones, y_min_ones, width_ones, height_ones = int(x_min_ones), int(y_min_ones), int(width_ones), int(height_ones)
  mask_instance = np.zeros((image.width,image.height))

  last_x = x_min_ones+width_ones
  last_y = y_min_ones+height_ones
  mask_instance[x_min_ones:last_x, y_min_ones:last_y] = np.ones((int(width_ones),int(height_ones)))
  return mask_instance.T


if __name__ == '__main__':

    print(f"API Key: {api_key}, DB Password: {db_password}")


    # From https://huggingface.co/datasets/keremberke/satellite-building-segmentation?row=0
    ds = load_dataset("keremberke/satellite-building-segmentation", name="full")
    example = ds['train'][0]

    im=example["image"]

    subprocess.getoutput("display %s"%im) #display not working
    print(example["image"]) 

    example["objects"]['bbox']
    print(example.keys())

    print(example["objects"]['bbox'])



    image = example["image"].copy()  # Create a copy to avoid modifying the original

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox in example["objects"]["bbox"]:
        x_min, y_min, width, height = bbox
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

    labelled_bbox = example["objects"]["bbox"][0]
    mask_instance = make_mask(labelled_bbox, image)
    plt.imshow(mask_instance, cmap='gray')
    plt.show()

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import cv2

    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(np.array(example["image"]))
    print(len(masks))
    print(masks[0].keys())

    binary_array = masks[10]['segmentation'].astype(int)
    plt.imshow(binary_array, cmap='gray')
    plt.show()

    for m in masks:
        print(m['bbox'])

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    image2 = example["image"].copy()  # Create a copy to avoid modifying the original

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for m in masks:
        bbox = m['bbox']
        x_min, y_min, width, height = bbox
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

    from scipy.spatial import distance

    for sam_box in masks:
        sam_seg = sam_box['segmentation'].astype(int)
        for label_box in example["objects"]["bbox"]:
            label_seg = make_mask(label_box, image)
            iou = np.sum(np.logical_and(sam_seg, label_seg)) / np.sum(np.logical_or(sam_seg, label_seg))
            if iou>0.3:
                print(iou)
                # Create a figure with 1 row and 2 columns
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                # Display the first image in the first subplot
                axes[0].imshow(label_seg, cmap='gray')
                axes[0].set_title('Labelled Segmentation')
                axes[0].axis('off')  # Hide axes for better visualization

                # Display the second image in the second subplot
                axes[1].imshow(sam_seg, cmap='gray')
                axes[1].set_title('SAM Segmentation')
                axes[1].axis('off')  # Hide axes for better visualization

                # Adjust spacing between subplots
                plt.tight_layout()
                plt.show()