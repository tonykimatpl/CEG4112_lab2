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

