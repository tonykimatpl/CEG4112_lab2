import os
import torch
import torchvision
import urllib.request

def main():
    # Print PyTorch and CUDA information
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())

    # Ensure the required directories exist
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Download an example image
    dog_image_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg"
    dog_image_path = os.path.join(images_dir, "dog.jpg")
    if not os.path.exists(dog_image_path):  # Avoid re-downloading
        print("Downloading the example image...")
        urllib.request.urlretrieve(dog_image_url, dog_image_path)
    else:
        print("Example image already exists at:", dog_image_path)

    # Download the pre-trained model
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    model_path = "sam_vit_h_4b8939.pth"
    if not os.path.exists(model_path):  # Avoid re-downloading
        print("Downloading the pre-trained model...")
        urllib.request.urlretrieve(model_url, model_path)
    else:
        print("Pre-trained model already exists at:", model_path)

if __name__ == "__main__":
    main()
