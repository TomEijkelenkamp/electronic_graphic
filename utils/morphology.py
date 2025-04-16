import cv2
import numpy as np
import torch
import kornia.morphology as km
from datasets import load_dataset
import random
import os

# Load a shuffled slice directly
total_samples = 1300000  # Total number of samples in the dataset
segment_size = 10  # Number of validation samples
start = random.randint(0, total_samples - segment_size)
dataset = load_dataset("benjamin-paine/imagenet-1k-64x64", split=f"train[{start}:{start + segment_size}]")

# Save path
os.makedirs("dilated_images", exist_ok=True)

for kernel_size in range(5, 25, 5):
        
    # Pre-create kernel
    kernel_kornia = torch.ones((kernel_size, kernel_size), device="cuda")

    # Iterate over the dataset and apply dilation
    for i, sample in enumerate(dataset):
        img_pil = sample["image"]
        img_rgb = np.array(img_pil.convert("RGB"))

        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 255.0
        dilated = km.dilation(img_tensor, kernel_kornia)

        dilated_np = (dilated.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        dilated_np_bgr = cv2.cvtColor(dilated_np, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(f"dilated_images/dilated_kornia_{i}.png", dilated_np_bgr)
