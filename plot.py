import torch
import numpy as np
import matplotlib.pyplot as plt

# Image size
height, width = 256, 256
sigma = 2.0  # Standard deviation of the Gaussian

# Create a distance map from a central point
center = torch.tensor([height // 2, width // 2], dtype=torch.float32)
y_coords = torch.arange(height).view(height, 1).expand(height, width)
x_coords = torch.arange(width).view(1, width).expand(height, width)
grid = torch.stack([x_coords, y_coords], dim=-1).float()

dist_sq = torch.sum((grid - center) ** 2, dim=-1)  # Squared distance

# Compute Gaussian intensity
intensity_map = torch.exp(-dist_sq / (2 * sigma ** 2))

# Plot the intensity map
plt.figure(figsize=(6, 6))
plt.imshow(intensity_map.numpy(), cmap='hot', origin='upper')
plt.colorbar(label='Intensity')
plt.title("Gaussian Intensity Decay Over Image")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()
