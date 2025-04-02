import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, dataset_path, image_width, image_height):
        self.dataset_path = dataset_path
        self.image_width = image_width
        self.image_height = image_height
        self.image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_path, self.image_files[idx])
        target_image = self.load_image(image_path)
        return target_image


    def load_image(self, image_path):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.Resize((self.image_width, self.image_height)),  # Resize
            transforms.ToTensor()  # Convert to tensor, auto normalizes to [0,1]
        ])
        img = transform(Image.open(image_path))
        return img

    # def load_image(self, image_path):
    #     img = Image.open(image_path).convert('L')  # Convert to grayscale
    #     img = img.resize((self.image_width, self.image_height))  # Ensure correct size
    #     img = np.array(img) / 255.0  # Normalize to [0,1]
    #     img = torch.tensor(img).unsqueeze(0)  # Add channel dimension: [1, H, W]
    #     return img
    
class DrawMLP(nn.Module):
    def __init__(self, image_width, image_height, hidden_size):
        super(DrawMLP, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Compute flattened size dynamically
        sample_input = torch.zeros(1, 2, image_width, image_height)
        with torch.no_grad():
            flattened_size = self._compute_feature_map_size(sample_input)

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 8)  # Output size is 8 (4 points with x, y coordinates)

    def _compute_feature_map_size(self, x):
        x = self.pool(F.silu(self.bn1(self.conv1(x))))
        x = self.pool(F.silu(self.bn2(self.conv2(x))))
        x = self.pool(F.silu(self.bn3(self.conv3(x))))
        x = self.pool(F.silu(self.bn4(self.conv4(x))))
        return x.numel()

    def forward(self, example_batch, drawning_batch):
        x = torch.cat((example_batch, drawning_batch), dim=1)

        x = self.pool(F.silu(self.bn1(self.conv1(x))))
        x = self.pool(F.silu(self.bn2(self.conv2(x))))
        x = self.pool(F.silu(self.bn3(self.conv3(x))))
        x = self.pool(F.silu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)

        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Normalize output

        return x

def cubic_bezier(t, p0, p1, p2, p3):
    """Computes a cubic Bézier curve for parameter t (batched)."""
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3

def estimate_curve_length(control_points, num_samples=16):
    """Estimates the Bézier curve length using piecewise linear approximation."""
    t_values = torch.linspace(0, 1, num_samples, device=control_points.device).view(1, -1, 1)
    curve_points = cubic_bezier(t_values, control_points[:, 0:1], control_points[:, 1:2], 
                                control_points[:, 2:3], control_points[:, 3:4])
    
    segment_lengths = torch.norm(curve_points[:, 1:] - curve_points[:, :-1], dim=-1)  # Distance between points
    return segment_lengths.sum(dim=1)  # Total curve length per batch item

def draw_batch(example_batch, drawing_batch, control_points):
    """
    Draws cubic Bézier curves on a batch of images using adaptive sampling.
    """
    device = drawing_batch.device
    batch_size, _, height, width = drawing_batch.shape

    with torch.no_grad():  # Exclude Bézier rendering from autograd
        # Rescale control points to [0, width-1] and [0, height-1]
        control_points[:, :, 0] *= (width - 1)
        control_points[:, :, 1] *= (height - 1)

        # Estimate curve lengths and determine adaptive num_points
        num_points = estimate_curve_length(control_points).long()
        max_num_points = num_points.max().item()
        
        # Generate t values with a valid mask
        steps_range = torch.arange(max_num_points, device=device).unsqueeze(0).expand(batch_size, -1)
        valid_mask = steps_range < num_points.unsqueeze(1)  # Per batch item
        
        t_values = steps_range / (num_points.unsqueeze(1) - 1).clamp(min=1)
        t_values[~valid_mask] = 0  # Avoid invalid values

        # Compute Bézier curve points
        curve_points = cubic_bezier(t_values.unsqueeze(-1), control_points[:, 0:1],
                                    control_points[:, 1:2], control_points[:, 2:3],
                                    control_points[:, 3:4])
        
        # Convert to integer coordinates
        int_points = curve_points.floor().long()
        frac = curve_points - int_points.float()
        x0, y0 = int_points[..., 0], int_points[..., 1]
        x1, y1 = x0 + 1, y0 + 1

        # Ensure coordinates are within bounds
        valid_x0 = (0 <= x0) & (x0 < width) & valid_mask
        valid_y0 = (0 <= y0) & (y0 < height) & valid_mask
        valid_x1 = (0 <= x1) & (x1 < width) & valid_mask
        valid_y1 = (0 <= y1) & (y1 < height) & valid_mask

        # Bilinear interpolation weights
        w00 = (1 - frac[..., 0]) * (1 - frac[..., 1])
        w01 = (1 - frac[..., 0]) * frac[..., 1]
        w10 = frac[..., 0] * (1 - frac[..., 1])
        w11 = frac[..., 0] * frac[..., 1]
        
        total_change = 0
        total_error = 0

        def update_pixel(x, y, w, valid):
            nonlocal total_change, total_error
            if valid:
                change = torch.clamp(drawing_batch[:, 0, y, x] + w, 0, 1) - drawing_batch[:, 0, y, x]
                drawing_batch[:, 0, y, x] += change
                total_change += change.sum()
                total_error += torch.abs(example_batch[:, 0, y, x] - drawing_batch[:, 0, y, x]).sum()

        # Apply updates
        update_pixel(x0, y0, w00, valid_x0 & valid_y0)
        update_pixel(x0, y1, w10, valid_x0 & valid_y1)
        update_pixel(x1, y0, w01, valid_x1 & valid_y0)
        update_pixel(x1, y1, w11, valid_x1 & valid_y1)
    
    return drawing_batch, total_change, total_error

def train_network(device, model, optimizer, num_epochs, dataloader, max_actions_per_image, save_interval=10):
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to the appropriate device
            example_batch = batch.to(device)

            # Initialize drawn image (white canvas) for the batch, requires gradients
            drawning_batch = torch.ones_like(batch, dtype=torch.float32, requires_grad=True).to(device)

            total_loss = 0

            for action in range(max_actions_per_image):
                # Let the model predict the coordinates
                predicted_coords = model(example_batch, drawning_batch)

                # Draw the line on the image canvas
                drawning_batch, total_change, total_error = draw_batch(example_batch, drawning_batch, predicted_coords.view(-1, 4, 2))

                # Compute loss only on the drawn pixels using the simplified loss
                loss = total_error / total_change if total_change > 0 else torch.tensor(0.0, device=device)

                optimizer.zero_grad()
                loss.backward()  # Retain graph for further backward passes
                optimizer.step()  # Update weights

                total_loss += loss

            if epoch % 1 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss:.4f}, Batch {batch_idx}/{len(dataloader)}")

        # Save the model every `save_interval` epochs
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            model_path = os.path.join(model_save_folder, f"draw_mlp_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved: {model_path}")

            # Save drawn images for all images in the batch (without showing them)
            print(f"Shape of example_batch: {example_batch.shape}")
            for i in range(example_batch.size(0)):  # Loop through each image in the batch
                drawn_image_path = os.path.join(drawings_save_folder, f"drawn_image_epoch_{epoch}_batch_{batch_idx}_image_{i}.png")
                img = drawning_batch[i].detach().cpu().numpy()
                print(f"Image shape: {img.shape}")
                plt.imsave(drawn_image_path, drawning_batch[i].detach().cpu().squeeze(0).numpy(), cmap='gray', format='png')
                print(f"Drawn image saved: {drawn_image_path}")

# Make sure you specify a folder where you want to save the models
model_save_folder = './models'  # Folder to save models
if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)

drawings_save_folder = './drawings'  # Folder to save drawn images
if not os.path.exists(drawings_save_folder):
    os.makedirs(drawings_save_folder)

# Initialize parameters
image_width, image_height = 224, 224
hidden_size = 128
num_epochs = 5000
max_actions_per_image = 100
dataset_path = ".//data//train//"  # Change this!

# Initialize the dataset and dataloader
dataset = ImageDataset(dataset_path=".//data//train//", image_width=image_width, image_height=image_height)
batch_size = 32  # You can adjust this based on your GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU
model = DrawMLP(image_width, image_height, hidden_size).to(device)
print("Model initialized and moved to device.")

# Ensure model parameters require gradients
for param in model.parameters():
    param.requires_grad = True

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Call the train function with DataLoader
train_network(device, model, optimizer, num_epochs=5000, dataloader=dataloader, max_actions_per_image=100, save_interval=4)
