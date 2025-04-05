import torch
import torchvision.utils
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class DrawMLP(nn.Module):
    def __init__(self, image_width, image_height, num_points=4):
        super(DrawMLP, self).__init__()

        self.num_points = num_points  # Number of Bézier control points

        # Convolutional layers
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Compute flattened size dynamically
        sample_input = torch.zeros(1, 6, image_width, image_height)
        with torch.no_grad():
            flattened_size = self._compute_feature_map_size(sample_input)

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_points * 2 + 4)  # num_points * 2 + 2 (intensity & thickness)

    def _compute_feature_map_size(self, x):
        """Computes the feature map size after convolutions and pooling."""
        x = self.pool(F.silu(self.bn1(self.conv1(x))))
        x = self.pool(F.silu(self.bn2(self.conv2(x))))
        x = self.pool(F.silu(self.bn3(self.conv3(x))))
        return x.view(x.size(0), -1).shape[1]  # Extract feature size dynamically

    def forward(self, example_batch, drawing_batch):
        batch_size = example_batch.shape[0]

        assert not torch.isnan(example_batch).any(), "NaN detected in example_batch!"
        assert not torch.isnan(drawing_batch).any(), "NaN detected in drawing_batch!"


        # Concatenate example and drawing images along the channel dimension
        x = torch.cat((example_batch, drawing_batch), dim=1)
        # print("Input mean and std before conv1:", x.mean(), x.std())

        # print(f"Input shape: {x.shape}")
        # print(x)

        # Apply convolutional layers with activation and pooling
        x = self.pool(F.silu(self.bn1(self.conv1(x))))
        # print(f"After conv1: {x.shape}")
        # print(x)
        x = self.pool(F.silu(self.bn2(self.conv2(x))))
        # print(f"After conv2: {x.shape}")
        # print(x)
        x = self.pool(F.silu(self.bn3(self.conv3(x))))
        # print(f"After conv3: {x.shape}")
        # print(x)

        # Flatten
        x = x.view(batch_size, -1)
        # print(f"Flattened shape: {x.shape}")
        # print(x)

        # Fully connected layers with activation
        x = F.silu(self.fc1(x))
        # print(f"After fc1: {x.shape}")
        # print(x)
        x = F.silu(self.fc2(x))
        # print(f"After fc2: {x.shape}")
        # print(x)
        x = torch.sigmoid(self.fc3(x))  # Normalize output to [0, 1]
        # print(f"After fc3: {x.shape}")
        # print(x)

        # Split outputs properly
        control_points = x[:, :self.num_points * 2].view(batch_size, self.num_points, 2)
        color = x[:, self.num_points * 2:self.num_points * 2 + 3] + 0.01  # Avoid zero intensity
        thickness = x[:, self.num_points * 2 + 3] * 3.0 + 0.5  # Scale thickness

        return control_points, color, thickness

class MNISTDataset(Dataset):
    def __init__(self, dataset_split, image_width, image_height):
        self.data = dataset[dataset_split]
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.Resize((self.image_width, self.image_height)),  # Resize
            transforms.ToTensor()  # Convert to tensor, auto normalizes to [0,1]
        ])
        self.epoch = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"]  # PIL image
        img = self.transform(img)

        # Blend factor increases over time (clipped to 1.0 max)
        blend_factor = min(self.epoch * 0.01, 1.0)

        # Original grayscale expanded to RGB
        grayscale_img = img.repeat(3, 1, 1)

        # Random color tint (same as before)
        rand_color = torch.rand(3) * 0.75 + 0.25
        color_tinted_img = img * rand_color.view(3, 1, 1)

        # Blend between grayscale and color-tinted versions
        blended_img = (1 - blend_factor) * grayscale_img + blend_factor * color_tinted_img

        # Invert the image
        final_img = 1 - blended_img

        return final_img

class ImageDataset(Dataset):
    def __init__(self, dataset_path, image_width, image_height):
        self.dataset_path = dataset_path
        self.image_width = image_width
        self.image_height = image_height
        self.image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')])
        self.epoch = 0

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_path, self.image_files[idx])
        target_image = self.load_image(image_path)
        return target_image
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_path, self.image_files[idx])
        img = self.load_image(image_path)

        # Blend factor increases over time (clipped to 1.0 max)
        blend_factor = min(self.epoch * 0.01, 1.0)

        # Original grayscale expanded to RGB
        grayscale_img = img.repeat(3, 1, 1)

        # Inverted grayscale image
        inverted_img = 1 - grayscale_img

        # Random color tint (same as before)
        rand_color = torch.rand(3) * 0.75 + 0.25
        color_tinted_img = inverted_img * rand_color.view(3, 1, 1)

        # Blend between grayscale and color-tinted versions
        final_img = (1 - blend_factor) * inverted_img + blend_factor * color_tinted_img

        # Invert the image
        final_img = 1 - final_img

        return final_img

    def load_image(self, image_path):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.Resize((self.image_width, self.image_height)),  # Resize
            transforms.ToTensor()  # Convert to tensor, auto normalizes to [0,1]
        ])
        img = transform(Image.open(image_path))
        return img

def cubic_bezier(control_points, num_samples=100):
    """Computes a cubic Bézier curve for parameter t (batched)."""
    p0, p1, p2, p3 = control_points[:, 0], control_points[:, 1], control_points[:, 2], control_points[:, 3]

    p0, p1, p2, p3 = [p.unsqueeze(1) for p in [p0, p1, p2, p3]]  # Shape: [batch, 1, 2]

    batch_size = control_points.shape[0]
    t = torch.linspace(0, 1, num_samples, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)

    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3  # Shape: [batch, num_samples, 2]

def draw_bezier_curve(control_points, color, thickness, canvas, num_samples=100):
    """Draws a Bézier curve on a batch of canvases using a Gaussian function (fully vectorized)."""
    batch_size, _, height, width = canvas.shape
    device = canvas.device  # Ensure all tensors are on the same device

    # Compute Bézier curve points
    curve_points = cubic_bezier(control_points, num_samples)  # Shape: [batch, num_samples, 2]

    # Scale from normalized [0,1] to pixel coordinates
    curve_points[..., 0] *= (width - 1)  # Scale X
    curve_points[..., 1] *= (height - 1)  # Scale Y

    # Create a coordinate grid: [batch, height, width, 2]
    y_coords = torch.arange(height, device=device).view(1, height, 1).expand(batch_size, height, width)
    x_coords = torch.arange(width, device=device).view(1, 1, width).expand(batch_size, height, width)
    grid = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(1)  # Shape: [batch, 1, height, width, 2]

    # Expand curve points for broadcasting
    curve_points = curve_points.unsqueeze(2).unsqueeze(3)  # Shape: [batch, num_samples, 1, 1, 2]

    # Compute squared Euclidean distances
    dist_sq = torch.sum((grid - curve_points) ** 2, dim=-1)  # Shape: [batch, num_samples, height, width]

    # 🔹 **Fix: Ensure Thickness & Intensity Have Correct Shapes**
    thickness = thickness.view(batch_size, 1, 1, 1) # Shape: [batch, 1, 1, 1]
    color = color.view(batch_size, 3, 1, 1) # Shape: [batch, 3, 1, 1]

    sharpness = 5
    intensity_map = (1 - torch.tanh(sharpness * (torch.sqrt(dist_sq) - thickness)))

    # Compute Gaussian intensity and sum over sampled points
    # intensity_map = intensity * torch.exp(-dist_sq / (2 * thickness ** 2))  # Shape: [batch, num_samples, height, width]

    intensity_map = color * intensity_map.sum(dim=1, keepdim=True)  # Reduce over sampled points → [batch, 3, height, width]

    # Draw the curve on the canvas
    new_canvas = torch.nn.functional.softplus(canvas - intensity_map, beta=50)  # Ensures values remain positive
    # new_canvas = torch.nn.functional.relu(canvas - intensity_map)  # Apply ReLU to ensure non-negativity

    return new_canvas


def save_results(canvas, example, prefix="output", folder="test_images"):
    """Saves each canvas in the batch as a separate PNG image."""
    batch_size = canvas.shape[0]
    for i in range(batch_size):
        combined = torch.cat((example[i], canvas[i]), dim=2)

        filename = os.path.join(folder, f"{prefix}_{i}.png")
        # Concat the canvas with the original image for visualization

        torchvision.utils.save_image(combined, filename)

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0


# Load MNIST dataset from Hugging Face
# dataset = load_dataset("ylecun/mnist")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_folder = "results//char_dataset_results_3"
if not os.path.exists(image_folder):
    os.makedirs(image_folder, exist_ok=True)

# Random batch of Control points: 4 images, 4 control points (2D)
num_points = 4
# batch_size = 8
# height, width = 128, 128  # Canvas size
batch_size = 192
height, width = 64, 64  # Canvas size

# Initialize the dataset and dataloader
dataset = ImageDataset(dataset_path=".//data//character_images//", image_width=height, image_height=width)
# dataset = MNISTDataset(dataset_split="train", image_width=height, image_height=width)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



# Initialize the model
model = DrawMLP(image_width=width, image_height=height, num_points=num_points).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_curves = 5  # Number of curves to draw

start_epoch = load_checkpoint(model, optimizer, filename="models//checkpoint.pth")

# Training loop
for epoch in range(start_epoch, 500):
    dataset.epoch = epoch  # Update the dataset's epoch for blending
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to the appropriate device
        example_batch = batch.to(device)
        # print(control_points)

        batch_size, _, height, width = example_batch.shape
        # print(f"Batch size: {batch_size}, Height: {height}, Width: {width}")

        # Shape: [batch_size, 1, height, width]
        canvas = torch.ones(batch_size, 3, height, width, device=device, requires_grad=True)

        for i in range(num_curves):

            optimizer.zero_grad()  # Reset gradients

            # Control points in the range [0, 1]
            control_points, color, thickness = model(example_batch, canvas)  # Shape: [batch_size, num_points, 2]


            # print(f"Control points shape: {control_points.shape}, Color shape: {color.shape}, Thickness shape: {thickness.shape}")
            # print(control_points)
            # print(color)
            # print(thickness)
            # Draw the curves
            new_canvas = draw_bezier_curve(control_points, color, thickness, canvas)

            # Compute the mse loss between the new canvas and the example batch
            # print(f"Canvas shape: {new_canvas.shape}, Example batch shape: {example_batch.shape}")
            # print(new_canvas)
            # print(example_batch)
            loss = torch.nn.functional.mse_loss(new_canvas, example_batch)

            # print(f"Epoch {epoch}, Batch {batch_idx}, Curve {i}, Loss: {loss.item()}")

            loss.backward()  # Retain graph for further backward passes
            optimizer.step()  # Update weights

            canvas = new_canvas.detach().clone()  # Detach the canvas to avoid tracking gradients

        if batch_idx % 25 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # Save the canvas to a file
    save_results(new_canvas, example_batch, prefix=f"mnist_{epoch}", folder=image_folder)

    save_checkpoint(model, optimizer, epoch, filename="models//checkpoint.pth")


            

