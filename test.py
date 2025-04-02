import torch
import torchvision.utils
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader


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

def cubic_bezier(control_points, num_samples=400):
    """Computes a cubic Bézier curve for parameter t (batched)."""
    p0, p1, p2, p3 = control_points[:, 0], control_points[:, 1], control_points[:, 2], control_points[:, 3]

    p0, p1, p2, p3 = [p.unsqueeze(1) for p in [p0, p1, p2, p3]]  # Shape: [batch, 1, 2]

    batch_size = control_points.shape[0]
    t = torch.linspace(0, 1, num_samples, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)

    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3  # Shape: [batch, num_samples, 2]

def draw_bezier_curve(control_points, canvas, num_samples=400, sigma=1.5):
    """Draws a Bézier curve on a batch of canvases using a Gaussian function (fully vectorized)."""
    batch_size, _, height, width = canvas.shape
    curve_points = cubic_bezier(control_points, num_samples)  # Shape: [batch, num_samples, 2]

    # Convert normalized curve points to pixel coordinates
    curve_points = (curve_points * (height - 1))  # Scale to image size, shape: [batch, num_samples, 2]

    # Create a coordinate grid of shape [batch, height, width, 2]
    y_coords = torch.arange(height, device=device).view(1, height, 1).expand(batch_size, height, width)
    x_coords = torch.arange(width, device=device).view(1, 1, width).expand(batch_size, height, width)
    grid = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(1)  # Shape: [batch, 1, height, width, 2]

    # Expand curve points for broadcasting
    curve_points = curve_points.unsqueeze(2).unsqueeze(3)  # Shape: [batch, num_samples, 1, 1, 2]

    # Compute squared distances in a vectorized way
    dist_sq = torch.sum((grid - curve_points) ** 2, dim=-1)  # Shape: [batch, num_samples, height, width]

    # Compute Gaussian intensity and sum over sampled points
    intensity_map = torch.exp(-dist_sq / (2 * sigma ** 2)).sum(dim=1, keepdim=True)  # Shape: [batch, 1, height, width]

    # Draw the curve on the canvas
    new_canvas = 0.5 * (torch.tanh(canvas - intensity_map) + 1)

    return new_canvas


def save_canvases(canvas, prefix="output", folder="test_images"):
    """Saves each canvas in the batch as a separate PNG image."""
    batch_size = canvas.shape[0]
    for i in range(batch_size):
        filename = os.path.join(folder, f"{prefix}_{i}.png")
        torchvision.utils.save_image(canvas[i], filename)


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_folder = "test_images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder, exist_ok=True)

# Random batch of Control points: 4 images, 4 control points (2D)
batch_size = 4
num_points = 4
height, width = 256, 256  # Canvas size

# Initialize the dataset and dataloader
dataset = ImageDataset(dataset_path=".//data//train//", image_width=height, image_height=width)
batch_size = 4  # You can adjust this based on your GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch_idx, batch in enumerate(dataloader):
    # Move batch to the appropriate device
    example_batch = batch.to(device)

    # Control points in the range [0, 1]
    control_points = torch.rand(batch_size, num_points, 2, device=device, requires_grad=True)
    print(control_points)

    # Shape: [batch_size, 1, height, width]
    canvas = torch.ones(batch_size, 1, height, width, device=device, requires_grad=True)

    # Draw the curves
    new_canvas = draw_bezier_curve(control_points, canvas)

    # Compute the mse loss between the new canvas and the example batch
    loss = torch.nn.functional.mse_loss(new_canvas, example_batch)
    print(f"Loss: {loss.item()}")

    loss.backward()
    print(control_points.grad)  # Should now be populated

    print(f"Canvas shape: {canvas.shape}")  # Expected: [4, 1, 64, 64]

    # Save the canvas to a file
    save_canvases(new_canvas)

