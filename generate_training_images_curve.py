import torch
import torchvision.utils
import os
import math


def cubic_bezier(control_points, num_samples=400):
    """Computes a cubic Bézier curve for parameter t (batched)."""
    p0, p1, p2, p3 = control_points[:, 0], control_points[:, 1], control_points[:, 2], control_points[:, 3]

    p0, p1, p2, p3 = [p.unsqueeze(1) for p in [p0, p1, p2, p3]]  # Shape: [batch, 1, 2]

    batch_size = control_points.shape[0]
    t = torch.linspace(0, 1, num_samples, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)

    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3  # Shape: [batch, num_samples, 2]

def smooth_tanh(x, alpha=2):
    return 0.5 * (torch.tanh(alpha * x) + 1)

def draw_bezier_curve(control_points, intensity, thickness, canvas, num_samples=100):
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
    intensity = intensity.view(batch_size, 1, 1, 1) # Shape: [batch, 1, 1, 1]

    # Compute Gaussian intensity and sum over sampled points

    # Lines style tanh
    sharpness = 5
    intensity_map = (1 - torch.tanh(sharpness * (torch.sqrt(dist_sq) - thickness)))
    intensity_map = intensity * intensity_map.sum(dim=1, keepdim=True)  # Reduce over sampled points → [batch, 1, height, width]

    # Lines style gaussian
    # intensity_map = torch.exp(-dist_sq / (2 * thickness ** 2))  # Shape: [batch, num_samples, height, width]
    # intensity_map = intensity * intensity_map.sum(dim=1, keepdim=True)  # Reduce over sampled points → [batch, 1, height, width]

    # Line style gaussian with exponential drop
    # intensity_map = torch.exp(-torch.sqrt(dist_sq) / thickness)
    # intensity_map = intensity * intensity_map.sum(dim=1, keepdim=True)  # Reduce over sampled points → [batch, 1, height, width]

    # Reciprocal Sharpness
    # intensity_map = intensity / (1 + dist_sq / thickness**2)
    # intensity_map = intensity_map.sum(dim=1, keepdim=True)  # Reduce over sampled points → [batch, 1, height, width]

    # Inverse Quadratic
    # intensity_map = intensity / (1 + (dist_sq / thickness**2))**2
    # intensity_map = intensity_map.sum(dim=1, keepdim=True)  # Reduce over sampled points → [batch, 1, height, width]

    # Draw the curve on the canvas
    new_canvas = torch.nn.functional.softplus(canvas - intensity_map, beta=50)  # Ensures values remain positive
    # new_canvas = torch.nn.functional.relu(canvas - intensity_map)  # Apply ReLU to ensure non-negativity

    return new_canvas


def save_results(canvas, batch_idx, prefix="image", folder="test_images"):
    """Saves each canvas in the batch as a separate PNG image."""
    batch_size = canvas.shape[0]
    for i in range(batch_size):
        index = batch_idx * batch_size + i  # Calculate the index for the filename
        filename = os.path.join(folder, f"{prefix}_{index}.png")
        # Concat the canvas with the original image for visualization

        torchvision.utils.save_image(canvas[i], filename)


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_folder = "style_test_images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder, exist_ok=True)

# Random batch of Control points: 4 images, 4 control points (2D)
batch_size = 64
num_points = 4
height, width = 64, 64  # Canvas size

num_curves = 3  # Number of curves to draw

for batch_idx in range(1):  # Simulate a single batch
    print(f"Batch size: {batch_size}, Height: {height}, Width: {width}")

    # Shape: [batch_size, 1, height, width]
    canvas = torch.ones(batch_size, 1, height, width, device=device, requires_grad=True)

    for i in range(num_curves):

        # Control points in the range [0, 1]
        control_points, intensity, thickness = torch.rand(batch_size, num_points, 2, device=device), (torch.rand(batch_size, device=device) / 10.0) + 0.00, torch.rand(batch_size, device=device) * 3.0 + 0.5

        print(f"Intensity:")
        print(intensity)
        # Draw the curves
        new_canvas = draw_bezier_curve(control_points, intensity, thickness, canvas)

        canvas = new_canvas.detach().clone()  # Detach the canvas to avoid tracking gradients


    # Save the canvas to a file
    save_results(new_canvas, batch_idx, folder=image_folder)

            

