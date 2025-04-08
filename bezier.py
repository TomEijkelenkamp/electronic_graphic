import torch

def cubic_bezier(control_points, num_samples=100):
    """Computes a cubic Bézier curve for parameter t (batched)."""
    p0, p1, p2, p3 = control_points[:, 0], control_points[:, 1], control_points[:, 2], control_points[:, 3]

    p0, p1, p2, p3 = [p.unsqueeze(1) for p in [p0, p1, p2, p3]]  # Shape: [batch, 1, 2]

    batch_size = control_points.shape[0]
    t = torch.linspace(0, 1, num_samples, device=control_points.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)

    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3  # Shape: [batch, num_samples, 2]

def draw_bezier_curve(control_points, color, thickness, sharpness, canvas, num_samples=100):
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
    sharpness = sharpness.view(batch_size, 1, 1, 1) # Shape: [batch, 1, 1, 1]
    
    intensity_map = (1 - torch.tanh(sharpness * (torch.sqrt(dist_sq) - thickness)))
    
    intensity_map = color * intensity_map.sum(dim=1, keepdim=True)  # Reduce over sampled points → [batch, 3, height, width]

    # Draw the curve on the canvas
    new_canvas = torch.nn.functional.softplus(canvas - intensity_map, beta=50)  # Ensures values remain positive

    return new_canvas