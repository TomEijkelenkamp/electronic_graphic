import torch

def cubic_bezier(t, p0, p1, p2, p3):
    """Computes a cubic Bézier curve for parameter t (batched)."""
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3

def drawing_batch(example_batch, drawing_batch, control_points, num_points=1000, t_values=None):
    """
    Draws cubic Bézier curves on a batch of images using vectorized operations with anti-aliasing.
    Clamps only modified pixels to keep values within [0, 1] and computes total change in pixel values.
    
    Args:
        example_batch (torch.Tensor): The batch of reference images for comparison.
        drawing_batch (torch.Tensor): The input tensor (B, H, W) where B is the batch size.
        control_points (torch.Tensor): A (B, 4, 2) tensor with x, y control points for each image.
        num_points (int): Number of points to sample along each curve.
        t_values (torch.Tensor, optional): Precomputed t values to avoid recomputation.
    """
    device = drawing_batch.device  # Ensure we use the correct device
    batch_size, height, width = drawing_batch.shape
    
    # Precompute t_values if not provided
    if t_values is None:
        t_values = torch.linspace(0, 1, num_points, device=device).view(1, -1, 1).expand(batch_size, -1, -1)
    
    # Compute curve points for all images in batch
    curve_points = cubic_bezier(t_values, control_points[:, 0:1], control_points[:, 1:2], control_points[:, 2:3], control_points[:, 3:4])
    
    # Ensure points are within tensor bounds
    curve_points = torch.clamp(curve_points, min=0, max=torch.tensor([width - 1.0, height - 1.0], device=device))
    
    # Get integer and fractional parts for anti-aliasing
    int_points = curve_points.floor().long()
    frac = curve_points - int_points.float()
    
    x0, y0 = int_points[..., 0], int_points[..., 1]
    x1, y1 = torch.clamp(x0 + 1, max=width - 1), torch.clamp(y0 + 1, max=height - 1)
    
    # Bilinear weight calculation
    w00 = (1 - frac[..., 0]) * (1 - frac[..., 1])
    w01 = (1 - frac[..., 0]) * frac[..., 1]
    w10 = frac[..., 0] * (1 - frac[..., 1])
    w11 = frac[..., 0] * frac[..., 1]
    
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num_points)
    
    # Combine all indices and values for a single scatter_add_
    indices = torch.cat([
        torch.stack((batch_indices, y0, x0), dim=-1),
        torch.stack((batch_indices, y1, x0), dim=-1),
        torch.stack((batch_indices, y0, x1), dim=-1),
        torch.stack((batch_indices, y1, x1), dim=-1)
    ], dim=0).view(-1, 3)
    
    values = torch.cat([w00, w01, w10, w11], dim=0).view(-1)
    
    # Store original values before modification
    original_values = drawing_batch[indices[:, 0], indices[:, 1], indices[:, 2]].clone()
    
    # Apply scatter addition
    drawing_batch.scatter_add_(0, indices.t(), values.to(drawing_batch.dtype))
    
    # Clamp only modified pixels
    modified_values = drawing_batch[indices[:, 0], indices[:, 1], indices[:, 2]]
    drawing_batch.scatter_(0, indices.t(), torch.clamp(modified_values, 0, 1))
    
    # Compute total change in pixel values
    total_change = torch.abs(drawing_batch[indices[:, 0], indices[:, 1], indices[:, 2]] - original_values).sum()
    
    # Compute the error on the example batch
    total_error = torch.abs(example_batch[indices[:, 0], indices[:, 1], indices[:, 2]] - drawing_batch[indices[:, 0], indices[:, 1], indices[:, 2]]).sum()
    
    return drawing_batch, total_change, total_error

# Example usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
drawing_batch_tensor = torch.zeros((batch_size, 256, 256), device=device).half()  # Convert to float16 for performance
control_points = torch.tensor([
    [[50, 200], [100, 50], [150, 50], [200, 200]],
    [[30, 220], [90, 60], [160, 70], [210, 190]],
    [[20, 180], [80, 40], [140, 60], [190, 210]],
    [[60, 190], [110, 55], [170, 80], [220, 180]]
], dtype=torch.float16, device=device)

t_values = torch.linspace(0, 1, 1000, device=device).view(1, -1, 1).expand(batch_size, -1, -1)  # Precompute t_values
image_with_curves, total_change, total_error = drawing_batch(drawing_batch_tensor, control_points, t_values=t_values)
