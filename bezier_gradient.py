import torch

def cubic_bezier(t, p0, p1, p2, p3):
    """Computes a cubic Bézier curve for parameter t (batched)."""

    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3

def bezier_derivative(t, p0, p1, p2, p3):
    """Computes the derivative of the cubic Bézier curve for parameter t (batched)."""

    print(f"t: {t.shape}, p0: {p0.shape}, p1: {p1.shape}, p2: {p2.shape}, p3: {p3.shape}")
    # Expand the control points to match the shape of t
    p0 = p0.unsqueeze(1).expand(t.shape[0], -1)
    p1 = p1.unsqueeze(1).expand(t.shape[0], -1)
    p2 = p2.unsqueeze(1).expand(t.shape[0], -1)
    p3 = p3.unsqueeze(1).expand(t.shape[0], -1)
    print(f"t: {t.shape}, p0: {p0.shape}, p1: {p1.shape}, p2: {p2.shape}, p3: {p3.shape}")

    # Compute the derivative of the Bézier curve
    return 3 * (1 - t)**2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t**2 * (p3 - p2)

def bezier_curve_length(p0, p1, p2, p3, num_samples=1000):
    """Estimates the Bézier curve length using numerical integration."""
    # Define t values from 0 to 1
    t_values = torch.linspace(0, 1, num_samples)

    # Compute the Bézier curve at these t values
    curve_points = cubic_bezier(t_values, p0, p1, p2, p3)

    # Compute the magnitude (norm) of the derivative at each sample point
    magnitudes = torch.norm(curve_points, dim=-1)

    # Use the trapezoidal rule for numerical integration to estimate the length
    length = torch.sum((magnitudes[1:] + magnitudes[:-1]) * 0.5)  # Trapezoidal rule

    return length

# Use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example usage
control_points = torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 0.0], requires_grad=True, device=device)

# Control points for the cubic Bézier curve
p0 = control_points[0:2]
p1 = control_points[2:4]
p2 = control_points[4:6]
p3 = control_points[6:8]

# Compute the Bézier curve length
length = bezier_curve_length(p0, p1, p2, p3)
length.backward()  # Compute gradients
print(f"Estimated Bézier curve length: {length.item()}")
print(f"Gradient w.r.t. control points: {control_points.grad}")
