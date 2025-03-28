import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
class DrawMLP(torch.nn.Module):
    def __init__(self, hidden_size):
        super(DrawMLP, self).__init__()
        self.fc1 = torch.nn.Linear(4, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Normalize output (0-1)
        return x

# Convert normalized coords to pixel values
def denormalize_coords(coords, width, height):
    x1, y1, x2, y2 = coords
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

# Draw a line on an image
def draw_line(image, x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]:
            image[y1, x1] = 0  # Draw black pixel
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return image

# Function to evaluate the model
def evaluate_model(model_path, eval_folder, output_folder, image_width=224, image_height=224, max_actions_per_image=20):
    # Load trained model
    model = DrawMLP(hidden_size=128)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of evaluation images
    eval_images = sorted([f for f in os.listdir(eval_folder) if f.endswith('.png')])

    for img_name in eval_images:
        img_path = os.path.join(eval_folder, img_name)
        
        # Start with a blank canvas
        drawn_image = np.ones((image_height, image_width), dtype=np.float32)

        for _ in range(max_actions_per_image):
            random_input = torch.rand(1, 4)  # Random input for line prediction
            predicted_coords = model(random_input)

            x1, y1, x2, y2 = denormalize_coords(predicted_coords.detach().numpy()[0], image_width, image_height)

            drawn_image = draw_line(drawn_image, x1, y1, x2, y2)

        # Convert to PIL image and save
        output_img = Image.fromarray((drawn_image * 255).astype(np.uint8))
        output_path = os.path.join(output_folder, img_name)
        output_img.save(output_path)

        print(f"Processed: {img_name}")

# Example usage
evaluate_model(
    model_path="draw_mlp_epoch_5000.pth",  # Change if needed
    eval_folder="path/to/evaluation/folder",
    output_folder="path/to/output/folder"
)
