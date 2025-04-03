import numpy as np
from PIL import Image, ImageDraw
import random

def generate_image(shape_type="circle", size=(224, 224)):
    """Generates an image with a simple shape."""
    # Create a blank canvas (128x128, grey scale)
    img = Image.new('L', size, color=random.randint(25,200))  # 'L' mode for grayscale
    draw = ImageDraw.Draw(img)
    color_value = random.randint(25, 200)  # Random color value for the shape

    # Draw different shapes based on shape_type
    if shape_type == "circle":
        # Random circle: center and radius
        center = (random.randint(15, 209), random.randint(15, 209))  # within bounds
        radius = random.randint(10, min(center[0], center[1], size[0] - center[0], size[1] - center[1]))
        draw.ellipse([center[0] - radius, center[1] - radius,
                      center[0] + radius, center[1] + radius], fill=color_value)
        
    elif shape_type == "rectangle":
        # Random rectangle: top left and bottom right corners
        top = random.randint(5, 199)
        bottom = random.randint(top + 5, 219)  # Ensure bottom is below top
        left = random.randint(5, 199)
        right = random.randint(left + 5, 219)
        draw.rectangle([left, top, right, bottom], fill=color_value)
        
    elif shape_type == "line":
        # Random line: start and end points with thicker line
        start = (random.randint(5, 219), random.randint(5, 219))
        end = (random.randint(5, 219), random.randint(5, 219))
        draw.line([start, end], fill=color_value, width=10)  # Thicker line (width=5)
    
    elif shape_type == "triangle":
        # Random triangle: three points more spaced out
        point1 = (random.randint(5, 219), random.randint(5, 219))
        point2 = (random.randint(5, 219), random.randint(5, 219))
        point3 = (random.randint(5, 219), random.randint(5, 219))
        draw.polygon([point1, point2, point3], fill=color_value)

    # Return the image object
    return img

def save_image(image, filename, folder="generated_images"):
    """Saves the image to a specified folder."""
    image.save(f"{folder}/{filename}", format="PNG")

save_folder = "generated_images"
# Ensure the save folder exists
import os
if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)

# Example usage: generate and save different shape images
for i in range(1250):
    shape = random.choice(["circle", "rectangle", "line", "triangle"])  # Randomly choose shape
    img = generate_image(shape_type=shape)
    save_image(img, f"image_{i:04d}.png")
