import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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
        target_tensor = torch.tensor(target_image, dtype=torch.float32).view(1, -1)  # Flatten the image
        return target_tensor

    def load_image(self, image_path):
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((self.image_width, self.image_height))  # Ensure correct size
        img = np.array(img) / 255.0  # Normalize to [0,1]
        return img

class DrawMLP(nn.Module):
    def __init__(self, image_width, image_height, hidden_size):
        super(DrawMLP, self).__init__()
        # Input size: 2 * image_width * image_height (for example and drawing)
        self.fc1 = nn.Linear(2 * image_width * image_height, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4)  # x1, y1, x2, y2

    def forward(self, example_batch, drawning_batch):
        # Flatten example_batch and drawning_batch
        example_batch_flat = example_batch.view(example_batch.size(0), -1)  # Flatten to (batch_size, features)
        drawning_batch_flat = drawning_batch.view(drawning_batch.size(0), -1)  # Flatten to (batch_size, features)
        
        # Concatenate the example and drawn image batches along the feature dimension
        x = torch.cat((example_batch_flat, drawning_batch_flat), dim=1)
        
        # Pass through the network layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Normalize output (0-1)

        return x
    
def draw_line(drawning_batch, predicted_coords):
    batch_size = predicted_coords.size(0)
    
    # Loop over each image in the batch
    for i in range(batch_size):
        # Get the predicted coordinates for this image
        x1, y1, x2, y2 = predicted_coords[i]

        # Convert the coordinates to integers for indexing
        # Check the shape of drawning_batch and adapt the indexing accordingly
        if len(drawning_batch.shape) == 4:  # If batch_size, channels, height, width
            height, width = drawning_batch.shape[2], drawning_batch.shape[3]
            image_tensor = drawning_batch[i, 0]  # Get the single channel image
        elif len(drawning_batch.shape) == 3:  # If batch_size, height, width
            height, width = drawning_batch.shape[1], drawning_batch.shape[2]
            image_tensor = drawning_batch[i]  # Get the image directly (no channel dimension)
        else:
            raise ValueError("Unexpected shape for drawning_batch.")

        # Scale the predicted coordinates to match image dimensions
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), \
                           int(x2 * width), int(y2 * height)

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < width and 0 <= y1 < height:
                # Draw black pixel (since it's a canvas, we assume 0 is black)
                image_tensor[y1, x1] = 0  # Update the pixel to black

            if x1 == x2 and y1 == y2:
                break

            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    return drawning_batch

# Training loop with fixed denormalization and model outputs
def train_network(device, model, optimizer, criterion, num_epochs, dataloader, max_actions_per_image, save_interval=500):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to the appropriate device
            example_batch = batch.to(device)
            print(f"Batch {batch_idx}/{len(dataloader)}: {example_batch.shape}")

            # Initialize drawn image (white canvas) for the batch, requires gradients
            drawning_batch = torch.ones_like(batch, dtype=torch.float32, requires_grad=True).to(device)
            print(f"Drawning batch initialized: {drawning_batch.shape}")

            for action in range(max_actions_per_image):
                print(f"Action {action}/{max_actions_per_image}")
                # Let the model predict the coordinates
                predicted_coords = model(example_batch, drawning_batch)
                print(f"Predicted coordinates: {predicted_coords.shape}")

                # Draw the line on the image canvas
                draw_line(drawning_batch, predicted_coords)
                print(f"Drawn image shape: {drawning_batch.shape}")

                # Compute loss
                loss = criterion(example_batch, drawning_batch)
                print(f"Loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward(retain_graph=True)  # Retain the graph for subsequent backward passes
                optimizer.step()  # Update weights

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, Batch {batch_idx}/{len(dataloader)}")

        # Save the model every `save_interval` epochs
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            model_path = os.path.join(model_save_folder, f"draw_mlp_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved: {model_path}")

            # Save drawn images for all images in the batch (without showing them)
            for i in range(example_batch.size(0)):  # Loop through each image in the batch
                drawn_image_path = os.path.join(drawings_save_folder, f"drawn_image_epoch_{epoch}_batch_{batch_idx}_image_{i}.png")
                plt.imsave(drawn_image_path, drawning_batch[i].cpu().numpy(), cmap='gray')
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
criterion = nn.MSELoss()

# Call the train function with DataLoader
train_network(device, model, optimizer, criterion, num_epochs=5000, dataloader=dataloader, max_actions_per_image=20)
