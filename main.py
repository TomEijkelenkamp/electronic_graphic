from train import run
import os
from torch.utils.data import DataLoader
from datasets import load_dataset
from resnet import ResNet32
from dataset import HuggingfaceDataset
from torch.optim.lr_scheduler import StepLR
import torch

name = "bezier_imagenet_resnet_v1"
image_folder = os.path.join("results", name)
model_path = os.path.join("models", f"{name}.pth")

# Load MNIST dataset from Hugging Face
# dataset = load_dataset("ylecun/mnist")
dataset = load_dataset("zh-plus/tiny-imagenet")

batch_size = 192
height, width = 64, 64  # Canvas size

# Initialize the dataset and dataloader
# dataset = ImageDataset(dataset_path=".//data//character_images//", image_width=height, image_height=width)
train_dataset = HuggingfaceDataset(dataset, "train", image_width=height, image_height=width)
val_dataset = HuggingfaceDataset(dataset, "valid", image_width=height, image_height=width)
train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

# Initialize the model
model = ResNet32()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# Load the model checkpoint if it exists
start_epoch = model.load_checkpoint(optimizer, scheduler, filename=model_path)

# Train the model
run(model, train_loader, val_loader, optimizer, scheduler, num_epochs=500, start_epoch=start_epoch, num_curves=8, model_path=model_path, image_folder=image_folder)