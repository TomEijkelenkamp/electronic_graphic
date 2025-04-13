from train import run
import os
from torch.utils.data import DataLoader
from datasets import load_dataset
from resnet import ResNet128
from dataset import HuggingfaceDataset
from torch.optim.lr_scheduler import StepLR
import torch

name = "bezier_imagenet_resnet_128_thickness_sharpness"
image_folder = os.path.join("results", name)
model_path = os.path.join("models", f"{name}.pth")

print("Load dataset...")
# Load MNIST dataset from Hugging Face
# dataset = load_dataset("ylecun/mnist")
# dataset = load_dataset("zh-plus/tiny-imagenet")
dataset = load_dataset("benjamin-paine/imagenet-1k-64x64")

batch_size_train = 256
batch_size_val = 10
height, width = 64, 64  # Canvas size

number_of_curves = 12  # Number of curves to draw

# Initialize the dataset and dataloader
# dataset = ImageDataset(dataset_path=".//data//character_images//", image_width=height, image_height=width)
train_dataset = HuggingfaceDataset(dataset, "train", image_width=height, image_height=width)
val_dataset = HuggingfaceDataset(dataset, "validation", image_width=height, image_height=width)
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)

# Initialize the model
print("Initialize model...")
model = ResNet128()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

# Load the model checkpoint if it exists
start_epoch = model.load_checkpoint(optimizer, scheduler, filename=model_path)

# 1. Manually set learning rate in optimizer
for param_group in optimizer.param_groups:
    param_group['lr'] = 0.0001

# 2. Re-initialize the scheduler with the updated optimizer
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# 3. (Optional but clean) Step the scheduler if you need to "sync" it with the current epoch
# for example, if start_epoch=3, call scheduler.step() 3 times:
for _ in range(start_epoch):
    scheduler.step()

# Train the model
run(model, train_loader, val_loader, optimizer, scheduler, num_epochs=500, start_epoch=start_epoch, num_curves=number_of_curves, save_and_evaluate_every=1000, model_path=model_path, image_folder=image_folder)