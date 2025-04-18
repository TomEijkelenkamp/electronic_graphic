from train import train
import os
from resnet import ResNet128
from torch.optim.lr_scheduler import StepLR
import torch

# Datasets
# Isamu136/big-animal-dataset
# lucabaggi/animal-wildlife
# AlvaroVasquezAI/Animal_Image_Classification_Dataset

# Names
name = "bezier_imagenet_resnet_128_full_control"
model_path = os.path.join("models", f"{name}.pth")
image_folder = os.path.join("results", name)

# Model
model = ResNet128()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Train the model
train(
    model=model, 
    dataset="emnist/bymerge", 
    dataset_type="tf",
    image_width=64, 
    image_height=64, 
    batch_size_train=256, 
    batch_size_val=5,
    num_val_images=100,
    optimizer=optimizer,
    scheduler=scheduler, 
    num_epochs=500, 
    start_epoch=model.load_checkpoint(optimizer, scheduler, filename=model_path), 
    num_curves=30, 
    save_and_evaluate_every=250, 
    model_path=model_path, 
    image_folder=image_folder,
    dilate=False,
    colorize=True,
    invert=True,
    rotate=True,
    )