
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
import random
from datasets import get_dataset_split_names
import torch
import kornia.morphology as km


def get_train_dataset_loader(dataset, image_width, image_height, batch_size):
    print(f"Loading dataset {dataset}...")
    train_split = get_dataset_split_names(dataset)[0]
    train_dataset = HuggingfaceDataset(load_dataset(dataset, split=train_split), image_width=image_width, image_height=image_height)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader

def get_val_dataset_loader(dataset, num_images, image_width, image_height, batch_size):
    val_split = get_dataset_split_names(dataset)[1]  # Get the second split (usually "validation")
    val_size = len(load_dataset(dataset, split=val_split))
    start = random.randint(0, val_size - num_images)
    val_dataset = load_dataset("benjamin-paine/imagenet-1k-64x64", split=f"{val_split}[{start}:{start + num_images}]")
    val_dataset_object = HuggingfaceDataset(val_dataset, image_width=image_width, image_height=image_height)
    val_loader = DataLoader(val_dataset_object, batch_size=batch_size, shuffle=True)    
    return val_loader

class HuggingfaceDataset(Dataset):
    def __init__(self, dataset, image_width, image_height, add_color=False, invert=False):
        self.data = dataset
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure 3 channels
            transforms.Resize((self.image_width, self.image_height)),
            transforms.ToTensor(),  # Converts HWC [0-255] PIL image to CHW [0.0-1.0] tensor
        ])
        self.add_color = add_color
        self.invert = invert
        self.epoch = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.data)

    def dilate(self, img):
        kernel_size = 25 - self.epoch * 5  # Decrease kernel size over time
        if kernel_size > 0:
            kernel = torch.ones((kernel_size, kernel_size), device=self.device)
            img = img.unsqueeze(0).to(self.device)
            img = km.dilation(img, kernel)
            img = img.squeeze(0)  # (C, H, W)
        return img

    def __getitem__(self, idx):
        img = self.data[idx]["image"]  # PIL image
        img = self.transform(img)
        img = self.dilate(img)
        return img

class ImageDataset(Dataset):
    def __init__(self, dataset_path, image_width, image_height):
        self.dataset_path = dataset_path
        self.image_width = image_width
        self.image_height = image_height
        self.image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')])
        self.epoch = 0

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_path, self.image_files[idx])
        img = self.load_image(image_path)

        # Blend factor increases over time (clipped to 1.0 max)
        blend_factor = min(self.epoch * 0.01, 1.0)

        # Original grayscale expanded to RGB
        grayscale_img = img.repeat(3, 1, 1)

        # Inverted grayscale image
        inverted_img = 1 - grayscale_img

        # Random color tint (same as before)
        rand_color = torch.rand(3) * 0.75 + 0.25
        color_tinted_img = inverted_img * rand_color.view(3, 1, 1)

        # Blend between grayscale and color-tinted versions
        final_img = (1 - blend_factor) * inverted_img + blend_factor * color_tinted_img

        # Invert the image
        final_img = 1 - final_img

        return final_img

    def load_image(self, image_path):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.Resize((self.image_width, self.image_height)),  # Resize
            transforms.ToTensor()  # Convert to tensor, auto normalizes to [0,1]
        ])
        img = transform(Image.open(image_path))
        return img