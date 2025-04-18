
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
import random
from datasets import get_dataset_split_names
import torch
import kornia.morphology as km
import os
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf


class MyDataset(Dataset):
    def __init__(self, image_width, image_height, dilate=False, colorize=False, invert=False, rotate=False):
        self.image_width = image_width
        self.image_height = image_height
        self.epoch = 0
        self.dilate = dilate
        self.colorize = colorize
        self.invert = invert
        self.rotate = rotate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = self.build_transform()

    def build_transform(self):
        steps = [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((self.image_width, self.image_height)),
            transforms.ToTensor(),
        ]

        if self.invert:
            # Apply random inversion with 50% probability
            steps.append(transforms.Lambda(lambda t: 1 - t if random.random() < 0.5 else t))
            
        if self.colorize:
            # Apply random colorization to either the white or black areas of the image
            steps.append(transforms.Lambda(self.random_colorize))

        if self.dilate:
            # Apply dilation with decreasing kernel size over epochs
            steps.append(transforms.Lambda(self.apply_dilate))

        if self.rotate:
            # Random rotation of the image by 0, 90, 180, or 270 degrees
            steps.append(transforms.Lambda(lambda t: t.rot90(k=random.randint(0, 3), dims=[1, 2])))

        return transforms.Compose(steps)

    def random_colorize(self, img):
        """
        Converts grayscale tensor image (3, H, W) to RGB (3, H, W),
        by multiplying either the white or black areas by a random color.
        """

        # Pick random color
        color = torch.rand(3, 1, 1) * 0.75 + 0.25  # avoid too-dark or too-light

        # Randomly choose to tint whites (1s) or blacks (0s)
        if random.random() < 0.5:
            # Colorize whites: img stays as-is, color applied to value regions
            img = img * color
        else:
            # Colorize blacks: invert, multiply, then re-invert
            img = (1 - img) * color
            img = 1 - img

        return img
    
    def disk_kernel(self, radius):
        """Create a disk-shaped binary kernel for morphological ops."""
        diameter = 2 * radius + 1
        y, x = torch.meshgrid(
            torch.arange(diameter, device=self.device),
            torch.arange(diameter, device=self.device),
            indexing="ij"  # for PyTorch 1.10+
        )
        center = radius
        dist = ((x - center)**2 + (y - center)**2).sqrt()
        kernel = (dist <= radius).float()  # 1.0 inside the disk, 0.0 outside
        return kernel

    def apply_dilate(self, img):
        kernel_radius = 12 - self.epoch * 2  # Decrease kernel size over time
        if kernel_radius > 0:
            kernel = self.disk_kernel(kernel_radius)
            img = img.unsqueeze(0).to(self.device)
            img = km.dilation(img, kernel)
            img = img.squeeze(0)  # (C, H, W)
        return img

    @staticmethod
    def get_split_index(split):
        if split == "train":
            return 0
        elif split == "val":
            return 1
        elif split == "test":
            return 2
        else:
            raise ValueError("Invalid split name. Use 'train', 'val', or 'test'.")

    @staticmethod
    def get_dataset_loader(path, type, split, image_width, image_height, batch_size, num_images=-1, dilate=False, colorize=False, invert=False, rotate=False):
        print(f"Loading dataset {path}...")

        if type == "huggingface":
            dataset_object = HuggingfaceDataset(path, image_width=image_width, image_height=image_height, split=split, num_images=num_images, dilate=dilate, colorize=colorize, invert=invert, rotate=rotate)
        elif type == "image":
            dataset_object = ImageDataset(path, image_width=image_width, image_height=image_height, split=split, num_images=num_images,  dilate=dilate, colorize=colorize, invert=invert, rotate=rotate)
        elif type == "tf": 
            dataset_object = TFDatasetWrapper(path, image_width=image_width, image_height=image_height, split=split, num_images=num_images, dilate=dilate, colorize=colorize, invert=invert, rotate=rotate)
        else:
            raise ValueError("Unsupported dataset type.")

        return DataLoader(dataset_object, batch_size=batch_size, shuffle=True)
    
class HuggingfaceDataset(MyDataset):
    def __init__(self, path, image_width, image_height, split, num_images=-1, dilate=False, colorize=False, invert=False, rotate=False):
        super().__init__(image_width, image_height, dilate=dilate, colorize=colorize, invert=invert, rotate=rotate)
        split = get_dataset_split_names(path)[MyDataset.get_split_index(split)]
        if num_images > 0:
            split_size = len(load_dataset(path, split=split))
            start = random.randint(0, split_size - num_images)
            split = f"{split}[{start}:{start + num_images}]"
        self.data = load_dataset(path, split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"]  # PIL image
        img = self.transform(img)
        return img

class ImageDataset(MyDataset):
    def __init__(self, path, image_width, image_height, split, num_images=-1, dilate=False, colorize=False, invert=False, rotate=False):
        super().__init__(image_width, image_height, dilate=dilate, colorize=colorize, invert=invert, rotate=rotate)
        if os.path.isdir(path):
            subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            if len(subfolders) > 0:
                self.dataset_path = os.path.join(path, subfolders[MyDataset.get_split_index(split)])
            else:
                self.dataset_path = path
        else:
            raise ValueError("Path is not a directory.")
        
        if num_images > 0:
            image_files = sorted([f for f in os.listdir(self.dataset_path) if f.endswith('.png')])
            start = random.randint(0, len(image_files) - num_images)
            self.image_files = image_files[start:start + num_images]
        else:
            self.image_files = sorted([f for f in os.listdir(self.dataset_path) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_path, self.image_files[idx])
        img = Image.open(image_path)
        img = self.transform(img)
        return img

class TFDatasetWrapper(MyDataset):
    def __init__(self, path, image_width, image_height, split, num_images=-1, dilate=False, colorize=False, invert=False, rotate=False):
        super().__init__(image_width, image_height, dilate=dilate, colorize=colorize, invert=invert, rotate=rotate)
        builder = tfds.builder(path)

        builder.download_and_prepare()

        # Choose a split by index or name
        split_names = list(builder.info.splits.keys())  # e.g., ['train', 'test', 'validation']
        split_name = split_names[MyDataset.get_split_index(split)]  # or just 'train', 'test', etc.

        if num_images > 0:
            # Get the number of examples in that split
            split_size = builder.info.splits[split_name].num_examples
            print(f"Split size for {split_name}: {split_size}")
            start = random.randint(0, split_size - num_images)
            split_name = f"{split_name}[{start}:{start + num_images}]"
            print(f"Loading {num_images} images from {split_name}")

        tf_dataset = tfds.load(path, split=split_name, as_supervised=False)
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        self.tf_dataset = list(tf_dataset)  # Convert to list to support indexing
        
    def __len__(self):
        return len(self.tf_dataset)

    def __getitem__(self, idx):
        example = self.tf_dataset[idx]
        image = example['image'].numpy()  # Convert to numpy array        
        image = np.squeeze(image)  # removes dimensions of size 1
        image = Image.fromarray(image) # Convert to PIL image
        image = self.transform(image)
        return image
    
class MixedDataset(MyDataset):
    def __init__(self, datasets, dataset_types, image_width, image_height, dilate=False, colorize=False, invert=False, rotate=False):
        super().__init__(image_width, image_height, dilate=dilate, colorize=colorize, invert=invert, rotate=rotate)
        self.datasets = datasets
        self.dataset_weights = [1.0 / len(datasets)] * len(datasets)  # Equal weights for each dataset
        self.dataset_types = dataset_types

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for dataset_idx, dataset in enumerate(self.datasets):
            if idx < len(dataset):
                break
            idx -= len(dataset)
        dataset = self.datasets[dataset_idx]
        dataset_type = self.dataset_types[dataset_idx]
        
        if dataset_type == "huggingface":
            return dataset[idx]["image"]
        elif dataset_type == "image":
            image_path = os.path.join(self.dataset_path, self.image_files[idx])
            img = Image.open(image_path)
            img = self.transform(img)
            return img
            return dataset[idx]
        elif dataset_type == "tf":
            example = dataset[idx]
            image = example['image'].numpy()  # Convert to numpy array        
            image = np.squeeze(image)  # removes dimensions of size 1
            image = Image.fromarray(image) # Convert to PIL image
            image = self.transform(image)
            return image
        
         