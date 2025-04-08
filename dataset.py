
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class HuggingfaceDataset(Dataset):
    def __init__(self, dataset, dataset_split, image_width, image_height, add_color=False, invert=False):
        self.data = dataset[dataset_split]
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.Resize((self.image_width, self.image_height)),  # Resize
            transforms.ToTensor()  # Convert to tensor, auto normalizes to [0,1]
        ])
        self.add_color = add_color
        self.invert = invert


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"]  # PIL image
        img = self.transform(img)

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

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