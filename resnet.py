import torch
import torch.nn as nn
import os

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet32(nn.Module):
    def __init__(self, num_classes=8+3+1+1):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(6, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 16, 5)
        self.layer2 = self._make_layer(BasicBlock, 32, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 5, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, example_batch, drawing_batch):
        batch_size = example_batch.shape[0]

        # Concatenate example and drawing images along the channel dimension
        x = torch.cat((example_batch, drawing_batch), dim=1)

        x = self.relu(self.bn1(self.conv1(x)))   # [B, 16, 64, 64]
        x = self.layer1(x)                       # [B, 16, 64, 64]
        x = self.layer2(x)                       # [B, 32, 32, 32]
        x = self.layer3(x)                       # [B, 64, 16, 16]
        x = self.avgpool(x)                      # [B, 64, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = torch.sigmoid(x)  # Normalize output to [0, 1]
        
        # Split outputs properly
        control_points = x[:, :8].view(batch_size, 4, 2)
        color = x[:, 8:8 + 3]
        thickness = x[:, 8 + 3] * 10.0 + 0.5  # Scale thickness
        sharpness = x[:, 8 + 4] * 10.0 + 0.5  # Scale sharpness

        return control_points, color, thickness, sharpness
    
    def save_checkpoint(self, optimizer, scheduler, epoch, model_path="checkpoint.pth"):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }

        torch.save(checkpoint, model_path)
        print(f"Checkpoint saved at '{model_path}'")
        
    def load_checkpoint(self, optimizer, scheduler, filename="checkpoint.pth"):
        if os.path.isfile(filename):
            print(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 👈 new
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            return start_epoch
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0
