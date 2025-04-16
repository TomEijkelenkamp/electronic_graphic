import torch
import torch.nn as nn
import os
from tqdm import tqdm

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)
        return out

class ResNet128(nn.Module):
    def __init__(self, num_classes=8 + 3 + 1 + 1):
        print("Initialize model...")
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(6, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act = nn.SiLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 16, 3)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes)
        )

        # Apply Xavier initialization to the entire model
        self.apply(self.init_weights)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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

        x = torch.cat((example_batch, drawing_batch), dim=1)  # [B, 6, 64, 64]
        x = self.act(self.bn1(self.conv1(x)))                 # [B, 32, 64, 64]
        x = self.layer1(x)                                    # [B, 32, 64, 64]
        x = self.layer2(x)                                    # [B, 64, 32, 32]
        x = self.layer3(x)                                    # [B, 128, 16, 16]
        x = self.avgpool(x)                                   # [B, 128, 1, 1]
        x = torch.flatten(x, 1)                               # [B, 128]
        x = self.fc(x)                                        # [B, num_classes]

        control_points = x[:, :8].view(batch_size, 4, 2)
        color = torch.sigmoid(x[:, 8:8 + 3])
        thickness = torch.nn.functional.softplus(x[:, 11:12], beta=50)  # [B, 1]
        sharpness = torch.nn.functional.softplus(x[:, 12:13], beta=50)  # [B, 1]         

        return control_points, color, thickness, sharpness

    def init_weights(self, m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def check_for_invalid_params(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if torch.isnan(param).any():
                    tqdm.write(f"NaN detected in parameter: {name}")
                    return True
                if torch.isinf(param).any():
                    tqdm.write(f"Inf detected in parameter: {name}")
                    return True
        return False

    def save_checkpoint(self, optimizer, scheduler, epoch, avg_loss, model_path="checkpoint.pth"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, model_path)
        tqdm.write(f"✅ Checkpoint at epoch {epoch} with average loss {avg_loss:.4f} saved at '{model_path}'")

    def load_checkpoint(self, optimizer, scheduler, filename="checkpoint.pth"):
        if os.path.isfile(filename):
            print(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, learning rate: {scheduler.get_last_lr()}")
            self.check_for_invalid_params()
            return start_epoch
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0
