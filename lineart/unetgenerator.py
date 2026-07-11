import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim

# -----------------------
# Residual Block
# -----------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.in1(self.conv1(x)))
        h = self.in2(self.conv2(h))
        return x + h


# -----------------------
# Dilated Conv Block
# -----------------------
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, dilation=dilation, padding=dilation)
        self.inorm = nn.InstanceNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.inorm(self.conv(x)))

# -----------------------
# U-Net Generator (1ch Input / 1ch Output)
# -----------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), 
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            ResBlock(64)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            ResBlock(128)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            DilatedConvBlock(128, 256, dilation=2),
            ResBlock(256)
        )
        self.pool3 = nn.MaxPool2d(2)


        self.bottleneck = nn.Sequential(
        DilatedConvBlock(256, 512, dilation=4), 
        ResBlock(512),
        DilatedConvBlock(512, 512, dilation=4), 
        )

        self.up3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Sequential(
        nn.Conv2d(512, 256, 3, padding=1),
        nn.InstanceNorm2d(256),
        nn.ReLU(True),
        ResBlock(256)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            ResBlock(128)
        )
        
        self.up1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Sequential(
            DilatedConvBlock(128, 64, dilation=1),
            ResBlock(64)
        )

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out_conv(d1)