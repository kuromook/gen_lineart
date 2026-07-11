"""Small CNN used by dataset/source gate checkpoints."""

import torch.nn as nn
import torch.nn.functional as F


class GateCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)
