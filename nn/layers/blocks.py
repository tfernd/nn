from __future__ import annotations

import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, kernel_size: int = 1, expand: int = 4):
        # parameters
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.expand = expand

        layers = [
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                in_channels * expand,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            nn.BatchNorm2d(in_channels * expand),
            nn.Conv2d(in_channels * expand, in_channels, kernel_size=1),
        ]
        super().__init__(*layers)
