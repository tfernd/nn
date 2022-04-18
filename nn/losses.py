from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torchvision


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = torchvision.models.vgg16().eval()
        for p in vgg.parameters():
            p.requires_grad = False

        blocks = [
            vgg.features[:4],
            vgg.features[4:9],
            vgg.features[9:16],
            vgg.features[16:23],
        ]
        self.blocks = torch.nn.ModuleList(blocks)
        self.interpolate = torch.nn.Upsample(
            size=(224, 244), mode="bilinear", align_corners=False
        )

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x = self.interpolate(x)
        y = self.interpolate(y)

        loss = F.mse_loss(x, y)
        for (i, block) in enumerate(self.blocks):
            x, y = block(x), block(y)

            loss += F.l1_loss(x, y)

        return loss
