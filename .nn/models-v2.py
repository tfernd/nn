from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod

from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor


def block(features: int, expand: float, kernel_size: int):
    mid_features = round(features * expand)

    conv1 = partial(
        nn.Conv2d,
        bias=False,
        kernel_size=1,
    )
    convk = partial(
        nn.Conv2d,
        bias=False,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
    )

    return nn.Sequential(
        conv1(features, mid_features),
        nn.BatchNorm2d(mid_features),
        nn.ReLU(inplace=True),
        #
        convk(mid_features, mid_features),
        nn.BatchNorm2d(mid_features),
        nn.ReLU(inplace=True),
        #
        conv1(mid_features, features),
    )


class Module(nn.Module, ABC):
    log: list[float]
    optimizer: Optional[torch.optim.Optimizer] = None

    device: torch.device

    def __init__(self):
        super().__init__()

        self.log = []
        self.cpu()

    def cpu(self):
        self.device = torch.device("cpu")
        return self.to(self.device)

    def cuda(self):
        self.device = torch.device("cuda")
        return self.to(self.device)

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        super().to(self.device)

        return self

    def configure_optimizers(self, lr: float):
        self.optimizer = torch.optim.Adamax(self.parameters(), lr=lr)

    @abstractmethod
    def training_step(self, batch, idx: int, /) -> float:
        ...


class FFTPatchSuperResolution(Module):
    def __init__(
        self,
        patch: int,
        scale: int,
        kernel_size: int,
        num_layers: int,
        expand: float,
        num_channels: int = 3,
    ):
        super().__init__()

        self.patch = patch

        self.shape = (num_channels, patch, patch)
        self.features = np.prod(self.shape)

        self.scale = Parameter(torch.zeros(num_layers))
        self.blocks = nn.ModuleList(
            [
                block(
                    self.features,
                    expand,
                    kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.upsample = partial(F.interpolate, scale_factor=scale)

    def patchfy(self, x: Tensor, /) -> Tensor:
        "(N, C, H, W) -> (N, C, p, p, h, w)"

        batch, channels, height, width = x.shape

        p = self.patch
        x = x.unflatten(dim=3, sizes=(width // p, p))
        x = x.unflatten(dim=2, sizes=(height // p, p))
        x = x.permute(0, 1, 3, 5, 2, 4)

        return x

    def unpatchfy(self, x: Tensor, /) -> Tensor:
        "(N, C, p, p, h, w) -> (N, C, H, W)"

        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.flatten(4, 5)
        x = x.flatten(2, 3)

        return x

    def forward(self, x: Tensor, /) -> Tensor:
        batch, channels, height, width = x.shape
        assert x.dtype == torch.uint8

        x = x.to(self.device)

        x = x.div(255)  # normalize to [0, 1]
        x = self.upsample(x)
        x = self.patchfy(x)
        x = x.flatten(1, 3)

        for scale, block in zip(self.scale, self.blocks):
            x = x + scale * block(x)

        x = x.unflatten(1, self.shape)
        x = self.unpatchfy(x)
        x = x.clip(0, 1)
        x = x.mul(255)

        return x

    def training_step(self, batch: tuple[Tensor, Tensor], idx: int, /) -> float:
        self.train()

        x, target = batch
        x, target = x.to(self.device), target.to(self.device)

        assert self.optimizer is not None
        self.optimizer.zero_grad()

        out = self.forward(x)
        loss = F.mse_loss(out, target.float()).div(255)

        loss.backward()
        self.optimizer.step()

        self.log.append(loss.item())

        return loss.item()
