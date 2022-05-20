from __future__ import annotations
from typing import Optional

from functools import partial

import math
import random
from more_itertools import padded

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor

from .base import Base


def rfft(x: Tensor, /) -> Tensor:
    f = torch.fft.rfft2(x, dim=(2, 3))
    f = torch.cat([f.real, f.imag], dim=1)

    return f


def irfft(f: Tensor, /, *, size: Optional[tuple[int, int]] = None) -> Tensor:
    fr, fi = f.chunk(2, dim=1)
    f = torch.complex(fr, fi)

    x = torch.fft.irfft2(f, dim=(2, 3), s=size)

    return x


class Block(nn.Module):
    def __init__(
        self,
        channels: int,
        expand: float,
        kernel_size: int = 3,
    ):
        super().__init__()

        assert kernel_size % 2 == 1

        mid = round(channels * expand)

        kwargs = dict(kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, mid, **kwargs),
            nn.BatchNorm2d(mid),
            nn.ReLU(),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, channels, **kwargs),
            nn.BatchNorm2d(channels),
        )

        kwargs = dict(kernel_size=1, bias=False)
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(2 * channels),
            nn.Conv2d(2 * channels, 2 * mid, **kwargs),
            nn.BatchNorm2d(2 * mid),
            nn.ReLU(),
            nn.BatchNorm2d(2 * mid),
            nn.Conv2d(2 * mid, 2 * channels, **kwargs),
            nn.BatchNorm2d(2 * channels),
        )

        self.gamma = Parameter(torch.zeros(2))

    def forward(self, x: Tensor, /) -> Tensor:
        batch, channels, height, width = x.shape

        yc = self.conv(x)
        yfft = irfft(self.mlp(rfft(x)), size=(height, width))

        g1, g2 = self.gamma
        out = x + g1 * yc + g2 * yfft

        return out


class FFTModel(Base):
    def __init__(
        self,
        scale: int,
        num_steps: int,
        hidden_size: int = 64,
        num_layers: int = 4,
        expand: float = 2,
        kernel_size: int = 3,
        num_channel: int = 3,
    ):
        super().__init__()

        # parameters
        self.scale = scale
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.expand = expand
        self.kernel_size = kernel_size
        self.num_channel = num_channel

        # variables
        self.sub_scale = scale ** (1 / num_steps)

        # methods
        self.upscale = partial(F.interpolate, scale_factor=self.sub_scale)

        # layers
        self.emb = nn.Conv2d(num_channel, hidden_size, kernel_size=1)

        args = (hidden_size, expand, kernel_size)
        self.blocks = nn.Sequential(*[Block(*args) for _ in range(num_layers)])

        self.dec = nn.Conv2d(hidden_size, num_channel, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x: Tensor, /, *, repeat: int = 1) -> Tensor:
        assert x.dtype == torch.uint8

        if repeat > 1:
            x = self.forward(x, repeat=repeat - 1)

        x = x.to(self.device)
        xn = x.div(255)  # normalize to [0, 1]

        xs = self.upscale(xn)
        xe = self.emb(xs)
        xb = self.blocks(xe)

        out = xs + self.dec(xb) * self.gamma
        out = out.clamp(0, 1).mul(255)

        return out

    def train_step(self, target: Tensor, idx: int, /) -> float:
        self.train()

        assert self.optimizer is not None
        self.optimizer.zero_grad()

        repeat = random.randint(1, self.num_steps + 1)
        repeat = 1

        height, width = target.shape[2:]
        for _ in range(repeat):
            height = math.ceil(height / self.sub_scale)
            width = math.ceil(width / self.sub_scale)

        target = target.to(self.device)
        data = F.interpolate(target, size=(height, width))

        out = self.forward(data, repeat=repeat)

        ft = rfft(target.div(255))
        fo = rfft(out.div(255))
        loss = torch.mean(torch.abs(ft - fo).pow(2))
        # loss = F.mse_loss(out, target.float())
        # loss /= 255

        loss_item = loss.item()

        self.log.append(loss_item)

        loss.backward()
        self.optimizer.step()

        return loss_item
