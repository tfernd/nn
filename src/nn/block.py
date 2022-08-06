from __future__ import annotations
from typing import Optional

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        ratio: float = 1 / 32,
    ) -> None:
        super().__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels = out_channels or in_channels
        self.ratio = ratio
        self.mid_channels = mid_channels = math.ceil(in_channels * ratio)

        assert ratio > 0
        assert in_channels >= 1
        assert out_channels >= 1

        # layers
        self.scale = Parameter(torch.ones(in_channels))
        self.shift = Parameter(torch.zeros(in_channels))

        self.layer = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, out_channels),
        )

        # squeeze-excitation
        self.se = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, out_channels),
            nn.Sigmoid(),
        )

        self.skip = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        # normalize
        xn = x * self.scale + self.shift

        # extract mean features (if available)
        dims = tuple(range(1, x.ndim - 1))
        xmean = xn.mean(dims, keepdim=True) if len(dims) > 0 else xn

        out = self.skip(x) + self.layer(xn) * self.se(xmean)

        return out

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        args = ", ".join(
            [
                f"in_channels={self.in_channels}",
                f"out_channels={self.out_channels}",
                f"ratio={self.ratio}",
            ]
        )

        return f"{name}({args})"
