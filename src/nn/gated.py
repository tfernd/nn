from __future__ import annotations
from typing import Optional

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor


class Gated(nn.Module):
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

        # layers
        self.scale = Parameter(torch.ones(in_channels))
        self.shift = Parameter(torch.zeros(in_channels))

        self.gate = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.GLU(dim=-1),
        )

        # squeeze and excite
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

    def forward(self, x: Tensor, /) -> Tensor:
        # normalize
        xn = x * self.scale + self.shift

        # extract mean features (if available)
        dims = tuple(range(1, x.ndim - 1))
        xmean = xn.mean(dims, keepdim=True) if len(dims) > 0 else xn

        # gated and squeeze-excite features
        out = self.gate(xn) * self.se(xmean)

        # residual connection
        out = self.skip(x) + out

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
