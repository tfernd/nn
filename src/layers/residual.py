from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor


class Residual(nn.Sequential):
    def __init__(self, *layers: nn.Module, scale: float = 0.0):
        super().__init__(*layers)

        self.scale = Parameter(torch.tensor(scale))

    def forward(self, x: Tensor, /) -> Tensor:
        return x + self.scale * super().forward(x)
