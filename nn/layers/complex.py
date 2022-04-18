from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FromComplex(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()

        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([x.real, x.imag], dim=self.dim)


class ToComplex(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()

        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x_real, x_imag = torch.unbind(x, dim=self.dim)

        return torch.complex(x_real, x_imag)
