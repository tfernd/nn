from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Reshape(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()

        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(*self.shape)
