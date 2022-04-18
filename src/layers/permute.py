from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Permute(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()

        self.dims = dims

    def forward(self, x: Tensor, /) -> Tensor:
        return x.permute(*self.dims)
