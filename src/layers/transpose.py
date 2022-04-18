from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int, /):
        super().__init__()

        assert dim0 != dim1, f"dim0 ({dim0}) and dim1 ({dim1}) must be different"

        self.dim = (dim0, dim1)

    def forward(self, x: Tensor, /) -> Tensor:
        return x.transpose(*self.dim)
