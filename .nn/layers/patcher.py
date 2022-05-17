from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor


# TODO make 1d version
class _Patcher(nn.Module):
    def __init__(self, height: int, width: Optional[int] = None):
        super().__init__()

        if width is None:
            width = height

        self.height = height
        self.width = width

    def patchfy(self, x: Tensor) -> Tensor:
        """(N, C, H, W) -> (N, C, ph, pw, H//ph, W//pw)"""

        assert x.ndim == 4, "Input must be 4D"

        # (N, C, H/ph, ph, W/pw, pw)
        x = x.unflatten(3, (-1, self.width))
        x = x.unflatten(2, (-1, self.height))

        # (N, C, ph, pw, H/ph, W/pw)
        x = x.permute(0, 1, 3, 5, 2, 4)

        return x

    def unpatchfy(self, x: Tensor) -> Tensor:
        """(N, C, ph, pw, H//ph, W//pw) -> (N, C, H, W)"""

        assert x.ndim == 6, "Input must be 6D"

        assert x.shape[2] == self.height, "Height mismatch"
        assert x.shape[3] == self.width, "Width mismatch"

        # (N, C, H/ph, ph, W/pw, pw)
        x = x.permute(0, 1, 4, 2, 5, 3)

        # (N, C, H, W)
        x = x.flatten(4, 5)
        x = x.flatten(2, 3)

        return x


class Patchfy(_Patcher):
    def forward(self, x: Tensor) -> Tensor:
        return self.patchfy(x)


class Unpatchfy(_Patcher):
    def forward(self, x: Tensor) -> Tensor:
        return self.unpatchfy(x)
