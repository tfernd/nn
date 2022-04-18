from __future__ import annotations
from typing import Optional, Literal

import torch
import torch.nn as nn
from torch import Tensor


class _BaseFFT(nn.Module):
    _fft: torch.fft.fftn | torch.fft.ifftn | torch.fft.rfftn | torch.fft.irfftn

    def __init__(
        self,
        dim: int | tuple[int, ...] = -1,
        size: Optional[int | tuple[int, ...]] = None,
        norm: Literal["forward", "backward", "ortho"] = "backward",
    ):
        super().__init__()

        self.dim = dim if isinstance(dim, tuple) else (dim,)
        self.size = size
        self.norm = norm

        if self.size is not None:
            self.size = self.size if isinstance(self.size, tuple) else (self.size,)

            assert len(self.size) == len(self.dim)

    def forward(self, x: Tensor) -> Tensor:
        return self._fft(x, s=self.size, dim=self.dim, norm=self.norm)


class rFFT(_BaseFFT):
    _fft = torch.fft.rfftn


class irFFT(_BaseFFT):
    _fft = torch.fft.irfftn


class FFT(_BaseFFT):
    _fft = torch.fft.fftn


class iFFT(_BaseFFT):
    _fft = torch.fft.ifftn
