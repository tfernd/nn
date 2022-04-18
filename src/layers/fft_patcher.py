from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from .fft import rFFT, irFFT
from .patcher import Patchfy, Unpatchfy
from .complex import FromComplex, ToComplex


class _FFTPatcherBase(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        num_channels: int,
        compression: int | float = 1.0,
        kernel_size: int | tuple[int, int] = 1,
        *,
        scale: float = 0.0,
    ):
        super().__init__()

        # TODO make into own utils function
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        for (i, k) in enumerate(kernel_size):
            assert k % 2 == 1, f"kernel_size[{i}] ({k}) must be odd"
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # parameters
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.compression = compression
        self.kernel_size = kernel_size
        self.scale = scale

        # variables
        self.padding = padding
        self.fft_shape = (num_channels, height, width // 2 + 1, 2)
        self.fft_features = np.prod(self.fft_shape)
        self.latent_size = (
            compression
            if isinstance(compression, int)
            else round(compression * self.fft_features)
        )
        assert self.latent_size >= 1, f"compression ({compression}) is too low."

        # Fourier wave-vectors sorting
        kh = torch.fft.fftfreq(height)
        kw = torch.fft.fftfreq(width)[: width // 2 + 1]

        Kh, Kw = torch.meshgrid(kh, kw, indexing="ij")
        K = torch.sqrt(Kh**2 + Kw**2)

        # make into shape (C, ph, pw//2+1, 2)
        K = K.view(1, height, width // 2 + 1, 1)
        K = K.tile(num_channels, 1, 1, 2)

        # sort from smallest to largest frequency
        idx = K.flatten().argsort()

        # select values to pass by the fourier filter
        kh, kw = kernel_size
        W = torch.randn(self.latent_size, self.fft_features, kh, kw)
        W *= scale

        # initial weigths
        for (n, i) in enumerate(idx[: self.latent_size]):
            W[n, i, kh // 2, kw // 2] = 1
        self._W = W

        self.__post_init__()

    def __post_init__(self):
        raise NotImplementedError


class FFTPatchEncoder(_FFTPatcherBase):
    def __post_init__(self):
        self.patchfy = Patchfy(self.height, self.width)
        self.rfft = rFFT(dim=(2, 3), size=(self.height, self.width))
        self.from_complex = FromComplex(dim=4)
        self.flatten = nn.Flatten(1, 4)

        self.conv = nn.Conv2d(
            self.fft_features,
            self.latent_size,
            self.kernel_size,
            bias=False,
            padding=self.padding,
        )
        self.conv.weight.data = self._W

    def make_decoder(self):
        return FFTPatchDecoder(
            self.height,
            self.width,
            self.num_channels,
            self.compression,
            self.kernel_size,
            scale=self.scale,
        )

    def forward(self, x: Tensor, /) -> Tensor:
        x = self.patchfy(x)
        x = self.rfft(x)
        x = self.from_complex(x)
        x = self.flatten(x)
        x = self.conv(x)

        return x


class FFTPatchDecoder(_FFTPatcherBase):
    def __post_init__(self):
        self.conv = nn.Conv2d(
            self.latent_size,
            self.fft_features,
            self.kernel_size,
            bias=False,
            padding=self.padding,
        )
        self.conv.weight.data = self._W.transpose(0, 1)

        self.unflatten = nn.Unflatten(1, self.fft_shape)
        self.to_complex = ToComplex(dim=4)
        self.irfft = irFFT(dim=(2, 3), size=(self.height, self.width))
        self.unpatchfy = Unpatchfy(self.height, self.width)

    def make_encoder(self):
        return FFTPatchEncoder(
            self.height,
            self.width,
            self.num_channels,
            self.compression,
            self.kernel_size,
            scale=self.scale,
        )

    def forward(self, x: Tensor, /) -> Tensor:
        x = self.conv(x)
        x = self.unflatten(x)
        x = self.to_complex(x)
        x = self.irfft(x)
        x = self.unpatchfy(x)

        return x
