from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

import pytorch_lightning as pl

from .layers import FFTPatchEncoder
from .layers import Patchfy, Unpatchfy
from .layers import Residual
from .layers import ConvBlock

from .losses import VGGPerceptualLoss


class FFTPatchAutoEncoder(pl.LightningModule):
    lr: float = 1e-5

    def __init__(
        self,
        height: int,
        width: int,
        num_channels: int,
        compression: int | float,
        kernel_size: int | tuple[int, int] = 1,
    ):
        super().__init__()

        # parameters
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.compression = compression
        self.kernel_size = kernel_size

        # layers
        self.encoder = FFTPatchEncoder(
            height, width, num_channels, compression, kernel_size
        )
        self.decoder = self.encoder.make_decoder()

        self.latent_size = self.encoder.latent_size

        self.latent_mixer = nn.Identity()
        self.mixer = nn.Identity()

        self.vgg_loss = VGGPerceptualLoss()

    def add_latent_mixer(
        self,
        kernel_size: int = 1,
        expand: int = 4,
        num_layers: int = 1,
    ):
        self.latent_mixer = nn.Sequential(
            *[
                Residual(ConvBlock(self.latent_size, kernel_size, expand))
                for _ in range(num_layers)
            ]
        )

    def add_mixer(
        self,
        patch_size: int,
        kernel_size: int = 1,
        expand: int = 4,
        num_layers: int = 1,
    ):
        shape = (self.num_channels, patch_size, patch_size)
        in_channels = np.prod(shape)

        self.mixer = nn.Sequential(
            *[
                Residual(
                    Patchfy(patch_size),
                    nn.Flatten(1, 3),
                    ConvBlock(in_channels, kernel_size, expand),
                    nn.Unflatten(1, shape),
                    Unpatchfy(patch_size),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, img: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, channels, height, width = img.shape
        assert img.dtype == torch.uint8

        x = out = img.div(255)  # normalize to [0, 1]

        z = self.latent_mixer(self.encoder(x))
        out = self.mixer(self.decoder(z)).clip(0, 1)

        # loss = out.sub(x).pow(2).mean()
        loss = self.vgg_loss(x, out)

        out = out.detach().mul(255)

        return out, z, loss

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        out, z, loss = self.forward(batch)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer
