from __future__ import annotations
from typing import Optional

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor

from .base import Base


def block(in_channels: int, expand: int | float):
    out_channels = round(in_channels * expand)

    return nn.Sequential(
        nn.LayerNorm(in_channels),
        nn.Linear(in_channels, out_channels, bias=False),
        nn.LayerNorm(out_channels),
        nn.GELU(),
        nn.Linear(out_channels, in_channels, bias=False),
        nn.LayerNorm(in_channels),
    )


class FFTPatch(Base):
    def __init__(
        self,
        *,
        size: int,
        emb_size: int,
        expand: int | float = 2,
        num_layers: int = 1,
        num_channels: int = 3,
    ):
        super().__init__()

        # parameters
        self.size = size
        self.emb_size = emb_size
        self.expand = expand
        self.num_layers = num_layers
        self.num_channels = num_channels

        # FFTs
        self.rfft_patch = partial(self.rfft, dim=(2, 4))
        self.irfft_patch = partial(self.irfft, dim=(2, 4))

        self.rfft_slice = partial(self.rfft, dim=(1, 3))
        self.irfft_slice = partial(self.irfft, dim=(1, 3))

        # encode/decode pixels
        self.emb = nn.Linear(num_channels, emb_size)
        self.mask_emb = Parameter(torch.randn(1, 1, 1, emb_size))
        self.dec = nn.Linear(emb_size, num_channels * 256, bias=False)

        # TODO init dec to be inverse of emb

        # mixing layers
        self.scale = Parameter(torch.zeros(num_layers, 3))
        self.mixer = nn.ModuleList()
        for n in range(num_layers):
            sub = nn.ModuleList()

            sub.add_module("channel", block(emb_size, expand))
            sub.add_module("patch", block(2 * emb_size, expand))
            sub.add_module("slice", block(2 * emb_size, expand))

            self.mixer.add_module(f"layer_{n}", sub)

    def patchfy(self, x: Tensor, /) -> Tensor:
        batch, height, width, channel = x.shape

        s = self.size
        x = x.unflatten(2, (width // s, s))
        x = x.unflatten(1, (height // s, s))

        return x

    def unpatchfy(self, x: Tensor, /) -> Tensor:
        x = x.flatten(1, 2)
        x = x.flatten(2, 3)

        return x

    def rfft(self, x: Tensor, /, *, dim: tuple[int, int]) -> Tensor:
        x = torch.fft.rfft2(x, dim=dim)
        x = torch.cat([x.real, x.imag], dim=-1)

        return x

    def irfft(self, x: Tensor, /, *, dim: tuple[int, int]) -> Tensor:
        xr, xi = x.chunk(2, dim=-1)
        x = torch.complex(xr, xi)

        x = torch.fft.irfft2(x, dim=dim)

        return x

    def forward(
        self,
        x: Tensor,
        /,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert x.dtype == torch.uint8

        x = x.to(self.device)
        x = self.emb(x.float())

        # if mask is not None:
        # x = x.mul(mask) + self.mask_emb.mul(~mask)

        # x = self.patchfy(x)

        # for (mixer, scale) in zip(self.mixer, self.scale):
        #     mix_channel, mix_patch, mix_slice = mixer  # type: ignore
        #     s1, s2, s3 = scale

        #     out = mix_channel(x)
        #     x = x + s1 * out

        #     out = self.rfft_patch(x)
        #     out = out + s2 * mix_patch(out)
        #     x = self.irfft_patch(out)

        #     out = self.rfft_slice(x)
        #     out = out + s3 * mix_slice(out)
        #     x = self.irfft_slice(out)

        # x = self.unpatchfy(x)

        x = self.dec(x)
        logits = x.unflatten(dim=-1, sizes=(self.num_channels, 256))
        prob = logits.softmax(dim=-1)

        ids = torch.arange(256, device=self.device).view(1, 1, 1, 1, -1)
        mean = torch.sum(prob * ids, dim=-1)

        diff = ids.sub(mean.unsqueeze(-1)).pow(2)
        std = torch.sum(prob * diff, dim=-1).sqrt()

        return logits, mean, std

    def train_step(self, batch: tuple[Tensor, Tensor], idx: int, /) -> float:
        self.train()

        assert self.optimizer is not None, "optimizer not configured"
        self.optimizer.zero_grad()

        data, mask = tuple(b.to(self.device) for b in batch)

        logits, mean, std = self.forward(data, mask)

        loss = F.cross_entropy(logits.flatten(0, 3), data.flatten())
        if loss.item() > 1:
            loss += 0.1 * F.mse_loss(mean, data.float())
        loss = loss / 255

        loss.backward()
        self.optimizer.step()

        self.log.append(loss.item())

        return loss.item()

    def random_mask(self, x: Tensor, /, *, prob: float) -> Tensor:
        assert 0 <= prob <= 1

        prob /= 2

        s = self.size
        batch, height, width, channels = x.shape

        # pixel mask
        shape = (batch, height, width, 1)
        pixel_mask = torch.rand(*shape, device=self.device) <= prob

        # patch mask
        shape = (batch, height // s, 1, width // s, 1, 1)
        patch_mask = torch.rand(*shape, device=self.device) <= prob
        patch_mask = patch_mask.expand(-1, -1, s, -1, s, -1)
        patch_mask = self.unpatchfy(patch_mask)

        mask = pixel_mask | patch_mask

        return mask
