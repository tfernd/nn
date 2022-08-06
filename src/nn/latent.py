from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class MaskLatent(nn.Module):
    prob: Tensor
    masks: Tensor

    def __init__(
        self,
        features: int,
        # p: float = 0, # TODO add support for p
    ) -> None:
        super().__init__()

        self.features = features

        prob = torch.linspace(1, 1 / features, features)
        self.register_buffer("prob", prob)

        # always leave the first feature off the mask
        masks = ~torch.eye(features).cumsum(0).bool()
        self.register_buffer("masks", masks)

    def mask(self, z: Tensor, /) -> tuple[Tensor, Optional[Tensor]]:
        if not self.training:
            return z, None

        *shape, C = z.shape
        idx = torch.randint(0, self.features, shape, device=z.device)

        mask = self.masks[idx]
        z = z.masked_fill(mask, 0)

        return z, mask

    def crop(
        self,
        z: Tensor,
        /,
        n: Optional[int] = None,
    ) -> Tensor:
        if n is None:
            return z

        assert 1 <= n <= self.features

        return z[..., :n]

    def expand(self, z: Tensor, /) -> Tensor:
        *shape, C = z.shape

        if C == self.features:
            return z

        zeros = torch.zeros(*shape, self.features - C, device=z.device)
        z = torch.cat([z, zeros], dim=-1)

        return z

    def __repr__(self) -> str:
        name = self.__class__.__qualname__
        out = f"{name}(features={self.features}, p={self.p})"

        return out
