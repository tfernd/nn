from __future__ import annotations
from typing import Iterable, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from ..nn import Block, MaskLatent
from ..utils import normalize, denormalize


class DynamicAutoEncoder(pl.LightningModule):
    lr: float = 1e-4
    batch_size: int = 64

    train_dataset: Dataset[Tensor | tuple[Tensor, ...]]
    val_dataset: Dataset[Tensor | tuple[Tensor, ...]]

    def __init__(
        self,
        features: int,
        ratio: float = 1 / 16,
        num_layers: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()

        # parameters
        self.save_hyperparameters()

        self.name = f"DyAE_{features}_{ratio}_{num_layers}"

        ne, nd = (num_layers, num_layers) if isinstance(num_layers, int) else num_layers
        self.num_layers = num_layers = (ne, nd)
        self.features = features

        # layers
        self.encoder = nn.Sequential(*[Block(features, ratio=ratio) for _ in range(ne)])
        self.latent = MaskLatent(features)
        self.decoder = nn.Sequential(*[Block(features, ratio=ratio) for _ in range(nd)])

    def encode(
        self,
        x: Tensor,
        n: Optional[int] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        x = x.to(self.device)

        with torch.set_grad_enabled(self.training):
            x = normalize(x)
            x = self.encoder(x)
            x, mask = self.latent.mask(x)
            x = self.latent.crop(x, n)

        return x, mask

    def decode(self, z: Tensor) -> Tensor:
        z = z.to(self.device)

        with torch.set_grad_enabled(self.training):
            z = self.latent.expand(z)
            z = self.decoder(z)
            z = denormalize(z)

        return z

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def loss(self, data: Tensor, out: Tensor) -> Tensor:
        data = data.to(self.device)
        out = out.to(self.device)

        return F.l1_loss(normalize(data), normalize(out))

    def training_step(
        self,
        batch: Tensor | Iterable[Tensor],
        idx: int,
    ) -> Tensor:
        data = batch if isinstance(batch, Tensor) else next(iter(batch))

        z, mask = self.encode(data)
        out = self.decode(z)

        loss = self.loss(data, out)
        self.log("loss/training", loss)

        return loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: Tensor | Iterable[Tensor],
        idx: int,
    ) -> Tensor:
        self.eval()

        data = batch if isinstance(batch, Tensor) else next(iter(batch))

        ns = [
            2**i
            for i in range(math.ceil(math.log2(self.features)) + 1)
            if 2**i <= self.features
        ]
        if self.features not in ns:
            ns.append(self.features)

        z, mask = self.encode(data)
        assert mask is None
        out = [self.decode(z[:, :n]) for n in ns]
        loss = [self.loss(data, o) for o in out]

        metric = {f"n={n}": l.item() for n, l in zip(ns, loss)}

        # TODO fix type hinting
        self.logger.experiment.add_scalars(
            "loss/validation",
            metric,
            self.current_epoch,
        )

        return loss[-1]

    def add_dataset(self, train: Dataset, val: Dataset) -> None:
        self.train_dataset = train
        self.val_dataset = val

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
