from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn


class Base(nn.Module):
    optimizer: Optional[torch.optim.Optimizer] = None
    log: list[float]

    def __init__(self):
        super().__init__()

        self.log = []
        self.cpu()

    def to(self, device: Literal["cpu", "cuda"] | torch.device, /):
        super().to(device)

        self.device = torch.device(device)

        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def configure_optimizers(self, *, lr: float):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
