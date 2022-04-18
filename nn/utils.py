from __future__ import annotations

import torch.nn as nn


def num_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
