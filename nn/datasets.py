from __future__ import annotations

from pathlib import Path
from multiprocessing.pool import ThreadPool

from PIL import Image

import math
import numpy as np
import torch


class Images:
    def __init__(
        self,
        root: str | Path,
        width: int,
        height: int,
        suffix: str = ".jpg",
    ):
        self.width = width
        self.height = height

        root = Path(root)
        self.imgs = list(root.glob(f"*{suffix}"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)

        path = self.imgs[idx]
        img = Image.open(path).convert("RGB")

        w, h = img.size
        # resize if too small
        if w < self.width or h < self.height:
            ratio = max(self.width / w, self.height / h)

            w, h = math.ceil(w * ratio), math.ceil(h * ratio)
            img = img.resize((w, h), Image.BICUBIC)

        # random crop to width/height
        i = np.random.randint(0, w - self.width) if w != self.width else 0
        j = np.random.randint(0, h - self.height) if h != self.height else 0

        crop = img.crop((i, j, i + self.width, j + self.height))

        data = torch.from_numpy(np.asarray(crop))
        data = data.permute(2, 0, 1)  # (C, H, W)

        return data

    def batch(self, batch_size: int = 1, steps: int = 1, processes: int = 8):
        if processes > batch_size:
            processes = batch_size

        with ThreadPool(processes) as pool:
            for _ in range(0, steps):
                idx = torch.randint(0, len(self), (batch_size,)).tolist()

                data = pool.map(self.__getitem__, idx)

                yield torch.stack(data)
