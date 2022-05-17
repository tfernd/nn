from __future__ import annotations

from pathlib import Path
from multiprocessing.pool import ThreadPool

from PIL import Image

import numpy as np

import torch
from torch import Tensor
import torchvision.transforms as tf

from tqdm.notebook import tqdm


class ImageDataset:
    def __init__(
        self,
        root: str | Path,
        *,
        size: int,
        scale: int,
        suffix: str = ".jpg",
    ):
        assert size % scale == 0

        root = Path(root)

        self.imgs_path, self.imgs_size = [], []
        for path in tqdm(list(root.rglob(f"*{suffix}"))):
            img = Image.open(path)
            img_size = img.size

            if min(img_size) < size:
                continue

            self.imgs_path.append(path)
            self.imgs_size.append(img_size)

        self.crop = tf.RandomCrop(size)
        self.scale = tf.Resize(size // scale)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        path = self.imgs_path[idx]

        img = Image.open(path).convert("RGB")

        target = self.crop(img)
        data = self.scale(target)

        data = img2tensor(data)
        target = img2tensor(target)

        if data.float().std() < 10:
            return self[idx]

        return data, target

    def batch(self, batch_size: int = 1, steps: int = 1, processes: int = 8):
        if processes > batch_size:
            processes = batch_size

        with ThreadPool(processes) as pool:
            for _ in range(0, steps):
                idx = torch.randint(0, len(self), (batch_size,)).tolist()

                out = pool.map(self.__getitem__, idx)

                data = torch.stack([d for (d, t) in out])
                target = torch.stack([t for (d, t) in out])

                yield data, target


def img2tensor(img, /) -> Tensor:
    data = torch.from_numpy(np.asarray(img))
    return data.permute(2, 0, 1)
