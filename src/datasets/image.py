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
    imgs_path: list[Path]

    def __init__(
        self,
        root: str | Path,
        *,
        size: int,
        suffix: str = "jpg",
    ):
        root = Path(root)

        self.imgs_path = []
        for path in tqdm(list(root.rglob(f"*.{suffix}"))):
            img = Image.open(path)
            img_size = img.size

            if min(img_size) < size:
                continue

            self.imgs_path.append(path)

        self.crop = tf.RandomCrop(size)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx: int) -> Tensor:
        path = self.imgs_path[idx]

        with Image.open(path) as img:
            img = self.crop(img)
        data = img2tensor(img)

        return data

    def batch(self, *, batch_size: int = 1, steps: int = 1, processes: int = 8):
        with ThreadPool(processes) as pool:
            for _ in range(0, steps):
                idx = torch.randint(0, len(self), (batch_size,)).tolist()

                out = pool.map(self.__getitem__, idx)

                data = torch.stack(out)

                yield data


def img2tensor(img, /) -> Tensor:
    data = torch.from_numpy(np.asarray(img))

    return data
