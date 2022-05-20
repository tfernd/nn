#%%
from __future__ import annotations
from typing import Literal, Optional

%load_ext autoreload
%autoreload 2

import torch
import torch.nn as nn
from torch import Tensor

from tqdm.notebook import tqdm

from matplotlib import pyplot as plt

from src.models import FFTModel
from src.datasets import ImageDataset
from src.utils import num_parameters

ds = ImageDataset(r"C:\Users\thale\#DCIM\wpp", size=128)

model = FFTModel(
    scale=2,
    num_steps=10,
    hidden_size=32,
    num_layers=2,
    expand=2,
    kernel_size=3,
    num_channel=3,
).cuda()
print(f"Parameters: {num_parameters(model):,}")
model.sub_scale

#%%
model.configure_optimizers(lr=1e-8)
steps=1_000
with tqdm(total=steps) as pbar:
    for idx, target in enumerate(ds.batch(batch_size=4, steps=steps)):
        loss = model.train_step(target, idx)

        pbar.set_postfix(loss=loss)
        pbar.update()

#%%
plt.plot(model.log)
# plt.yscale("log")

#%%
from PIL import Image

data = next(ds.batch(batch_size=1, steps=1))
with torch.no_grad():
    out = model(data, repeat=2)

out = out[0].to(torch.uint8).cpu().permute(1,2,0).numpy()
Image.fromarray(out)
