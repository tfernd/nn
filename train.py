#%%
from __future__ import annotations

# %load_ext autoreload
# %autoreload 2

from pathlib import Path

from PIL import Image
from matplotlib import pyplot as plt

import numpy as np

import torch
from torch import Tensor

from tqdm.notebook import tqdm

from src.datasets import ImageDataset
from src.models import FFTPatchSuperResolution
from src.utils import num_parameters

model = FFTPatchSuperResolution(
    patch=4,
    scale=4,
    kernel_size=3,
    expand=2,
    num_layers=2,
).cuda()
print(f"Parameters: {num_parameters(model):,}")

#%%
ds = ImageDataset(
    r"C:\\Users\\thale\\#Code\\db\\pics",
    # r"C:\\Users\\thale\\#DCIM\\Camera",
    size=512,
    scale=4,
)

#%% training
steps = 1_000
model.configure_optimizers(lr=1e-4)
dl = ds.batch(16, steps)
with tqdm(dl) as pbar:
    for i, batch in enumerate(pbar):
        try:
            loss = model.training_step(batch, i)

            pbar.set_postfix(loss=loss)
        except KeyboardInterrupt:
            break

#%%
plt.plot(model.log)

#%%
model.eval()
dl = ds.batch(1, steps)
data, target = next(dl)
out = model(data).detach()
d = out[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()

Image.fromarray(d)
