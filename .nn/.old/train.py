#%%
from __future__ import annotations

from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch

import pytorch_lightning as pl

from src.datasets import Images
from model import FFTPatchAutoEncoder
from src.utils import num_parameters

ds = Images("datasets/WebToon/episodes", width=512, height=512 * 4)

model = FFTPatchAutoEncoder(
    sh=16,
    sw=16,
    num_channels=3,
    compression=1 / 64,
    kernel_size=3,
    num_layers=4,
)
print(f"Parameters: {num_parameters(model):,}")
out = np.array(model.features) / np.array(model.latent_sizes)
print(out)

#%% test
data = ds[310].unsqueeze(0).cuda()

model.cuda()
with torch.no_grad():
    out, loss = model.forward(data.cuda())
    print(loss)

# out = data
Image.fromarray(out[0].to(torch.uint8).permute(1, 2, 0).cpu().numpy())
# plt.imshow(z.view(z.shape[-2], -1).cpu().numpy(), aspect="auto")


#%%
model.lr = 3e-4

dl = ds.batch(batch_size=5, steps=50_000)

trainer = pl.Trainer(gpus=1, log_every_n_steps=1, auto_lr_find=True)
# trainer.tune(model, dl)
trainer.fit(model, dl)
