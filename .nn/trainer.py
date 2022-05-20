#%%
from __future__ import annotations

# %load_ext autoreload
# %autoreload 2

from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from nn.datasets import Images
from nn.utils import num_parameters
from nn.models import FFTPatchAutoEncoder

ds = Images("datasets/WebToon/episodes", width=512, height=512)

model = FFTPatchAutoEncoder(
    height=8,
    width=8,
    num_channels=3,
    compression=1 / 4,
    kernel_size=3,
    mask_prob=1 / 4,
)
model.add_latent_mixer(kernel_size=5, num_layers=4, expand=4)
model.add_mixer(patch_size=4, kernel_size=3, num_layers=4, expand=2)
print(f"Parameters: {num_parameters(model):,}")
print(f"Parameters: {num_parameters(model.encoder):,}")
print(f"Parameters: {num_parameters(model.latent_mixer):,}")
print(f"Parameters: {num_parameters(model.mixer):,}")


#
#%% test
data = ds[310].unsqueeze(0).cuda()

model.cuda()
with torch.no_grad():
    batch, channels, height, width = data.shape

    mask = model.random_mask(data)

    out, z, loss = model.forward(data.cuda(), mask=mask)
    print(loss)

# out = data
Image.fromarray(out[0].to(torch.uint8).permute(1, 2, 0).cpu().numpy())

#%% train
model.lr = 5e-4
dl = ds.batch(batch_size=4, steps=10_000)

checkpoint = ModelCheckpoint(save_top_k=2, monitor="loss")
trainer = pl.Trainer(
    gpus=1,
    log_every_n_steps=1,
    auto_lr_find=True,
    max_epochs=1,
    callbacks=[checkpoint],
)
# trainer.tune(model, dl)
trainer.fit(model, dl)
