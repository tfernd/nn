#%%
from __future__ import annotations

%load_ext autoreload
%autoreload 2

import torch

from PIL import Image
from matplotlib import pyplot as plt

from tqdm.notebook import tqdm

from src.models import FFTPatch
from src.utils import num_parameters
from src.datasets import ImageDataset

torch.cuda.empty_cache()

# ds = ImageDataset(r'C:\\Users\\thale\\#aDCIM\\Camera', size=256)
ds = ImageDataset(r'.database', size=128)

model = FFTPatch(size=8, emb_size=128, expand=2, num_layers=1)
model.cuda()
print(f"Number of parameters: {num_parameters(model):,}")


#%% train
model.configure_optimizers(lr=1e-4)

steps=1000
with tqdm(total=steps)as pbar:
    for i, data in enumerate(ds.batch(batch_size=4, steps=steps)):
        mask = model.random_mask(data, prob=0.0)

        batch = (data, mask)
        loss = model.train_step(batch, i)

        pbar.set_postfix(loss=loss)
        pbar.update()

#%%
plt.plot(model.log)

#%%
model.eval()
data = next(ds.batch(1, steps))
mask = model.random_mask(data, prob=0.0)
logits, mean, std = model.forward(data, mask)

d = mean[0].long().to(torch.uint8).cpu().numpy()
d = logits[0].argmax(-1).long().to(torch.uint8).cpu().numpy()

Image.fromarray(d)

