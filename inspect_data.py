#!/usr/bin/env python
# In[1]:
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src import get_data_dir
from src.models.util import ffn_model

datasets = get_data_dir() / "datasets"
char = datasets / "char"

char_train = char / "Char_train.npz"
# char_train = char / "Char_valid.npz"

ar = np.load(char_train)
ar


# In[2]:


list(ar.keys())


# In[3]:


ar["raw_data"].shape


# In[4]:


ar["variable"]


# In[5]:


UNK = -99.99


# In[6]:


l = ar["raw_data"][:, 53, 0]

# Replace -99.9 with np.nan
l[l == -99.99] = np.nan

pd.Series(l).dropna().cumsum().pipe(np.exp).plot()


# In[7]:


# raw_data = np.reshape(ar['raw_data'], newshape=(-1, 47))
# raw_data = raw_data[raw_data[:, 0] != UNK]
data = ar["raw_data"]
np.random.shuffle(data)

data.shape


# In[8]:


# np.random.shuffle(raw_data)

# In[9]:


# In[10]:


tf.config.list_physical_devices("GPU")


# In[11]:


total_epochs: int = 10
steps_per_epoch: int = 100
print_every_n_steps: int = 10

for epoch in range(total_epochs):
    print("Epoch: ", epoch)
    print("Train SDF network first")

    # for step in tqdm(range(steps_per_epoch)):
    for step in range(steps_per_epoch):
        loss, sharpe = train_step(
            data=data,
            sdf_net=sdf_net,
            moment_net=moment_net,
            sdf_optimizer=sdf_optimizer,
            moment_optimizer=moment_optimizer,
            update_sdf_or_moment_network=True,
        )

        if step % print_every_n_steps == 0:
            print(f"Step #{step}, Loss: {loss}, Sharpe: {sharpe}")

    print("Train moment network")
    # for step in tqdm(range(steps_per_epoch)):
    for step in range(steps_per_epoch):
        loss, sharpe = train_step(
            data=data,
            sdf_net=sdf_net,
            moment_net=moment_net,
            sdf_optimizer=sdf_optimizer,
            moment_optimizer=moment_optimizer,
            update_sdf_or_moment_network=False,
        )

        if step % print_every_n_steps == 0:
            print(f"Step #{step}, Loss: {loss}, Sharpe: {sharpe}")
