#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
df = pd.DataFrame(np.random.uniform(-1, 1, (1000, 4)))

# %%
df.hist(bins=50)

#%%
df_mix = df @ np.random.uniform(-1, 1, (4, 4))

#%%
df_mix.hist(bins=50)

# %%
