# %%
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import ttest_1samp, ttest_ind
from scipy.stats import pearsonr


# git clone https://github.com/jeslago/epftoolbox.git && cd epftoolbox && pip install .
from  epftoolbox.evaluation import DM


def warn(*args, **kwargs):
    pass


warnings.warn = warn


# %%
tick = "1d"

df_bars = pd.read_csv(
    f"../data/ohlc_{tick}_gateio.csv", index_col=0, parse_dates=True)
df_close = df_bars.filter(axis="columns", like="close")
df_close.columns = [x.split("_")[0] for x in df_close.columns]

tick = "24h"
df_close = df_close.resample(tick).last()


# %%
n_train_days = 120
df_ica_pred = pd.read_csv(
    f"../data/ica_arma_predictions_{tick}_{n_train_days}D.csv",
    index_col=0,
    parse_dates=True)
df_arma_pred = pd.read_csv(
    f"../data/arma_predictions_{tick}_{n_train_days}D.csv",
    index_col=0,
    parse_dates=True).reindex(df_ica_pred.index)

# %%
df_ret = df_close.pct_change().fillna(0).loc[df_arma_pred.index, df_arma_pred.columns].unstack()

df_ica_pred = df_ica_pred.unstack()
df_arma_pred = df_arma_pred.unstack()


# %% [markdown]
#  # Empirical Results

#  ## R (Pearson Correlation)
# - **Pearson Correlation** of ica_arima **better** than just arima

# %%
pearsonr(df_ret, df_ica_pred, alternative="greater"), \
    pearsonr(df_ret, df_arma_pred, alternative="greater")


# %% [markdown]
# ## Directional accuracy
# - Diebold Mariano test shows **Directional Accuarcy** (correct sign of returns predicted) for ica_arima significantly **better** than arima

#%%
print("Diebold Mariano p-value:",
DM(
    np.sign(df_ret).values.reshape(-1, 2),
    np.sign(df_arma_pred).values.reshape(-1, 2),
    np.sign(df_ica_pred).values.reshape(-1, 2),
    norm=1,
    version="multivariate"))


# %% [markdown]
# ## Mean Absolute Error
# - Diebold Mariano test shows **Mean Absolute Error** of ica_arima predicitons significantly **worse** than arima

# %%
#%%
print("Diebold Mariano p-value:",
DM(
    df_ret.values.reshape(-1, 2),
    df_arma_pred.values.reshape(-1, 2),
    df_ica_pred.values.reshape(-1, 2),
    norm=1,
    version="multivariate"))


# %%
# %% [markdown]
# ## Mean Squared Error
# - Diebold Mariano test shows **Mean Squared Error** of ica_arima predicitons significantly **worse** than arima

#%%
print("Diebold Mariano p-value:",
DM(
    df_ret.values.reshape(-1, 2),
    df_arma_pred.values.reshape(-1, 2),
    df_ica_pred.values.reshape(-1, 2),
    norm=2,
    version="multivariate"))

# %% [markdown]
# ## Comaparison of ARIMA orders
# - for just ARIMA the AutoARIMA algorithm determined Brownian Noise with order (0, 1, 0) to be the best model of the process 
# way more often than for ica + ARIMA, indicating application of ICA improved the Signal to Noise Ratio


#%%
ttest_ind(
    (np.sign(df_ret) == np.sign(df_ica_pred)),
    (np.sign(df_ret) == np.sign(df_arma_pred)),
    alternative="greater")

ttest_ind((
    (df_ret - df_ica_pred)**2).values,
    ((df_ret - df_arma_pred)**2).abs().values,
    alternative="less")

ttest_ind(
    (df_ica_pred - df_ret).abs().values, 
    (df_arma_pred - df_ret).abs().values,
    alternative="less")
