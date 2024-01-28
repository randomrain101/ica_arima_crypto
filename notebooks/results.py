# %%
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import ttest_1samp, ttest_ind
from scipy.stats import pearsonr


def warn(*args, **kwargs):
    pass


warnings.warn = warn


# %%
tick = "15m"

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
df_ret = df_close.pct_change().fillna(0).reindex(df_arma_pred.index)

# %%
#
# --- fitted arima orders ---
#
ica_orders = pd.read_csv(f"../data/ica_arma_orders_{tick}_{n_train_days}D.csv", index_col=0)
arima_orders = pd.read_csv(f"../data/arma_orders_{tick}_{n_train_days}D.csv", index_col=0)
orders = pd.concat((
    ica_orders.unstack().value_counts().rename("ica_arima_orders"),
    arima_orders.unstack().value_counts().rename("arima_orders")
    ), axis=1, join="inner")

orders.iloc[np.argsort(orders.sum(axis=1))[:-30:-1]].plot.bar(figsize=(15, 5))
plt.xlabel("ARIMA order (p, d, q)")
plt.ylabel("count")

# %%
#
# --- Mean Absolute Error ---
#
ttest_ind(
    (df_ica_pred - df_ret).abs().unstack().values, 
    (df_arma_pred - df_ret).abs().unstack().values,
    alternative="less")

# %%
#
# --- Mean Squared Error ---
#
ttest_ind((
    (df_ret - df_ica_pred)**2).unstack().values,
    ((df_ret - df_arma_pred)**2).abs().unstack().values,
    alternative="less")

# %%
#
# --- R (Pearson Correlation) ---
#
pearsonr(df_ret.unstack(), df_ica_pred.unstack(), alternative="greater")

# %%
#
# --- directional accuracy (correct sign of returns was predicted) ---
#
ttest_ind(
    (np.sign(df_ret.unstack()) == np.sign(df_ica_pred.unstack())),
    (np.sign(df_ret.unstack()) == np.sign(df_arma_pred.unstack())),
    alternative="greater")

#%%
