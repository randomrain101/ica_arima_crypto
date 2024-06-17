# %%
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import normaltest

# %%
tick = "1d"

df_bars = pd.read_csv(
    f"../data/ohlc_{tick}_gateio.csv", index_col=0, parse_dates=True)
df_close = df_bars.filter(axis="columns", like="close")
df_close.columns = [x.split("_")[0] for x in df_close.columns]

tick = "24h"
df_close = df_close.resample(tick).last()

#%%
#
# --- returns normality test ---
#
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html

df_ret = df_close.pct_change().iloc[1:]

# test individual series
df_ret.apply(normaltest, axis=0).set_index(pd.Index(["test_statistic", "p_value"]))

#%%
# test concatenated series
normaltest(df_ret.unstack())

#%%
#
# --- log returns normality test ---
#
df_log_ret = np.log(df_close).diff().iloc[1:]

# test individual series
df_log_ret.apply(normaltest, axis=0).set_index(pd.Index(["test_statistic", "p_value"]))

#%%
# test concatenated series
normaltest(df_log_ret.unstack())