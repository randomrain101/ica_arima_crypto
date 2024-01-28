# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pmdarima.arima import AutoARIMA, ARIMA
from scipy.stats import ttest_1samp, ttest_ind
from antropy import detrended_fluctuation
from tqdm import tqdm

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.model_selection import TimeSeriesSplit

# %%
tick = "15m"

df_bars = pd.read_csv(
    f"../data/ohlc_{tick}_gateio.csv", index_col=0, parse_dates=True)
df_close = df_bars.filter(axis="columns", like="close")
df_close.columns = [x.split("_")[0] for x in df_close.columns]

tick = "24h"
df_close = df_close.resample(tick).last()

# %%
n_test = 2
df_temp = df_close.loc["2019-01":"2019-04"]
sc = StandardScaler().fit(df_temp.iloc[:-n_test-1])
df_close_norm = pd.DataFrame(sc.transform(df_temp.iloc[:-n_test-1]), index=df_temp.index[:-n_test-1], columns=df_temp.columns)
ica = FastICA(max_iter=5000, algorithm="deflation").fit(df_close_norm)

#%%
df_components = pd.DataFrame(ica.components_, index=df_close.columns)
plt.imshow(df_components @ df_components.T)
df_components

#%%
for i in df_components.index:
    df_components.loc[i].plot.bar(grid=True)
    plt.show()

#%%
plt.imshow(df_components)

#%%
sns.jointplot(df_close_norm, x="BTC/USDT", y="LINK/USDT")

#%%
sns.pairplot(df_close_norm)

#%%
df_close_trans = pd.DataFrame(ica.transform(df_close_norm.iloc[:-n_test]), index=df_close_norm.iloc[:-n_test].index)
sns.jointplot(df_close_trans, x=0, y=1)

#%%
sns.pairplot(df_close_trans)

#%%
df_fractal_dim = df_close_trans.astype(np.float64).apply(detrended_fluctuation, raw=True)
df_fractal_dim.rename("ica").plot.hist(legend=True)
df_close_norm.astype(np.float64).apply(detrended_fluctuation, raw=True).rename("original").hist(legend=True)

# %%
df_close_norm.ewm(span=7).mean().plot(figsize=(10, 5))

#%%
df_close_trans.ewm(span=7).mean().plot(figsize=(10, 5))

#%%
pred_list = []
for c in df_close_trans.columns:
    s = df_close_trans[c].ewm(span=7).mean()
    #s = df_close_trans[c]
    #df_close_trans[c].plot(figsize=(10, 5))
    aa = AutoARIMA(
        max_p=6, 
        max_q=6, 
        #stationary=True,
        seasonal=False).fit(s)
    order = aa.model_.order
    s_pred = aa.predict(n_test)
    pred_list.append(s_pred)
    print(order)
    plt.plot(s_pred)
    s.plot(figsize=(10, 5), grid=True)
    plt.show()

# %%
df_pred = pd.DataFrame(
    ica.inverse_transform(pd.concat(pred_list, axis=1)), 
    index=pred_list[0].index, 
    columns=df_close_norm.columns)
#df_pred.iloc[0] += df_close_norm.iloc[-n_test-2]
#df_pred = df_pred.cumsum()
df_pred = pd.DataFrame(sc.inverse_transform(df_pred), index=df_pred.index, columns=df_pred.columns)
#df_pred = pd.DataFrame(sc.inverse_transform(df_close_norm.reindex), index=df_pred.index, columns=df_pred.columns)

#%%
for c in df_pred.columns:
    #df_close_norm[c].plot()
    df_close[c].reindex(df_close_norm.index).plot(grid=True)
    df_pred[c].plot(grid=True)
    plt.show()

# %%
