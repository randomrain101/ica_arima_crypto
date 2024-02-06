# %%
from joblib import dump
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

def fit_predict_arima(X, n_periods=1, standard_order=(2, 1, 0)):
    pred_list = []
    order_list = []
    root_list = []
    for symbol in X.columns:
        s = X[symbol]
        model = AutoARIMA(
            #max_d=1,
            #stationary=True,
            seasonal=False,
            max_p=6,
            max_q=6).fit(s)
        aa_order = order = model.model_.order
        roots = (model.model_.arroots(), model.model_.maroots())
        if order == (0, 0, 0) or order == (0, 1, 0):
            order = standard_order
            model = ARIMA(order).fit(s)
            roots = (model.arroots(), model.maroots())
        root_list.append(roots)
        order_list.append(aa_order)
        pred_list.append(model.predict(n_periods))
    
    out = pd.concat(pred_list, axis=1)
    out.columns = X.columns
    return out, order_list, root_list

# %%
tick = "15m"

df_bars = pd.read_csv(
    f"../data/ohlc_{tick}_gateio.csv", index_col=0, parse_dates=True)
df_close = df_bars.filter(axis="columns", like="close")
df_close.columns = [x.split("_")[0] for x in df_close.columns]

tick = "24h"
df_close = df_close.resample(tick).last().iloc[2*356:]

# %%
# walk forward split
n_train_days = 120
ts_split = list(TimeSeriesSplit(
    len(df_close) - n_train_days, 
    max_train_size=n_train_days, 
    test_size=1).split(df_close))

#%%
arma_predictions, arima_orders, arima_roots = zip(*Parallel(n_jobs=-1, backend="loky")(
        delayed(fit_predict_arima)(np.log(df_close.iloc[i_train])) for i_train, _ in tqdm(ts_split)))

#%%
df_arma_pred = np.exp(pd.concat(arma_predictions))
df_arma_pred = (df_arma_pred / df_close.shift().reindex(df_arma_pred.index)) - 1

df_arma_pred.hist(bins=50, figsize=(10, 10));

#%%
df_arma_pred.to_csv(f"../data/arma_predictions_{tick}_{n_train_days}D.csv")

# %%
arima_orders = pd.DataFrame(arima_orders, columns=df_close.columns)

#%%
arima_orders.to_csv(f"../data/arma_orders_{tick}_{n_train_days}D.csv")

#%%
dump(arima_roots, "../data/arima_roots.joblib")

#%%