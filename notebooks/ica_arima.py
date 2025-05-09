# %%
import warnings
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pmdarima.warnings import ModelFitWarning
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


def fit_predict_ica_arima(X, ica_kwargs, n_periods=1, standard_order=(2, 0, 0)):
    sc = StandardScaler().fit(X)
    X_norm = sc.transform(X)
    w_init = np.corrcoef(X_norm.T)
    ica = FastICA(**ica_kwargs, w_init=w_init, algorithm="deflation").fit(X_norm)
    #ica = FastICA(max_iter=200).fit(X_norm)
    X_trans = pd.DataFrame(ica.transform(X_norm), index=X.index)
    pred_list = []
    order_list = []
    root_list = []
    for component in X_trans.columns:
        s = X_trans[component]
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
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    model = ARIMA(order).fit(s)
                except (UserWarning,  ModelFitWarning, np.linalg.LinAlgError):
                    print("Using (2, 1, 0)")
                    warnings.resetwarnings()
                    model = ARIMA((2, 1, 0)).fit(s)
            roots = (model.arroots(), model.maroots())
        root_list.append(roots)
        order_list.append(aa_order)
        pred_list.append(model.predict(n_periods))
    
    out = sc.inverse_transform(ica.inverse_transform(pd.concat(pred_list, axis=1)))
    return pd.DataFrame(out, index=pred_list[0].index, columns=X.columns), order_list, root_list, ica
    
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
ica_kwargs = {
        "fun": "exp",
        "max_iter": 1000
    }


#%%
ica_arma_predictions, ica_orders, ica_roots, ica_list = zip(*Parallel(n_jobs=-1, backend="loky")(
        delayed(fit_predict_ica_arima)(df_close.iloc[i_train], ica_kwargs) for i_train, _ in tqdm(ts_split)))

#%%
df_ica_pred = pd.concat(ica_arma_predictions)
df_ica_pred = (df_ica_pred / df_close.shift()).reindex(df_ica_pred.index) - 1

df_ica_pred.hist(bins=50, figsize=(10, 10));

#%%
df_ica_pred.to_csv(f"../data/ica_arma_predictions_{tick}_{n_train_days}D.csv")

# %%
ica_orders = pd.DataFrame(ica_orders, columns=df_close.columns)

#%%
ica_orders.to_csv(f"../data/ica_arma_orders_{tick}_{n_train_days}D.csv")

#%%
dump(ica_roots, "../data/ica_roots.joblib")

#%%