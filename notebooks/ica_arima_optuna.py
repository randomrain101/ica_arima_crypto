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

import optuna

def fit_predict_ica_arima(X, n_periods=1, standard_order=(2, 0, 0), ica_kwargs={}):
    sc = StandardScaler().fit(X)
    X_norm = sc.transform(X)
    ica = FastICA(**ica_kwargs, w_init=np.corrcoef(X_norm.T), algorithm="deflation").fit(X_norm)
    X_trans = pd.DataFrame(ica.transform(X_norm), index=X.index)
    pred_list = []
    for component in X_trans.columns:
        s = X_trans[component]
        model = AutoARIMA(
            #max_d=1,
            #stationary=True,
            seasonal=False,
            max_p=6,
            max_q=6).fit(s)
        order = model.model_.order
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
        pred_list.append(model.predict(n_periods))
    
    out = sc.inverse_transform(ica.inverse_transform(pd.concat(pred_list, axis=1)))
    return pd.DataFrame(out, index=pred_list[0].index, columns=X.columns)

def objective(trial: optuna.Trial):
    ica_kwargs = {
        "fun": trial.suggest_categorical("fun", ["logcosh", "exp", "cube"]),
        "max_iter": trial.suggest_int("max_iter", 200, 5000, 100)
    }

    ica_arma_predictions = Parallel(n_jobs=-1, backend="loky")(
        delayed(fit_predict_ica_arima)(
            df_close.iloc[i_train], ica_kwargs=ica_kwargs) for i_train, _ in tqdm(ts_split))
    df_ica_pred = pd.concat(ica_arma_predictions)
    df_ica_pred = (df_ica_pred / df_close.shift().reindex(df_ica_pred.index)) - 1

    df_ret = df_close.reindex(df_ica_pred.index).pct_change().fillna(0)

    random = np.where(np.random.binomial(1, 0.5, df_ret.shape) == 0, -1, 1)
    df_dacc = pd.DataFrame(
        {
            "ica_arma": (np.sign(df_ret) == np.sign((df_ica_pred))).resample("60D").sum().unstack(),
            "random": (np.sign(df_ret) == random).resample("60D").sum().unstack()
        }
    )
    df_dacc = df_dacc.divide(df_ret.resample("60D").count().unstack().values, axis=0)

    return df_dacc["ica_arma"].mean()

    
# %%
tick = "15m"
ex = "gateio"

df_bars = pd.read_csv(
    f"../data/ohlc_{tick}_{ex}.csv", index_col=0, parse_dates=True)
df_close = df_bars.filter(axis="columns", like="close")
df_close.columns = [x.split("_")[0] for x in df_close.columns]

tick = "24h"
df_close = df_close.resample(tick).last()

# %%
# walk forward split
n_train_days = 120
ts_split = list(TimeSeriesSplit(
    len(df_close) - n_train_days, 
    max_train_size=n_train_days, 
    test_size=1).split(df_close))[:356]

study = optuna.create_study(
        storage="sqlite:////home/oliver/mtrade/data/optuna.db",
        direction="maximize",
        study_name=f"ica arima study",
        load_if_exists=True
    )

study.optimize(objective, n_trials=20)

# optuna-dashboard sqlite:////home/oliver/mtrade/data/optuna.db

#%%