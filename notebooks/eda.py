# %%
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
tick = "1d"

df_bars = pd.read_csv(
    f"../data/ohlc_{tick}_gateio.csv", index_col=0, parse_dates=True)
df_close = df_bars.filter(axis="columns", like="close")
df_close.columns = [x.split("_")[0] for x in df_close.columns]

tick = "24h"
n_train_days = 120
df_arma_pred = pd.read_csv(
    f"../data/arma_predictions_{tick}_{n_train_days}D.csv",
    index_col=0,
    parse_dates=True)

df_close = df_close[df_arma_pred.columns].resample(tick).last()
#df_close.to_csv("../data/close_1d_gateio.csv")

# %%
df_ret = df_close.pct_change().fillna(0)

#%%
import matplotlib
#matplotlib.use("pgf")
#matplotlib.rcParams.update({
#    "pgf.texsystem": "pdflatex",
#    'font.family': 'serif',
#    'text.usetex': True,
#    'pgf.rcfonts': False,
#})


ax = (df_close / df_close.iloc[0]).plot(figsize=(9, 5), legend=True)
ax.semilogy()
ax.set_ylabel("$p_t$ / $p_0$")
ax.set_xlabel("timestamp t")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125),
          fancybox=True, shadow=True, ncol=5)

plt.savefig('prices.png', bbox_inches="tight")

# %%
df_log_ret = np.log(df_close).diff()
heatmap = sns.heatmap(df_ret.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

# %%
