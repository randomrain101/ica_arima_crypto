# %%
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from statsmodels.stats.stattools import jarque_bera

def summary_statistics(df):
    # Dictionary to hold summary statistics for each column
    summary_dict = {
        'Mean': [],
        'Variance': [],
        'Skewness': [],
        'Kurtosis': [],
        'Jarque-Bera': [],
        'p-value': []
    }

    # Iterate over each column in the DataFrame
    for col in df.columns:
        data = df[col].dropna()  # Drop NaN values

        # Calculate first to fourth moments
        mean = data.mean()
        variance = data.var()
        skewness = skew(data)
        kurt = kurtosis(data, fisher=False)  # Fisher=False for population kurtosis (i.e., including +3)

        # Perform Jarque-Bera test
        jb_stat, jb_pvalue, _, _ = jarque_bera(data)

        # Append results to dictionary
        summary_dict['Mean'].append(mean)
        summary_dict['Variance'].append(variance)
        summary_dict['Skewness'].append(skewness)
        summary_dict['Kurtosis'].append(kurt)
        summary_dict['Jarque-Bera'].append(jb_stat)
        summary_dict['p-value'].append(jb_pvalue)

    # Create summary statistics DataFrame
    summary_df = pd.DataFrame(summary_dict, index=df.columns)
    
    return summary_df

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
# Get summary statistics
summary_df = summary_statistics(df_log_ret)

# Print the summary statistics table
print(summary_df)

# %%
summary_df.to_latex()