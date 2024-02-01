# %%
import warnings

import ccxt
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# %%
ex = ccxt.gateio()

# %%
def to_unix(x, dt="1ms"):
    return (x - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(dt)

def fetch_ohlc(symbol, length=pd.Timedelta("356 days"), timeframe="15m"):
    out = []
    # okx api -> -8 hours from UTC to Hong Kong Time
    now = pd.Timestamp.utcnow().round(pd.Timedelta(timeframe))# - pd.Timedelta("8 hours")
    last = now - length
    df = pd.DataFrame()
    print("fetching", symbol)
    while now - last > (2*pd.Timedelta(timeframe)):
        x = ex.fetch_ohlcv(symbol, timeframe, limit=1500, since=to_unix(last, "1ms"))
        df = pd.DataFrame(
            x, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        #print(df["timestamp"])
        df = df.set_index("timestamp", drop=True).sort_index()
        df = df.iloc[:-1]
        last = df.index[-1]
        df.columns = symbol + "_" + df.columns
        out.append(df)
        print(now - last, "to go ...")
    print("done.")
    return pd.concat(out, axis=0).sort_index()


# %%
# fetch data
tick = "1d"
length = pd.Timedelta(f"{6*356} days")
symbols = list(
    map(lambda x: x + "/USDT", ["BTC", "ETH", "SOL", "XRP", "LTC", "LINK", "ADA", "DOGE", "TRX", "DOT", "BCH", "XMR", "EOS", "XLM"]))
df_list = []
for symbol in tqdm(symbols[::-1]):
    try:
        df_list.append(fetch_ohlc(symbol, length, timeframe=tick))
    except Exception as e:
        print(symbol, "failed:", str(e))

# %%
df = pd.concat([x.loc[~x.index.duplicated(keep="first")]
               for x in df_list[::-1]], axis=1).iloc[1:-1].dropna(axis=1)
#df.filter(like="close", axis="columns").plot()

# %%
df.to_csv(f"../data/ohlc_{tick}_{ex.__class__.__name__}.csv")
