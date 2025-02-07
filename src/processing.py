import os
from pathlib import Path

import yfinance as yf

import pandas as pd

from config import target_candle, define_target_labels


def get_period(interval):
    if interval == '1d' or interval == '1w':
        return 'max'
    elif interval == '1h':
        return '730d'
    elif interval == '5m':
        return '60d'
    elif interval == '1m':
        return '8d'


def fetch_data(symbol, interval):
    saved_path = Path(os.path.join(os.getcwd(), f"data/archive/{symbol}_{interval}_archive.csv"))
    if os.path.exists(saved_path):
        df = pd.read_csv(saved_path, index_col=0, header=[0, 1])
    else:
        period=get_period(interval)
        df = yf.download(symbol, period=period, interval=interval, ignore_tz=True, progress=False)
        df.to_csv(Path(os.path.join(os.getcwd(), f"data/{symbol}_{interval}.csv")))
        print(f"Saved file to data/{symbol}_{interval}.csv")
    return df


def preprocess_data(df):
    df.index = pd.to_datetime(df.index, utc=True).map(lambda x: x.tz_convert('Singapore'))
    df.columns = df.columns.droplevel(1)

    from config import target_candle
    df["Future_Close"] = df["Close"].shift(-target_candle)
    df["Target"] = define_target_labels(df)
    # df = df.loc["1990-01-01":].copy()
    return df


def final_processing(df):
    df = df[:-target_candle]
    # print(df[df["Target"]==1])
    return df
