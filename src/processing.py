import os
from pathlib import Path

import numpy as np
import yfinance as yf

import pandas as pd

from config import target_candle, define_target_labels


def get_period(interval):
    # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    _max_days = ['1d','5d','1wk','1mo','3mo']
    _730_days = ['60m','1h']
    _60_days = ['2m','5m','15m','30m','90m']
    if interval in _max_days:
        return 'max'
    elif interval in _730_days:
        return '730d'
    elif interval in _60_days:
        return '60d'
    elif interval == '1m':
        return '8d'
    return None


def fetch_data(symbol, interval):
    # saved_path = Path(os.path.join(os.getcwd(), f"data/archive/{symbol}_{interval}_archive.csv"))
    saved_path = Path(os.path.join(os.getcwd(), f"data/frd_sample_futures_GC/GC_1min_sample.csv"))
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

    df = (df / 1e2).assign(Volume=df.Volume * 1e2)  # allow fractional shares

    from config import target_candle
    df["Future_High"] = df["High"].shift(-1).rolling(window=target_candle, min_periods=1).max()
    df["Future_Low"] = df["Low"].shift(-1).rolling(window=target_candle, min_periods=1).min()
    df["Future_Close"] = df["Close"].shift(-target_candle)
    df["Target"] = define_target_labels(df)
    df = df.loc["1990-01-01":].copy()
    return df


def final_processing(df):
    # df = df[:-target_candle]
    # print(df[df["Target"]==1])
    return df
