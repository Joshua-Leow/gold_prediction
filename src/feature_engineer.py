import numpy as np
import pandas as pd
import pandas_ta

from config import target_candle


# def add_technical_indicators(df):
#     """Add more technical indicators for better prediction"""
#     # RSI
#     delta = df['Close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     df['RSI'] = 100 - (100 / (1 + rs))
#
#     # Bollinger Bands
#     df['BB_middle'] = df['Close'].rolling(window=20).mean()
#     bb_std = df['Close'].rolling(window=20).std()
#     df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
#     df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
#     df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
#
#     # Volume indicators
#     df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
#     df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
#
#     # Additional momentum indicators
#     df['ROC'] = df['Close'].pct_change(periods=12) * 100
#
#     return df
#
#
# def get_rsi_features(df, horizons=None):
#     """Calculate RSI features with different horizons"""
#     if horizons is None:
#         horizons = [2, 5, 60, 250, 1000]
#     df['rsi'] = df['Close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
#     new_predictors = []
#     # Calculate RSI-based features for different horizons
#     for horizon in horizons:
#         # RSI
#         delta = df['Close'].diff()
#         RSI_column = f"RSI_Ratio_{horizon}"
#         gain = (delta.where(delta > 0, 0)).rolling(window=horizon).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=horizon).mean()
#         rs = gain / loss
#         df[RSI_column] = 100 - (100 / (1 + rs))
#
#         # Rolling mean of RSI difference
#         rolling_averages = df.Close.rolling(window=horizon, min_periods=1).mean()
#         rsi_ratio_column = f'rsi_ratio_{horizon}'
#         df[rsi_ratio_column] = df['RSI'] / rolling_averages
#         # new_predictors.append(rsi_ratio_column )
#
#         # Calculate RSI Trend
#         rsi_trend = f'rsi_trend_{horizon}'
#         rsi_changes = df['RSI'].pct_change(horizon)
#         # df[macd_changes] = macd_changes.rolling(window=horizon, min_periods=1).mean()
#         # new_predictors.append(rsi_trend)
#
#     return new_predictors, df

def total_random_signal(df, current_candle):
    return np.random.choice([1,2])

def add_total_random_signal(df):
    df['RandomSignal'] = df.apply(lambda row: total_random_signal(df, row.name), axis=1)
    return df

def get_atr(df, length=14):
    atr = pandas_ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=length)
    # normalised_atr = atr.sub(atr.mean()).div(atr.std())
    df['atr'] = atr
    new_predictors = ["atr"]
    return new_predictors, df


def get_bollinger_bands(df, length=20):
    df['bb_low'] = df['Close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=length).iloc[:,0])
    df['bb_mid'] = df['Close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=length).iloc[:,1])
    df['bb_high'] = df['Close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=length).iloc[:,2])
    new_predictors = ["bb_low", "bb_mid", "bb_high"]
    return new_predictors, df


def get_garman_klass_vol(df):
    df['garman_klass_vol'] = ((np.log(df['High'])-np.log(df['Low']))**2)/2-(2*np.log(2)-1)*((np.log(df['Close'])-np.log(df['Open']))**2)
    new_predictors = ["garman_klass_vol"]
    return new_predictors, df


def get_macd_features(df, horizons=None):
    """Calculate MACD features with different horizons"""
    # Calculate basic MACD
    if horizons is None:
        from config import feature_horizons
        horizons = feature_horizons
    macd = df.Close.ewm(span=12, adjust=False).mean() - df.Close.ewm(span=26, adjust=False).mean()
    df['macd'] = macd
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_diff'] = macd - signal

    # MACD crossover signals
    df["macd_cross"] = ((df['macd_diff'] > 0) &
                      (df['macd_diff'].shift(1) <= 0)).astype(int)

    new_predictors = ["macd", 'macd_diff', 'macd_cross']
    # Calculate MACD-based features for different horizons
    for horizon in horizons:
        rolling_averages = df.rolling(window=horizon).mean()

        # Calculate MACD diff ratios using only past data
        diff_ratio_column = f"MACD_Diff_Ratio_{horizon}"
        df[diff_ratio_column] = df["macd_diff"] / rolling_averages["macd_diff"]
        new_predictors.append(diff_ratio_column)
    return new_predictors, df


def get_close_ratio_and_trend(df, horizons=None):
    if horizons is None:
        from config import feature_horizons
        horizons = feature_horizons
    new_predictors = []

    for horizon in horizons:
        rolling_averages = df.rolling(window=horizon).mean()

        # Calculate price ratios using only past data
        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Close"] / rolling_averages["Close"]
        new_predictors.append(ratio_column)

        # Calculate trend of target
        trend_column = f"Close_Trend_{horizon}"
        df[trend_column] = df.shift(target_candle).rolling(window=horizon).sum()["Target"]
        new_predictors.append(trend_column)

    return new_predictors, df


def sort_features(predictors):
    # Hard-coded for now
    return [
        'Close_Ratio_128',  # 0.2700
        'Close_Ratio_512',  # 0.2593
        'macd',  # 0.0744
        'Close_Trend_128',  # 0.0482
        'garman_klass_vol',  # 0.0410
        'bb_low',  # 0.0397
        'bb_high',  # 0.0389
        'bb_mid',  # 0.0387
        'Close_Trend_512',  # 0.0303
        'Close_Trend_32',  # 0.0273
        'Close_Ratio_32',  # 0.0157
        'Close_Trend_8',  # 0.0142
        'Close_Ratio_8',  # 0.0114
        'Close_Trend_2',  # 0.0123
        'atr',  # 0.0079
        'Close_Ratio_2',  # 0.0029
        'MACD_Diff_Ratio_512',  # 0.0072
        'MACD_Diff_Ratio_128',  # 0.0048
        'MACD_Diff_Ratio_32',  # 0.0027
        'MACD_Diff_Ratio_8',  # 0.0021
        'MACD_Diff_Ratio_2',  # 0.0001
        'macd_diff',  # 0.0141
        'macd_cross'  # 0.0001
    ]
