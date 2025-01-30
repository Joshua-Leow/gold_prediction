import numpy as np
import pandas as pd


def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    return true_range.rolling(period).mean()


def add_technical_indicators(df):
    """Add more technical indicators for better prediction"""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']

    # Additional momentum indicators
    df['ROC'] = df['Close'].pct_change(periods=12) * 100

    # Volatility
    df['ATR'] = calculate_atr(df)

    return df


def get_macd_features(df, horizons=[2, 5, 60, 250, 1000]):
    """Calculate MACD features with different horizons"""
    # Calculate basic MACD
    macd = df.Close.ewm(span=12, adjust=False).mean() - df.Close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_diff'] = macd - signal

    new_predictors = []
    # Calculate MACD-based features for different horizons
    for horizon in horizons:
        # Rolling mean of MACD difference
        macd_ma = f'macd_ma_{horizon}'
        df[macd_ma] = df['macd_diff'].rolling(window=horizon, min_periods=1).mean()
        # new_predictors.append(macd_ma)

        # Rolling standard deviation of MACD difference
        macd_std = f'macd_std_{horizon}'
        df[macd_std] = df['macd_diff'].rolling(window=horizon, min_periods=1).std()
        # new_predictors.append(macd_std)

        # MACD momentum (rate of change)
        macd_mom = f'macd_mom_{horizon}'
        df[macd_mom] = df['macd_diff'].pct_change(horizon)
        # new_predictors.append(macd_mom)

        # MACD crossover signals
        macd_cross = f'macd_cross_{horizon}'
        df[macd_cross] = ((df['macd_diff'] > 0) &
                          (df['macd_diff'].shift(1) <= 0)).astype(int)
        new_predictors.append(macd_cross)

    return new_predictors, df


def get_close_ratio_and_trend(df, horizons=[2, 5, 60, 250, 1000]):
    new_predictors = []

    for horizon in horizons:
        # Calculate price ratios using only past data
        rolling_averages = df.Close.rolling(window=horizon, min_periods=1).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Close"] / rolling_averages
        new_predictors.append(ratio_column)

        # Calculate trend using price changes instead of target
        trend_column = f"Trend_{horizon}"
        price_changes = df["Close"].pct_change()
        df[trend_column] = price_changes.rolling(window=horizon, min_periods=1).mean()
        new_predictors.append(trend_column)

    return new_predictors, df
