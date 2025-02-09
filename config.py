import numpy as np

symbol = 'GC=F'                          # Trading symbol
interval = '1h'                          # Time interval
confidence = 0.5                         # Prediction confidence threshold
target_candle = 240                      # Future candle to predict
profit_perc = 4.00                       # Take profit percentage
stop_loss_perc = profit_perc/4           # Stop loss percentage
gap_between_trades = 0                   # Number of candles to wait before making the next trade
feature_horizons = [2, 8, 32, 128, 512]  # Feature Horizons to be trained with
max_positions = 10                       # Max number of open positions at a time


def define_target_labels(df):
    """
    Defines target labels for a given DataFrame.

    Args:
        df: DataFrame containing financial data with columns: ["Future_High", "Future_Low", "Close"]
        profit_perc: Profit percentage.
        stop_loss_perc: Stop-loss percentage.

    Returns: A Series of target labels (-1, 0, or 1).
    """
    long_condition = (df["Future_High"] > df["Close"] + (df["Close"] * profit_perc / 100)) & \
                     (df["Future_Low"] > df["Close"] - (df["Close"] * stop_loss_perc / 100))

    short_condition = (df["Future_Low"] < df["Close"] - (df["Close"] * profit_perc / 100)) & \
                      (df["Future_High"] < df["Close"] + (df["Close"] * stop_loss_perc / 100))

    return np.where(long_condition, 1,
            np.where(short_condition, -1, 0))