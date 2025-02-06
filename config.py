import numpy as np

symbol = 'GC=F'           # Trading symbol
interval = '1h'           # Time interval
confidence = 0.50         # Prediction confidence threshold
target_candle = 7         # Future candle to predict
profit_perc = 1.00        # Take profit percentage
stop_loss_perc = 0.20     # Stop loss percentage
gap_between_trades = 2    # Number of candles to wait before making the next trade


def define_target_labels(df):
    return np.where(
        df["Future_Close"] > df["Close"] + (df["Close"] * profit_perc / 100), 1,
        np.where(df["Future_Close"] < df["Close"] - (df["Close"] * profit_perc / 100), -1, 0)
    )
