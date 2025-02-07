import numpy as np

# symbol = 'GC=F'           # Trading symbol
symbol = 'GRAPHITE.NS'           # Trading symbol
interval = '1d'           # Time interval
confidence = 0.40         # Prediction confidence threshold
target_candle = 7         # Future candle to predict
profit_perc = 25.00        # Take profit percentage
stop_loss_perc = 10.00     # Stop loss percentage
gap_between_trades = 3    # Number of candles to wait before making the next trade


def define_target_labels(df):
    return np.where(
        df["Future_Close"] > df["Close"] + (df["Close"] * profit_perc / 100), 1,
        np.where(df["Future_Close"] < df["Close"] - (df["Close"] * profit_perc / 100), -1, 0)
    )
