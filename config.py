import numpy as np

symbol = 'GC=F'                    # Trading symbol
# symbol = 'GRAPHITE.NS'           # Trading symbol
interval = '1h'                    # Time interval
confidence = 0.40                  # Prediction confidence threshold
target_candle = 7                  # Future candle to predict
profit_perc = 20.70                 # Take profit percentage
stop_loss_perc = profit_perc/3     # Stop loss percentage
gap_between_trades = 3             # Number of candles to wait before making the next trade


def define_target_labels(df):
    return np.where(df["Future_High"] > df["Close"] + (df["Close"] * profit_perc /100), 1,
        np.where(df["Future_Low"] < df["Close"] - (df["Close"] * profit_perc /100), -1, 0)
    )
