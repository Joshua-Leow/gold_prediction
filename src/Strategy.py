import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 20

    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()


from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np


class MLStrategy(Strategy):
    # Strategy parameters
    target_candle = 240
    profit_perc = 4.00
    stop_loss_perc = 1.00
    max_positions = 10

    def init(self):
        """Initialize the strategy with predictions"""
        if 'Predictions' not in self.data.df:
            raise ValueError("Predictions column not found in data")

        # Store predictions as a custom indicator
        self.predictions = self.I(lambda: self.data.df['Predictions'])

    def next(self):
        """Define trading logic for each step"""
        current_bar = len(self.data) - 1

        # Check existing trades for closing conditions
        for trade in list(self.trades):
            current_price = self.data.Close[-1]
            price_change_perc = ((current_price - trade.entry_price) / trade.entry_price) * 100

            # Determine if we should close the position
            should_close = False

            # Check take profit
            if (trade.is_long and price_change_perc >= self.profit_perc) or \
                    (not trade.is_long and -price_change_perc >= self.profit_perc):
                should_close = True
                print(f"Trade hit take profit: {price_change_perc:.2f}%")

            # Check stop loss
            elif (trade.is_long and price_change_perc <= -self.stop_loss_perc) or \
                    (not trade.is_long and -price_change_perc <= -self.stop_loss_perc):
                should_close = True
                print(f"Trade hit stop loss: {price_change_perc:.2f}%")

            # Check if position has been open for too long
            elif (current_bar - trade.entry_bar) >= self.target_candle:
                should_close = True
                print(f"Trade hit time limit: {current_bar - trade.entry_bar} bars")

            if should_close:
                trade.close()

        # Check if we can open new positions
        if len(self.trades) >= self.max_positions:
            return

        # Calculate position size (fixed fractional position sizing)
        position_value = self.equity / self.max_positions
        position_size = position_value / self.data.Close[-1]
        position_size = int(position_size)  # Round down to whole units

        if position_size < 1:
            return  # Skip if position size is too small

        # Open new position based on prediction
        if self.predictions[-1] == 1 and len(self.trades) < self.max_positions:
            print(f"Opening long trade at {self.data.Close[-1]}")
            self.buy(size=position_size)

        elif self.predictions[-1] == -1 and len(self.trades) < self.max_positions:
            print(f"Opening short trade at {self.data.Close[-1]}")
            self.sell(size=position_size)