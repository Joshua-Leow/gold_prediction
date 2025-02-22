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
import numpy as np


class MLStrategy(Strategy):
    from config import target_candle, profit_perc, stop_loss_perc, max_positions
    # Strategy parameters
    target_candle = target_candle
    profit_perc = profit_perc
    stop_loss_perc = stop_loss_perc
    max_positions = max_positions

    def init(self):
        """Initialize the strategy with predictions and indicators"""
        if 'Predictions' not in self.data.df:
            raise ValueError("Predictions column not found in data")

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
        if self.data.Predictions[-1] == 1 and len(self.trades) < self.max_positions:
            print(f"Opening long trade at {self.data.Close[-1]}")
            self.buy(size=position_size)

        elif self.data.Predictions[-1] == -1 and len(self.trades) < self.max_positions:
            print(f"Opening short trade at {self.data.Close[-1]}")
            self.sell(size=position_size)


class MLTrailingStrategy(Strategy):
    # Trailing stop Strategy parameters
    max_positions = 10
    trailing_stop_atr_multiple = 3.0  # Multiple of ATR for trailing stop
    take_profit_atr_multiple = 10.0   # Multiple of ATR for take profit

    def init(self):
        # super().init()
        """Initialize the strategy with predictions and indicators"""
        if 'Predictions' not in self.data.df:
            raise ValueError("Predictions column not found in data")

    def next(self):
        # super().next()
        """Define trading logic for each step"""
        # Update trailing stops for existing positions
        # self.update_trailing_stops()
        for trade in self.trades:
            sltr = self.trailing_stop_atr_multiple * self.data.atr[-1]
            if trade.is_long:
                print(f"Updated long trade: {trade.entry_bar} stop loss from: {trade.sl} to ", end="")
                trade.sl = max(trade.sl or np.inf, self.data.Close[-1] - sltr)
                print(trade.sl)
            else:
                print(f"Updated short trade: {trade.entry_bar} stop loss from: {trade.sl} to ", end="")
                trade.sl = min(trade.sl or np.inf, self.data.Close[-1] + sltr)
                print(trade.sl)

        # Check if we can open new positions
        if len(self.trades) >= self.max_positions:
            return

        # Calculate position size (fixed fractional position sizing)
        position_value = self.equity / self.max_positions
        position_size = position_value / self.data.Close[-1]
        position_size = int(position_size)  # Round down to whole units

        if position_size < 1:
            return  # Skip if position size is too small

        if self.data.Predictions[-1] == 1 and len(self.trades) < self.max_positions:
            current_close = self.data.Close[-1]
            sl = current_close - self.trailing_stop_atr_multiple * self.data.atr[-1]
            tp = current_close + self.take_profit_atr_multiple * (self.trailing_stop_atr_multiple * self.data.atr[-1])
            print(f"Opening long trade on {self.data.index[-1]}, at price: {self.data.Close[-1]:.2f}")
            self.buy(size=position_size, sl=sl)
        elif self.data.Predictions[-1] == -1 and len(self.trades) < self.max_positions:
            current_close = self.data.Close[-1]
            sl = current_close + self.trailing_stop_atr_multiple * self.data.atr[-1]
            tp = current_close - self.take_profit_atr_multiple * (self.trailing_stop_atr_multiple * self.data.atr[-1])
            print(f"Opening short trade on {self.data.index[-1]}, at price:  {self.data.Close[-1]:.2f}")
            self.sell(size=position_size, sl=sl)


class RandomTrailingStrategy(Strategy):
    mysize = 0.01  # Trade size 1% of the account
    sl_atr_ratio = 3
    tp_sl_ratio = 1

    def init(self):
        super().init()

    def next(self):
        super().next()

        for trade in self.trades:
            sltr = self.sl_atr_ratio * self.data.atr[-1]
            if trade.is_long:
                trade.sl = max(trade.sl or -np.inf, self.data.Close[-1] - sltr)
            else:
                trade.sl = min(trade.sl or np.inf, self.data.Close[-1] + sltr)

        if self.data.RandomSignal[-1] == 2 and not self.position:
            # Open a new long position with calculated SL
            current_close = self.data.Close[-1]
            sl = current_close - self.sl_atr_ratio * self.data.atr[-1]  # SL below the close price
            # tp = current_close + self.tp_sl_ratio * (self.sl_atr_ratio * self.data.ATR[-1])  # TP above the close price
            self.buy(size=self.mysize, sl=sl)

        elif self.data.RandomSignal[-1] == 1 and not self.position:
            # short position
            current_close = self.data.Close[-1]
            sl = current_close + self.sl_atr_ratio * self.data.atr[-1]
            # tp = current_close - self.tp_sl_ratio * (self.sl_atr_ratio * self.data.ATR[-1])
            self.sell(size=self.mysize, sl=sl)