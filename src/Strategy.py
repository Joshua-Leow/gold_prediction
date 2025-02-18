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


class MLTrailingStrategy(Strategy):
    # Strategy parameters
    target_candle = 240
    profit_perc = 4.00
    stop_loss_perc = 1.00
    max_positions = 10

    # Trailing stop parameters
    trailing_stop_atr_multiple = 2.0  # Multiple of ATR for trailing stop
    min_trailing_stop_perc = 0.5     # Minimum trailing stop percentage
    take_profit_atr_multiple = 4.0   # Multiple of ATR for take profit
    atr_periods = 14                 # Periods for ATR calculation

    def init(self):
        """Initialize the strategy with predictions and indicators"""
        if 'Predictions' not in self.data.df:
            raise ValueError("Predictions column not found in data")

        # Store predictions as a custom indicator
        self.predictions = self.I(lambda: self.data.df['Predictions'])

        # Calculate ATR for dynamic stop loss and take profit
        self.atr = self.I(lambda: self.compute_atr(self.atr_periods))

        # Track highest/lowest prices since entry for each trade
        self.trade_highs = {}  # {trade_id: highest_price}
        self.trade_lows = {}  # {trade_id: lowest_price}

    def compute_atr(self, periods):
        """Compute Average True Range"""
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        tr = np.maximum(
            high - low,
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
        return pd.Series(tr).rolling(periods).mean()

    def update_trailing_stops(self):
        """Update stop loss and take profit levels for all open trades"""
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]

        for trade in self.trades:
            trade_id = id(trade)

            # Initialize tracking if new trade
            if trade_id not in self.trade_highs:
                self.trade_highs[trade_id] = trade.entry_price
                self.trade_lows[trade_id] = trade.entry_price

            # Update trade high/low water marks
            if trade.is_long:
                self.trade_highs[trade_id] = max(self.trade_highs[trade_id], current_price)
                trailing_stop = self.trade_highs[trade_id] - (current_atr * self.trailing_stop_atr_multiple)
                # Ensure minimum distance from entry
                min_stop = trade.entry_price * (1 - self.min_trailing_stop_perc / 100)
                trailing_stop = max(trailing_stop, min_stop)
                # Only update if new stop is higher than current
                if not trade.sl or trailing_stop > trade.sl:
                    trade.sl = trailing_stop
                    print(f"Updated long stop loss to: {trailing_stop:.2f}")

            else:  # Short trade
                self.trade_lows[trade_id] = min(self.trade_lows[trade_id], current_price)
                trailing_stop = self.trade_lows[trade_id] + (current_atr * self.trailing_stop_atr_multiple)
                # Ensure minimum distance from entry
                min_stop = trade.entry_price * (1 + self.min_trailing_stop_perc / 100)
                trailing_stop = min(trailing_stop, min_stop)
                # Only update if new stop is lower than current
                if not trade.sl or trailing_stop < trade.sl:
                    trade.sl = trailing_stop
                    print(f"Updated short stop loss to: {trailing_stop:.2f}")

            # Clean up closed trades
            for trade_id in list(self.trade_highs.keys()):
                if trade_id not in [id(t) for t in self.trades]:
                    del self.trade_highs[trade_id]
                    del self.trade_lows[trade_id]

    def next(self):
        """Define trading logic for each step"""
        # Update trailing stops for existing positions
        self.update_trailing_stops()

        # Check if we can open new positions
        if len(self.trades) >= self.max_positions:
            return

        # Calculate position size (fixed fractional position sizing)
        position_value = self.equity / self.max_positions
        position_size = position_value / self.data.Close[-1]
        position_size = int(position_size)  # Round down to whole units

        if position_size < 1:
            return  # Skip if position size is too small

        current_atr = self.atr[-1]
        current_price = self.data.Close[-1]

        # Open new position based on prediction
        if self.predictions[-1] == 1 and len(self.trades) < self.max_positions:
            print(f"Opening long trade at {current_price:.2f}")
            # Initial stop loss and take profit based on ATR
            initial_sl = current_price - (current_atr * self.trailing_stop_atr_multiple)
            initial_tp = current_price + (current_atr * self.take_profit_atr_multiple)
            self.buy(size=position_size, sl=initial_sl, tp=initial_tp)

        elif self.predictions[-1] == -1 and len(self.trades) < self.max_positions:
            print(f"Opening short trade at {current_price:.2f}")
            # Initial stop loss and take profit based on ATR
            initial_sl = current_price + (current_atr * self.trailing_stop_atr_multiple)
            initial_tp = current_price - (current_atr * self.take_profit_atr_multiple)
            self.sell(size=position_size, sl=initial_sl, tp=initial_tp)