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

        # Track position entry data using trade index
        self.trade_data = {}  # {trade_id: {'entry_price': price, 'entry_bar': bar}}

    def next(self):
        """Define trading logic for each step"""
        # First, check if any existing positions need to be closed
        current_bar = len(self.data) - 1

        for trade in list(self.trades):
            trade_id = id(trade)

            if trade_id in self.trade_data:
                entry_data = self.trade_data[trade_id]
                entry_price = entry_data['entry_price']
                entry_bar = entry_data['entry_bar']
                current_price = self.data.Close[-1]

                # Calculate price changes as percentages
                price_change_perc = ((current_price - entry_price) / entry_price) * 100
                print(f"Trade: {trade_id}: entry_price: {entry_price}, entry_date: {entry_bar}, current perc: {price_change_perc}")

                # Determine if we should close the position
                should_close = False

                # Check take profit
                if (trade.is_long and price_change_perc >= self.profit_perc) or \
                        (not trade.is_long and -price_change_perc >= self.profit_perc):
                    print("Trying to close with profit")
                    should_close = True
                # Check stop loss
                elif (trade.is_long and price_change_perc <= -self.stop_loss_perc) or \
                        (not trade.is_long and -price_change_perc <= -self.stop_loss_perc):
                    print("Trying to close with loss")
                    should_close = True
                # Check if position has been open for too long
                elif (current_bar - entry_bar) >= self.target_candle:
                    print("Trying to close. exceeded duration")
                    should_close = True

                if should_close:
                    trade.close()
                    del self.trade_data[trade_id]

        # Check if we can open new positions
        current_positions = len(self.trades)
        if current_positions > self.max_positions:
            print("Too many opened positions")
            return

        # Calculate position size (fixed fractional position sizing)
        position_value = self.equity / self.max_positions
        position_size = position_value / self.data.Close[-1]

        # Round down to ensure we don't exceed available cash
        position_size = int(position_size)

        if position_size < 1:
            print("Position size too small")
            return  # Skip if position size is too small

        # Open new position based on prediction
        if self.predictions[-1] == 1 and current_positions < self.max_positions:
            print("Trying to open long trade")
            trade = self.buy(size=position_size)
            # Store entry data using trade object's id
            self.trade_data[id(trade)] = {
                'entry_price': self.data.Close[-1],
                'entry_bar': current_bar
            }
        elif self.predictions[-1] == -1 and current_positions < self.max_positions:
            print("Trying to open short trade")
            trade = self.sell(size=position_size)
            # Store entry data using trade object's id
            self.trade_data[id(trade)] = {
                'entry_price': self.data.Close[-1],
                'entry_bar': current_bar
            }