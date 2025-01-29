import pandas as pd

from src.Stats import Stats


class Trade:
    def __init__(self, buy_price, buy_index, profit_perc, stop_loss_perc):
        self.buy_price = buy_price
        self.sell_price = None
        self.profit = None
        self.buy_index = buy_index
        self.sell_index = None
        self.profit_perc = profit_perc
        self.stop_loss_perc = stop_loss_perc
        self.is_closed = False
        self.take_profit_val = None
        self.take_stop_loss_val = None
        self.update()

    def update(self, curr_candle=None):
        self.take_profit_val = self.buy_price * (1 + self.profit_perc)
        self.take_stop_loss_val = self.buy_price * (1 - self.stop_loss_perc)

    def try_to_close(self, curr_candle):
        if self.is_closed:
            return False
        if curr_candle.Low < self.take_stop_loss_val:
            self.close_trade(self.take_stop_loss_val, curr_candle.name)
            return True
        elif curr_candle.High > self.take_profit_val:
            self.close_trade(self.take_profit_val, curr_candle.name)
            return True
        return False

    def close_trade(self, sell_price, sell_index):
        self.sell_price = sell_price
        self.sell_index = sell_index
        self.profit = self.sell_price - self.buy_price
        self.is_closed = True


def simulate_trades(df, predictions, initial_cash=10000, profit_perc=0.02, stop_loss_perc=0.01):
    trades = []
    active_trade = None
    cash = initial_cash
    predictions.index = pd.to_datetime(predictions.index).tz_convert('Asia/Singapore')

    for idx, row in df.iterrows():
        # Update active trade if exists
        if active_trade and not active_trade.is_closed:
            if active_trade.try_to_close(row):
                cash += active_trade.profit

        # Check for new trade signal
        try:
            if predictions.at[idx, "Predictions"] == 1 and (active_trade is None or active_trade.is_closed):
                active_trade = Trade(row.Close, idx, profit_perc, stop_loss_perc)
                trades.append(active_trade)
        except KeyError:
            continue

    # Calculate statistics
    closed_trades = [t for t in trades if t.is_closed]
    winning_trades = [t for t in closed_trades if t.profit > 0]

    total_profit = sum(t.profit for t in closed_trades)
    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0

    # Calculate buy and hold return
    buy_hold_return = ((df.Close.iloc[-1] - df.Close.iloc[0]) / df.Close.iloc[0]) * 100

    stats = Stats(
        num_trades=len(closed_trades),
        win_rate=round(win_rate, 2),
        num_wins=len(winning_trades),
        num_losses=len(closed_trades) - len(winning_trades),
        perc_return=round((total_profit / initial_cash) * 100, 2),
        perc_buy_hold_return=round(buy_hold_return, 2),
        initial_cash=initial_cash,
        total_profit=round(total_profit, 2)
    )

    return trades, stats
