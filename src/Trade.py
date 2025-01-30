from abc import abstractmethod, ABC

import pandas as pd

from src.Stats import Stats


class BaseTrade(ABC):
    def __init__(self, entry_price, entry_index, profit_perc, stop_loss_perc):
        self.entry_price = entry_price
        self.exit_price = None
        self.profit = None
        self.entry_index = pd.to_datetime(entry_index)
        self.exit_index = None
        self.profit_perc = profit_perc
        self.stop_loss_perc = stop_loss_perc
        self.is_closed = False
        self.take_profit_val = None
        self.take_stop_loss_val = None
        self.trade_type = None
        self.update()

    @abstractmethod
    def calculate_take_profit(self):
        pass

    @abstractmethod
    def calculate_stop_loss(self):
        pass

    @abstractmethod
    def calculate_profit(self, exit_price):
        pass

    @abstractmethod
    def should_close_at_profit(self, high_price, low_price):
        pass

    @abstractmethod
    def should_close_at_loss(self, high_price, low_price):
        pass

    def update(self, curr_candle=None):
        self.take_profit_val = self.calculate_take_profit()
        self.take_stop_loss_val = self.calculate_stop_loss()

    def try_to_close(self, curr_candle):
        if self.is_closed:
            return False

        if self.should_close_at_loss(curr_candle.High, curr_candle.Low):
            self.close_trade(self.take_stop_loss_val, curr_candle.name)
            return True
        elif self.should_close_at_profit(curr_candle.High, curr_candle.Low):
            self.close_trade(self.take_profit_val, curr_candle.name)
            return True
        return False

    def close_trade(self, exit_price, exit_index):
        self.exit_price = exit_price
        self.exit_index = pd.to_datetime(exit_index)
        self.profit = self.calculate_profit(exit_price)
        self.is_closed = True


class LongTrade(BaseTrade):
    def __init__(self, entry_price, entry_index, profit_perc, stop_loss_perc):
        super().__init__(entry_price, entry_index, profit_perc, stop_loss_perc)
        self.trade_type = "LONG"

    def calculate_take_profit(self):
        return self.entry_price * (1 + self.profit_perc)

    def calculate_stop_loss(self):
        return self.entry_price * (1 - self.stop_loss_perc)

    def calculate_profit(self, exit_price):
        return exit_price - self.entry_price

    def should_close_at_profit(self, high_price, low_price):
        return high_price > self.take_profit_val

    def should_close_at_loss(self, high_price, low_price):
        return low_price < self.take_stop_loss_val


class ShortTrade(BaseTrade):
    def __init__(self, entry_price, entry_index, profit_perc, stop_loss_perc):
        super().__init__(entry_price, entry_index, profit_perc, stop_loss_perc)
        self.trade_type = "SHORT"

    def calculate_take_profit(self):
        return self.entry_price * (1 - self.profit_perc)

    def calculate_stop_loss(self):
        return self.entry_price * (1 + self.stop_loss_perc)

    def calculate_profit(self, exit_price):
        return self.entry_price - exit_price

    def should_close_at_profit(self, high_price, low_price):
        return low_price < self.take_profit_val

    def should_close_at_loss(self, high_price, low_price):
        return high_price > self.take_stop_loss_val


def simulate_trades(df, predictions, initial_cash=10000, profit_perc=0.02, stop_loss_perc=0.01):
    trades = []
    active_trade = None
    cash = initial_cash

    try:
        # Ensure predictions has the correct index
        predictions.index = pd.to_datetime(predictions.index).tz_convert('Asia/Singapore')
        df.index = pd.to_datetime(df.index).tz_convert('Asia/Singapore')

        # Debug prints
        # print(f"DataFrame shape: {df.shape}")
        # print(f"Predictions shape: {predictions.shape}")
        # print(f"Sample of predictions:\n{predictions[predictions['Predictions'].notnull()].head()}")

        for idx, row in df.iterrows():
            if idx not in predictions.index:
                continue

            # Update active trade if exists
            if active_trade and not active_trade.is_closed:
                if active_trade.try_to_close(row):
                    cash += active_trade.profit

            # Check for new trade signal
            try:
                pred = predictions.at[idx, "Predictions"]
                if pred is not None and (active_trade is None or active_trade.is_closed):
                    if pred == 1:  # Long signal
                        active_trade = LongTrade(row.Close, idx, profit_perc, stop_loss_perc)
                        trades.append(active_trade)
                        # print(f"Created LONG trade at {idx} with entry price {row.Close}")
                    elif pred == 0:  # Short signal
                        active_trade = ShortTrade(row.Close, idx, profit_perc, stop_loss_perc)
                        trades.append(active_trade)
                        # print(f"Created SHORT trade at {idx} with entry price {row.Close}")
            except KeyError as e:
                print(f"KeyError at index {idx}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error at index {idx}: {e}")
                continue

        # Calculate statistics
        closed_trades = [t for t in trades if t.is_closed]
        winning_trades = [t for t in closed_trades if t.profit > 0]

        total_profit = sum(t.profit for t in closed_trades)
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0

        buy_hold_return = ((df.Close.iloc[-1] - df.Close.iloc[0]) / df.Close.iloc[0]) * 100 if len(df) > 0 else 0

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

        print(f"Simulation completed with {len(trades)} trades")
        return trades, stats

    except Exception as e:
        print(f"Error in simulate_trades: {e}")
        # Return empty trades list and default stats instead of None
        default_stats = Stats(
            num_trades=0,
            win_rate=0.0,
            num_wins=0,
            num_losses=0,
            perc_return=0.0,
            perc_buy_hold_return=0.0,
            initial_cash=initial_cash,
            total_profit=0.0
        )
        return [], default_stats
