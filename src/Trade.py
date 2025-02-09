from abc import abstractmethod, ABC
from datetime import timedelta

import pandas as pd

from config import gap_between_trades, target_candle, max_positions
from src.Stats import Stats


class BaseTrade(ABC):
    def __init__(self, entry_dollar_size, entry_price, entry_index, profit_perc, stop_loss_perc):
        self.entry_dollar_size = entry_dollar_size
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
    def calculate_take_profit_val(self):
        pass

    @abstractmethod
    def calculate_stop_loss_val(self):
        pass

    @abstractmethod
    def calculate_profit(self, entry_dollar_size, exit_price):
        pass

    @abstractmethod
    def should_close_at_profit(self, high_price, low_price):
        pass

    @abstractmethod
    def should_close_at_loss(self, high_price, low_price):
        pass

    def update(self, curr_candle=None):
        self.take_profit_val = self.calculate_take_profit_val()
        self.take_stop_loss_val = self.calculate_stop_loss_val()

    def try_to_close(self, curr_candle):
        if self.is_closed:
            return False
        if self.should_close_at_loss(curr_candle.High, curr_candle.Low):
            print("\t\t\t\t\t\t\t\t\t\t\t", end=' ')
            self.close_trade(self.take_stop_loss_val, curr_candle.name)
            print("(Loss)")
            return True
        elif self.should_close_at_profit(curr_candle.High, curr_candle.Low):
            print("\t\t\t\t\t\t\t\t\t\t\t", end=' ')
            self.close_trade(self.take_profit_val, curr_candle.name)
            print("(Profit)")
            return True
        return False

    def close_trade(self, exit_price, exit_index):
        print(f"Trade Closed at price: {exit_price:.2f}", end=' ')
        self.exit_price = exit_price
        self.exit_index = pd.to_datetime(exit_index)
        self.profit = self.calculate_profit(self.entry_dollar_size, exit_price)
        self.is_closed = True


class LongTrade(BaseTrade):
    def __init__(self, entry_dollar_size, entry_price, entry_index, profit_perc, stop_loss_perc):
        super().__init__(entry_dollar_size, entry_price, entry_index, profit_perc, stop_loss_perc)
        self.trade_type = "LONG"

    def calculate_take_profit_val(self):
        return self.entry_price * (1 + self.profit_perc)

    def calculate_stop_loss_val(self):
        return self.entry_price * (1 - self.stop_loss_perc)

    def calculate_profit(self, entry_dollar_size, exit_price):
        perc_change = (exit_price - self.entry_price) / self.entry_price
        return entry_dollar_size * perc_change

    def should_close_at_profit(self, high_price, low_price):
        return high_price > self.take_profit_val

    def should_close_at_loss(self, high_price, low_price):
        return low_price < self.take_stop_loss_val


class ShortTrade(BaseTrade):
    def __init__(self, entry_dollar_size, entry_price, entry_index, profit_perc, stop_loss_perc):
        super().__init__(entry_dollar_size, entry_price, entry_index, profit_perc, stop_loss_perc)
        self.trade_type = "SHORT"

    def calculate_take_profit_val(self):
        return self.entry_price * (1 - self.profit_perc)

    def calculate_stop_loss_val(self):
        return self.entry_price * (1 + self.stop_loss_perc)

    def calculate_profit(self, entry_dollar_size, exit_price):
        perc_change = (self.entry_price - exit_price) / self.entry_price
        return entry_dollar_size * perc_change

    def should_close_at_profit(self, high_price, low_price):
        return low_price < self.take_profit_val

    def should_close_at_loss(self, high_price, low_price):
        return high_price > self.take_stop_loss_val


# class TrailingStopMixin:
#     """Mixin class to add trailing stop loss functionality to trade classes."""
#     def __init__(self, *args, trail_percent=0.01, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.trail_percent = trail_percent
#         self.highest_price = self.entry_price
#         self.lowest_price = self.entry_price
#         self.trailing_stop_price = None
#
#     @abstractmethod
#     def update_trailing_stop(self, current_price):
#         pass
#
#     def update(self, curr_candle=None):
#         super().update(curr_candle)
#         if curr_candle is not None:
#             self.update_trailing_stop(curr_candle.Close)
#
#
# class TrailingLongTrade(TrailingStopMixin, LongTrade):
#     """Long trade with trailing stop loss that moves up as price increases."""
#     def __init__(self, entry_price, entry_index, profit_perc, stop_loss_perc, trail_percent=0.01):
#         super().__init__(entry_price, entry_index, profit_perc, stop_loss_perc, trail_percent=trail_percent)
#         self.trailing_stop_price = self.entry_price * (1 - self.trail_percent)
#
#     def update_trailing_stop(self, current_price):
#         if current_price > self.highest_price:
#             self.highest_price = current_price
#             self.trailing_stop_price = self.highest_price * (1 - self.trail_percent)
#
#     def should_close_at_loss(self, high_price, low_price):
#         return low_price < self.trailing_stop_price or super().should_close_at_loss(high_price, low_price)
#         # return low_price < self.trailing_stop_price
#
#
# class TrailingShortTrade(TrailingStopMixin, ShortTrade):
#     """Short trade with trailing stop loss that moves down as price decreases."""
#     def __init__(self, entry_price, entry_index, profit_perc, stop_loss_perc, trail_percent=0.01):
#         super().__init__(entry_price, entry_index, profit_perc, stop_loss_perc, trail_percent=trail_percent)
#         self.trailing_stop_price = self.entry_price * (1 + self.trail_percent)
#
#     def update_trailing_stop(self, current_price):
#         if current_price < self.lowest_price:
#             self.lowest_price = current_price
#             self.trailing_stop_price = self.lowest_price * (1 + self.trail_percent)
#
#     def should_close_at_loss(self, high_price, low_price):
#         return high_price > self.trailing_stop_price or super().should_close_at_loss(high_price, low_price)
#         # return high_price > self.trailing_stop_price
#
#
# class ScaledTrade(BaseTrade):
#     """Base class for trades that implement position scaling (scaling in/out)."""
#     def __init__(self, entry_price, entry_index, profit_perc, stop_loss_perc, num_scales=3):
#         super().__init__(entry_price, entry_index, profit_perc, stop_loss_perc)
#         self.num_scales = num_scales
#         self.scale_points = []
#         self.current_scale = 0
#         self.position_size = 1.0 / num_scales
#         self.setup_scale_points()
#
#     @abstractmethod
#     def setup_scale_points(self):
#         pass
#
#     @abstractmethod
#     def check_scale_points(self, current_price):
#         pass
#
#
# class ScaledLongTrade(ScaledTrade, LongTrade):
#     """Long trade that scales into position at predetermined price points."""
#     def setup_scale_points(self):
#         # Example: Scale in at progressively lower prices
#         scale_factor = self.stop_loss_perc / (self.num_scales + 1)
#         self.scale_points = [
#             self.entry_price * (1 - (i + 1) * scale_factor)
#             for i in range(self.num_scales - 1)
#         ]
#
#     def check_scale_points(self, current_price):
#         if self.current_scale < len(self.scale_points):
#             if current_price <= self.scale_points[self.current_scale]:
#                 self.current_scale += 1
#                 self.position_size += 1.0 / self.num_scales
#                 return True
#         return False
#
#
# class ScaledShortTrade(ScaledTrade, ShortTrade):
#     """Short trade that scales into position at predetermined price points."""
#     def setup_scale_points(self):
#         # Example: Scale in at progressively higher prices
#         scale_factor = self.stop_loss_perc / (self.num_scales + 1)
#         self.scale_points = [
#             self.entry_price * (1 + (i + 1) * scale_factor)
#             for i in range(self.num_scales - 1)
#         ]
#
#     def check_scale_points(self, current_price):
#         if self.current_scale < len(self.scale_points):
#             if current_price >= self.scale_points[self.current_scale]:
#                 self.current_scale += 1
#                 self.position_size += 1.0 / self.num_scales
#                 return True
#         return False


def simulate_trades(df, predictions, initial_cash=10000, profit_perc=0.02, stop_loss_perc=0.01):
    trades = []
    # active_trade = None
    trade_objects = {}
    cash = initial_cash
    rows_since_last_trade_closed = float('inf')  # Start with a high value

    try:
        # Ensure predictions has the correct index
        predictions.index = pd.to_datetime(predictions.index).tz_convert('Asia/Singapore')
        df.index = pd.to_datetime(df.index).tz_convert('Asia/Singapore')

        # Debug prints
        # print(f"DataFrame shape: {df.shape}")
        # print(f"Predictions shape: {predictions.shape}")
        # print(f"Sample of predictions:\n{predictions[predictions['Predictions'].notnull()].head()}")

        for idx, row in df.iterrows():
            # print(f"idx: {idx}, row: {row.Close}")
            if idx not in predictions.index:
                continue

            # Update active trade if exists
            for trade_name, active_trade in trade_objects.items():
                if active_trade and not active_trade.is_closed:
                    # print(f"{trade_name} Entry index: {active_trade.entry_index}\n\t\t\t\t\t\t\t\t\t Current index: {idx}")
                    # Close trade if trade opened for more than target_candle duration
                    current_row = df.index.get_loc(idx)
                    entry_row = df.index.get_loc(active_trade.entry_index)
                    row_difference = abs(current_row - entry_row)
                    if row_difference > target_candle:
                        print("\t\t\t\t\t\t\t\t\t\t\t", end=' ')
                        active_trade.close_trade(row.Close, row.name)
                        print(f"(Exceeded {target_candle} candles)")
                        cash += active_trade.profit
                        # trades.append(active_trade)
                        # trade_objects.pop(trade_name)
                    elif active_trade.try_to_close(row):
                        cash += active_trade.profit
                        # trades.append(active_trade)
                        # trade_objects.pop(trade_name)


            # If there's no active trade or the previous trade is closed
            # if active_trade is None or active_trade.is_closed:
            #     rows_since_last_trade_closed += 1

            # Check for new trade signal
            try:
                pred = predictions.at[idx, "Predictions"]
                if (pred is not None and pred != 0):
                    # and (active_trade is None or active_trade.is_closed)
                    # and rows_since_last_trade_closed > gap_between_trades):
                    if (pred == 1  # Long signal
                        and sum(not trade.is_closed for trade in trade_objects.values()) < max_positions):  #open trades < max_positions
                        trade_name = f"active_long_{idx}"
                        trade_objects[trade_name] = LongTrade(cash/max_positions, row.Close, idx, profit_perc, stop_loss_perc)
                        # active_trade = TrailingLongTrade(row.Close, idx, profit_perc, stop_loss_perc, trail_percent=stop_loss_perc/100)
                        # active_trade = ScaledLongTrade(row.Close, idx, profit_perc, stop_loss_perc, num_scales=3)
                        rows_since_last_trade_opened = 0
                        trades.append(trade_objects[trade_name])
                        print(f"  Created LONG  trade at {idx} with entry price {row.Close:.2f}")
                    elif (pred == -1 # Short signal
                        and sum(not trade.is_closed for trade in trade_objects.values()) < max_positions):  #open trades < max_positions
                        trade_name = f"active_short_{idx}"
                        trade_objects[trade_name] = ShortTrade(cash/max_positions, row.Close, idx, profit_perc, stop_loss_perc)
                        # active_trade = TrailingShortTrade(row.Close, idx, profit_perc, stop_loss_perc, trail_percent=stop_loss_perc/100)
                        # active_trade = ScaledShortTrade(row.Close, idx, profit_perc, stop_loss_perc, num_scales=3)
                        rows_since_last_trade_opened = 0
                        trades.append(trade_objects[trade_name])
                        print(f"  Created SHORT trade at {idx} with entry price {row.Close:.2f}")
            except KeyError as e:
                print(f"KeyError at index {idx}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error at index {idx}: {e}")
                continue

        # Calculate statistics
        closed_trades = [t for t in trades if t.is_closed]
        winning_trades = [t for t in closed_trades if t.profit > 0]
        # total_profit = sum(t.profit for t in closed_trades)
        total_profit = cash - initial_cash
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0

        total_num_days = int((df.index[-1] - df.index[0]) / timedelta(days=1))

        perc_return = round((total_profit / initial_cash) * 100, 2)

        buy_hold_return = ((df.Close.iloc[-1] - df.Close.iloc[0]) / df.Close.iloc[0]) * 100 if len(df) > 0 else 0

        stats = Stats(
            num_trades=len(closed_trades),
            win_rate=round(win_rate, 2),
            num_wins=len(winning_trades),
            num_losses=len(closed_trades) - len(winning_trades),
            per_annum_return=round( perc_return /(total_num_days/365), 2),
            perc_return=perc_return,
            perc_buy_hold_return=round(buy_hold_return, 2),
            initial_cash=initial_cash,
            total_profit=round(total_profit, 2)
        )

        print(f"      Trading Simulation completed with {len(trades)} trades")
        return trades, stats

    except Exception as e:
        print(f"Error in simulate_trades: {e}")
        # Return empty trades list and default stats instead of None
        default_stats = Stats(
            num_trades=0,
            win_rate=0.0,
            num_wins=0,
            num_losses=0,
            per_annum_return=0.0,
            perc_return=0.0,
            perc_buy_hold_return=0.0,
            initial_cash=initial_cash,
            total_profit=0.0
        )
        return [], default_stats
