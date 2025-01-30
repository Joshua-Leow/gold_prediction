from dataclasses import dataclass

@dataclass
class Stats:
    num_trades: int
    win_rate: float
    num_wins: int
    num_losses: int
    perc_return: float
    perc_buy_hold_return: float
    initial_cash: float
    total_profit: float

    def __repr__(self) -> str:
        return f"""
        Number of Trades: {self.num_trades}
        Win Rate: {self.win_rate}%
        Return [%] : {self.perc_return}%
        Buy and Hold Return [%]: {self.perc_buy_hold_return}%
        Total Profit: ${self.total_profit}
        Initial Capital: ${self.initial_cash}
        Winning Trades: {self.num_wins}
        Losing Trades: {self.num_losses}
        """
