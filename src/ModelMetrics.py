from dataclasses import dataclass

from sklearn.metrics import precision_score, recall_score, f1_score

from config import profit_perc, stop_loss_perc
from src.Stats import Stats
from src.Trade import simulate_trades


@dataclass
class ModelMetrics:
    model_name: str
    precision: float
    recall: float
    f1: float
    trading_stats: Stats

    def __repr__(self) -> str:
        return f"""
        Model: {self.model_name}
        Precision: {self.precision:.4f}
        Recall: {self.recall:.4f}
        F1 Score: {self.f1:.4f}
        Trading Stats:
        {self.trading_stats}
        """


def compute_model_metrics(model_name, filtered_predictions, data, predictions):
    precision = precision_score(filtered_predictions["Target"], filtered_predictions["Predictions"], average="weighted",
                                zero_division=0)
    recall = recall_score(filtered_predictions["Target"], filtered_predictions["Predictions"], average="weighted",
                          zero_division=0)
    f1 = f1_score(filtered_predictions["Target"], filtered_predictions["Predictions"], average="weighted",
                  zero_division=0)

    trades, stats = simulate_trades(data, predictions, profit_perc=profit_perc / 100,
                                    stop_loss_perc=stop_loss_perc / 100)

    return trades, ModelMetrics(model_name=model_name, precision=precision, recall=recall, f1=f1, trading_stats=stats)
