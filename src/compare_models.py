from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from dataclasses import dataclass
import pandas as pd

from src.Stats import Stats


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


def get_models():
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_split=50,
            max_depth=10,
            random_state=1
        )
        # ,"XGBoost": XGBClassifier(
        #     n_estimators=200,
        #     learning_rate=0.05,
        #     max_depth=6,
        #     min_child_weight=2,
        #     random_state=1
        # )
        # ,"LightGBM": LGBMClassifier(
        #     n_estimators=200,
        #     learning_rate=0.05,
        #     max_depth=6,
        #     num_leaves=31,
        #     random_state=1
        # )
        # ,"Gradient Boosting": GradientBoostingClassifier(
        #     n_estimators=200,
        #     learning_rate=0.05,
        #     max_depth=6,
        #     min_samples_split=50,
        #     random_state=1
        # )
    }
    return models


def compare_models_performance(metrics):
    # Create comparison DataFrames
    model_names = list(metrics.keys())

    # Trading performance comparison
    trading_data = {
        'Win Rate (%)': [m.trading_stats.win_rate for m in metrics.values()],
        'Return (%)': [m.trading_stats.perc_return for m in metrics.values()],
        'Total Profit ($)': [m.trading_stats.total_profit for m in metrics.values()],
        'Number of Trades': [m.trading_stats.num_trades for m in metrics.values()]
    }

    # ML metrics comparison
    ml_metrics_data = {
        'Precision': [m.precision for m in metrics.values()],
        'Recall': [m.recall for m in metrics.values()],
        'F1 Score': [m.f1 for m in metrics.values()]
    }

    trading_df = pd.DataFrame(trading_data, index=model_names)
    ml_metrics_df = pd.DataFrame(ml_metrics_data, index=model_names)

    return trading_df, ml_metrics_df