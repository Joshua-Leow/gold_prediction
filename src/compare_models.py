import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from xgboost import XGBClassifier

from config import confidence, target_candle, define_target_labels


def get_models():
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_split=50,
            max_depth=10,
            random_state=1                     )
        # ,"XGBoost": XGBClassifier(
        #     n_estimators=300,
        #     learning_rate=0.05,
        #     max_depth=6,
        #     min_child_weight=2,
        #     random_state=1       )
        # ,"LightGBM": LGBMClassifier(
        #     n_estimators=300,
        #     learning_rate=0.05,
        #     max_depth=6,
        #     num_leaves=31,
        #     random_state=1         )
        # ,"Gradient Boosting": GradientBoostingClassifier(
        #     n_estimators=300,
        #     learning_rate=0.05,
        #     max_depth=6,
        #     min_samples_split=50,
        #     random_state=1                              )
    }
    return models


def prepare_training_data(data, predictors, i):
    train = data.iloc[0:i - target_candle].copy()
    train_features = train[predictors].copy()
    train_target = define_target_labels(train)
    # print(np.unique(train_target))

    valid_mask = ~np.isnan(train_target)
    return train_features[valid_mask], train_target[valid_mask]


def predict_with_confidence(model, features, confidence_threshold=0.7):
    """Make predictions with confidence threshold for long (1) and short (-1) trades"""
    proba = model.predict_proba(features)
    print(list(model.classes_))
    # print(model.feature_importances_)
    # print(proba)
    long_proba, short_proba = None, None
    if len(list(model.classes_)) == 3:
        long_proba = proba[:, 2]  # Probability of class 1 (long trade)
        short_proba = proba[:, 0]  # Probability of class 2 (short trade)
    elif 1 in list(model.classes_):
        long_proba = proba[:, 1]  # Probability of class 1 (long trade)
    elif -1 in list(model.classes_):
        short_proba = proba[:, 0]  # Probability of class 2 (short trade)

    predictions = np.full(len(proba), 0)  # Initialize with 0
    if long_proba is not None: predictions[long_proba >= confidence_threshold] = 1  # Confident long trade
    if short_proba is not None: predictions[short_proba >= confidence_threshold] = -1  # Confident short trade

    # Count occurrences of each value
    count_minus_1 = np.count_nonzero(predictions == -1)
    count_0 = np.count_nonzero(predictions == 0)
    count_1 = np.count_nonzero(predictions == 1)
    print(f"Count of SHORT: {count_minus_1} , NEUTRAL: {count_0} , LONG: {count_1} predictions")
    return predictions




def create_combined_predictions(test, preds):
    test_target = define_target_labels(test)
    return pd.DataFrame({"Target": test_target, "Predictions": preds}, index=test.index)


def evaluate_model(model, model_name, data, predictors, start, step, model_counter):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        print(f"    Step {i}/{data.shape[0]} ({i / data.shape[0] * 100:.2f}%): ", end='')
        train, train_target = prepare_training_data(data, predictors, i)
        test = data.iloc[i:(i + step)].copy()

        model.fit(train, train_target)
        # print(test[predictors])
        preds = predict_with_confidence(model, test[predictors], confidence)
        try:
            combined = create_combined_predictions(test, preds)
            all_predictions.append(combined)
        except Exception as e:
            print(f"Error in {model_name}: {e}")
            continue

    return pd.concat(all_predictions) if all_predictions else None


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
