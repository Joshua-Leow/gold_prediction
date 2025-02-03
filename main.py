import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.Trade import simulate_trades
from src.compare_models import compare_models_performance, ModelMetrics, get_models
from config import target_candle, symbol, interval, stop_loss_perc, profit_perc, confidence
from src.feature_engineer import get_macd_features, get_close_ratio_and_trend, add_technical_indicators
from src.plot_chart import plot_finplot
from src.processing import fetch_data, preprocess_data, final_processing


def predict_with_confidence(model, features, confidence_threshold=0.7):
    """Make predictions with confidence threshold"""
    proba = model.predict_proba(features)[:, 1]
    predictions = np.full(len(proba), np.nan)
    predictions[proba >= confidence_threshold] = 1
    predictions[proba < (1-confidence_threshold)] = 0
    return predictions


def predict(train, test, predictors, model):
    from config import confidence
    # Ensure we're not using future data in training
    train_features = train[predictors].copy()
    train_target = (train["Tomorrow"] > train["Close"]).astype(int)

    # Remove any rows where we don't have the target yet
    valid_train_mask = ~train_target.isna()
    train_features = train_features[valid_train_mask]
    train_target = train_target[valid_train_mask]

    # Fit model and make predictions
    model.fit(train_features, train_target)
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= confidence] = 1
    preds[preds < 1-confidence] = 0
    preds[(preds >= 1-confidence) & (preds < confidence)] = None
    return pd.Series(preds, index=test.index.tz_convert('Asia/Singapore'), name="Predictions")

def backtest(data, model, predictors, start=2400, step=240):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i-target_candle].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)

        # Get actual targets for the test set
        test_targets = (test["Tomorrow"] > test["Close"]).astype(int)
        combined = pd.concat([test_targets, predictions], axis=1)
        combined.columns = ["Target", "Predictions"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)


def evaluate_models(data, predictors, start=2400, step=240):
    predictions, trades = None, None
    models = get_models()
    model_metrics = {}
    scaler = StandardScaler()

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        all_predictions = []

        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i - target_candle].copy()
            test = data.iloc[i:(i + step)].copy()

            # Scale features
            train_features = scaler.fit_transform(train[predictors])
            test_features = scaler.transform(test[predictors])
            train_target = (train["Tomorrow"] > train["Close"]).astype(int)

            # Remove NaN values
            valid_mask = ~np.isnan(train_target)
            train_features = train_features[valid_mask]
            train_target = train_target[valid_mask]

            # Fit and predict
            try:
                model.fit(train_features, train_target)
                preds = predict_with_confidence(model, test_features, confidence)
                predictions = pd.Series(preds, index=test.index)
                test_targets = (test["Tomorrow"] > test["Close"]).astype(int)
                combined = pd.concat([test_targets, predictions], axis=1)
                combined.columns = ["Target", "Predictions"]
                all_predictions.append(combined)
            except Exception as e:
                print(f"Error in {model_name}: {e}")
                continue

        if all_predictions:
            predictions = pd.concat(all_predictions)
            filtered_predictions = predictions.dropna(subset=["Predictions"])

            if len(filtered_predictions) > 0:
                precision = precision_score(filtered_predictions["Target"],
                                            filtered_predictions["Predictions"],
                                            zero_division=0)
                recall = recall_score(filtered_predictions["Target"],
                                      filtered_predictions["Predictions"],
                                      zero_division=0)
                f1 = f1_score(filtered_predictions["Target"],
                              filtered_predictions["Predictions"],
                              zero_division=0)

                trades, stats = simulate_trades(
                    data,
                    predictions,
                    profit_perc=profit_perc / 100,
                    stop_loss_perc=stop_loss_perc / 100
                )

                model_metrics[model_name] = ModelMetrics(
                    model_name=model_name,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    trading_stats=stats
                )

    return predictions, trades, model_metrics


def main():
    print("1. Fetching data...")
    df = fetch_data(symbol, interval)

    print("2. Pre-processing data...")
    df = preprocess_data(df)

    # print("3. Feature Engineering...")
    # print("  3.1 Adding technical indicators...")
    # df = add_technical_indicators(df)

    print("  3.2 Adding MACD and price features...")
    macd_predictors, df = get_macd_features(df)
    price_predictors, df = get_close_ratio_and_trend(df)

    # Combine all features
    predictors = macd_predictors + price_predictors
                  # ['RSI', 'BB_width', 'Volume_ratio', 'ROC', 'ATR'])
    print("Features used:", predictors)

    print("4. Final Processing of data...")
    df = final_processing(df)

    print("5. Evaluating multiple models...")
    predictions, trades, metrics = evaluate_models(df, predictors)

    print("\n6. Comparing model performances...")
    trading_comparison, ml_metrics_comparison = compare_models_performance(metrics)

    print("\nTrading Performance Comparison:")
    print(trading_comparison)
    print("\nML Metrics Comparison:")
    print(ml_metrics_comparison)

    # Find best model based on return
    best_model = max(metrics.items(), key=lambda x: x[1].precision)
    print(f"\nBest performing model (by precision): {best_model[0]}")
    print(best_model[1])

    # print("  5. Preparing model...")
    # model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    #
    # print("  6. Making Predictions...")
    # predictions = backtest(df, model, predictors)
    #
    # print(predictions["Predictions"].value_counts())
    # filtered_predictions = predictions.dropna(subset=["Predictions"])
    #
    # print("  7. Getting Precision Score...")
    # precision = precision_score(filtered_predictions["Target"], filtered_predictions["Predictions"])
    # print("Precision Score:", precision)
    # print(predictions["Target"].value_counts() / predictions.shape[0])
    #
    # print("  8. Simulating Trades...")
    # from config import  profit_perc, stop_loss_perc
    # # for TP in np.arange(0.001, 0.0001, -0.0001):
    # #     for SL in np.arange(0.001, 0.0001, -0.0001):
    # trades, stats = simulate_trades(df, predictions, profit_perc=profit_perc/100, stop_loss_perc=stop_loss_perc/100)
    # # print(f"\nTrading Statistics for TP: {TP:.4f}, SL: {SL:.4f}")
    # print(stats)
    #
    print("  9. Plotting Chart...")
    plot_finplot(df, predictions, trades)
    # print("############### COMMAND TO KILL PROCESS: ################\n"
    #       "ps | grep gold_prediction | awk '{print $1}' | xargs kill\n"
    #       "#########################################################\n")


if __name__ == "__main__":
    main()
