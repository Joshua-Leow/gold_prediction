import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.Trade import simulate_trades
from src.compare_models import compare_models_performance, ModelMetrics, get_models
from config import target_candle, symbol, interval, stop_loss_perc, profit_perc, confidence
from src.feature_engineer import get_macd_features, get_close_ratio_and_trend, add_technical_indicators, get_atr, \
    get_bollinger_bands, get_garman_klass_vol
from src.plot_chart import plot_finplot
from src.processing import fetch_data, preprocess_data, final_processing


def predict_with_confidence(model, features, confidence_threshold=0.7):
    """Make predictions with confidence threshold for long (1) and short (-1) trades"""
    proba = model.predict_proba(features)
    print(model.classes_)
    print(proba)
    long_proba = proba[:, 2]  # Probability of class 1 (long trade)
    short_proba = proba[:, 0]  # Probability of class 2 (short trade)
    predictions = np.full(len(proba), 0)  # Initialize with 0
    predictions[long_proba >= confidence_threshold] = 1  # Confident long trade
    predictions[short_proba >= confidence_threshold] = -1  # Confident short trade
    print(predictions)
    return predictions


def predict(train, test, predictors, model):
    from config import confidence
    # Ensure we're not using future data in training
    train_features = train[predictors].copy()
    train_target = (train["Future_Close"] > train["Close"]).astype(int)

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
        test_targets = (test["Future_Close"] > test["Close"]).astype(int)
        combined = pd.concat([test_targets, predictions], axis=1)
        combined.columns = ["Target", "Predictions"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)


def evaluate_models(data, predictors, start=2400, step=240):
    predictions, trades = None, None
    models = get_models()
    model_metrics = {}
    model_counter = 0

    for model_name, model in models.items():
        model_counter += 1
        print(f"  5.{model_counter} Evaluating {model_name}...")
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i - target_candle].copy()

            train_features = train[predictors].copy()
            # Define training target: 1 for long, -1 for short, 0 otherwise
            train_target = np.where(
                train["Future_Close"] > train["Close"] + (train["Close"] * profit_perc / 100)*0.1, 1,
                np.where(train["Future_Close"] < train["Close"] - (train["Close"] * profit_perc / 100)*0.1, -1, 0)
            )

            # Remove NaN values
            valid_mask = ~np.isnan(train_target)
            train_features = train_features[valid_mask]
            train_target = train_target[valid_mask]

            test = data.iloc[i:(i + step)].copy()
            # Fit and predict
            try:
                model.fit(train_features, train_target)
                preds = predict_with_confidence(model, test[predictors], confidence)
                predictions = pd.Series(preds, index=test.index)

                # Define test target for evaluation
                test_target = np.where(
                    test["Future_Close"] > test["Close"] + (test["Close"] * profit_perc / 100)*0.1, 1,
                    np.where(test["Future_Close"] < test["Close"] - (test["Close"] * profit_perc / 100)*0.1, -1, 0)
                )
                combined = pd.concat([pd.Series(test_target, index=test.index), predictions], axis=1)
                combined.columns = ["Target", "Predictions"]
                all_predictions.append(combined)
            except Exception as e:
                print(f"Error in {model_name}: {e}")
                continue

        if all_predictions:
            predictions = pd.concat(all_predictions)
            # filtered_predictions = predictions.dropna(subset=["Predictions"])
            # Filter out neutral (0) values
            filtered_predictions = predictions[predictions["Predictions"] != 0].dropna(subset=["Predictions"])
            print(f"    5.{model_counter}.1 Total number of predictions: {filtered_predictions.shape[0]}")
            print(filtered_predictions[filtered_predictions["Predictions"]==0])
            print(filtered_predictions[filtered_predictions["Predictions"]==1])
            print(filtered_predictions[filtered_predictions["Predictions"]==-1])

            if len(filtered_predictions) > 0:
                precision = precision_score(filtered_predictions["Target"],
                                            filtered_predictions["Predictions"],
                                            average="macro",
                                            zero_division=0)
                recall = recall_score(filtered_predictions["Target"],
                                      filtered_predictions["Predictions"],
                                      average="macro",
                                      zero_division=0)
                f1 = f1_score(filtered_predictions["Target"],
                              filtered_predictions["Predictions"],
                              average="macro",
                              zero_division=0)

                trades, stats = simulate_trades(
                    data,
                    filtered_predictions,
                    profit_perc=profit_perc /100,
                    stop_loss_perc=stop_loss_perc /100
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

    print("\n2. Pre-processing data...")
    df = preprocess_data(df)

    print("\n3. Feature Engineering...")
    predictors = []
    print("  3.1 Adding MACD and price features...")
    macd_predictors, df = get_macd_features(df)
    predictors += macd_predictors
    print("  3.2 Adding price ratios and trends...")
    price_predictors, df = get_close_ratio_and_trend(df)
    predictors += price_predictors
    print("  3.3 Adding ATR volatility features...")
    atr_predictors, df = get_atr(df)
    predictors += atr_predictors
    print("  3.4 Adding Bollinger Bands features...")
    bb_predictors, df = get_bollinger_bands(df)
    predictors += bb_predictors
    print("  3.5 Adding Garman Klass Volume features...")
    garman_klass_predictors, df = get_garman_klass_vol(df)
    predictors += garman_klass_predictors
    print("Features used:", predictors)

    print("\n4. Final Processing of data...")
    df = final_processing(df)

    print("\n5. Evaluating multiple models...")
    predictions, trades, metrics = evaluate_models(df, predictors)

    print("\n6. Comparing model performances...")
    trading_comparison, ml_metrics_comparison = compare_models_performance(metrics)

    print("\nTrading Performance Comparison:")
    print(trading_comparison)
    print("\nML Metrics Comparison:")
    print(ml_metrics_comparison)

    # Find best model based on return
    best_model = max(metrics.items(), key=lambda x: x[1].precision)
    print(f"\n7. Best performing model (by precision): {best_model[0]}")
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
