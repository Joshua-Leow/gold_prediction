import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.Trade import simulate_trades
from src.compare_models import compare_models_performance, ModelMetrics, get_models
from config import target_candle, symbol, interval, stop_loss_perc, profit_perc
from src.feature_engineer import get_macd_features, get_close_ratio_and_trend
from src.processing import fetch_data, preprocess_data, final_processing


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
    models = get_models()
    model_metrics = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        predictions = backtest(data, model, predictors, start, step)

        # Calculate metrics
        filtered_predictions = predictions.dropna(subset=["Predictions"])
        precision = precision_score(filtered_predictions["Target"], filtered_predictions["Predictions"])
        recall = recall_score(filtered_predictions["Target"], filtered_predictions["Predictions"])
        f1 = f1_score(filtered_predictions["Target"], filtered_predictions["Predictions"])

        # Simulate trades
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

    return model_metrics


def main():
    print("1. Fetching data...")
    df = fetch_data(symbol, interval)

    print("2. Pre-processing data...")
    df = preprocess_data(df)

    print("3. Feature Engineering...")
    macd_predictors, df = get_macd_features(df)
    price_predictors, df = get_close_ratio_and_trend(df)
    predictors = macd_predictors + price_predictors
    print("Features used:", predictors)

    print("4. Final Processing of data...")
    df = final_processing(df)

    print("5. Evaluating multiple models...")
    metrics = evaluate_models(df, predictors)

    print("\n6. Comparing model performances...")
    trading_comparison, ml_metrics_comparison = compare_models_performance(metrics)

    print("\nTrading Performance Comparison:")
    print(trading_comparison)
    print("\nML Metrics Comparison:")
    print(ml_metrics_comparison)

    # Find best model based on return
    best_model = max(metrics.items(), key=lambda x: x[1].trading_stats.perc_return)
    print(f"\nBest performing model (by return): {best_model[0]}")
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
    # print("  9. Plotting Chart...")
    # plot_finplot(df, predictions, trades)
    # print("############### COMMAND TO KILL PROCESS: ################\n"
    #       "ps | grep gold_prediction | awk '{print $1}' | xargs kill\n"
    #       "#########################################################\n")


if __name__ == "__main__":
    main()
