from src.compare_models import compare_models_performance, get_models, evaluate_model
from src.ModelMetrics import compute_model_metrics
from config import symbol, interval
from src.feature_engineer import get_macd_features, get_close_ratio_and_trend, get_atr, \
    get_bollinger_bands, get_garman_klass_vol
from src.plot_chart import plot_finplot
from src.processing import fetch_data, preprocess_data, final_processing


# def predict(train, test, predictors, model):
#     from config import confidence
#     # Ensure we're not using future data in training
#     train_features = train[predictors].copy()
#     train_target = (train["Future_Close"] > train["Close"]).astype(int)
#
#     # Remove any rows where we don't have the target yet
#     valid_train_mask = ~train_target.isna()
#     train_features = train_features[valid_train_mask]
#     train_target = train_target[valid_train_mask]
#
#     # Fit model and make predictions
#     model.fit(train_features, train_target)
#     preds = model.predict_proba(test[predictors])[:, 1]
#     preds[preds >= confidence] = 1
#     preds[preds < 1-confidence] = 0
#     preds[(preds >= 1-confidence) & (preds < confidence)] = None
#     return pd.Series(preds, index=test.index.tz_convert('Asia/Singapore'), name="Predictions")
#
# def backtest(data, model, predictors, start=2400, step=240):
#     all_predictions = []
#
#     for i in range(start, data.shape[0], step):
#         train = data.iloc[0:i-target_candle].copy()
#         test = data.iloc[i:(i+step)].copy()
#         predictions = predict(train, test, predictors, model)
#
#         # Get actual targets for the test set
#         test_targets = (test["Future_Close"] > test["Close"]).astype(int)
#         combined = pd.concat([test_targets, predictions], axis=1)
#         combined.columns = ["Target", "Predictions"]
#
#         all_predictions.append(combined)
#     return pd.concat(all_predictions)


# def evaluate_models(data, predictors, start=2400, step=240):
#     predictions, trades = None, None
#     models = get_models()
#     model_metrics = {}
#     model_counter = 0
#
#     for model_name, model in models.items():
#         model_counter += 1
#         print(f"  5.{model_counter} Evaluating {model_name}...")
#         all_predictions = []
#         for i in range(start, data.shape[0], step):
#             print(f"    Step {i}/{data.shape[0]} ({i/data.shape[0]*100:.2f}%): ", end='')
#             train = data.iloc[0:i - target_candle].copy()
#
#             train_features = train[predictors].copy()
#             # Define training target: 1 for long, -1 for short, 0 otherwise
#             train_target = np.where(
#                 train["Future_Close"] > train["Close"] + (train["Close"] * profit_perc / 100), 1,
#                 np.where(train["Future_Close"] < train["Close"] - (train["Close"] * profit_perc / 100), -1, 0)
#             )
#
#             # Remove NaN values
#             valid_mask = ~np.isnan(train_target)
#             train_features = train_features[valid_mask]
#             train_target = train_target[valid_mask]
#
#             test = data.iloc[i:(i + step)].copy()
#             # Fit and predict
#             try:
#                 model.fit(train_features, train_target)
#                 preds = predict_with_confidence(model, test[predictors], confidence)
#                 predictions = pd.Series(preds, index=test.index)
#
#                 # Define test target for evaluation
#                 test_target = np.where(
#                     test["Future_Close"] > test["Close"] + (test["Close"] * profit_perc / 100), 1,
#                     np.where(test["Future_Close"] < test["Close"] - (test["Close"] * profit_perc / 100), -1, 0)
#                 )
#                 combined = pd.concat([pd.Series(test_target, index=test.index), predictions], axis=1)
#                 combined.columns = ["Target", "Predictions"]
#                 all_predictions.append(combined)
#             except Exception as e:
#                 print(f"Error in {model_name}: {e}")
#                 continue
#
#         if all_predictions:
#             predictions = pd.concat(all_predictions)
#             # Filter out neutral (0) values
#             filtered_predictions = predictions[predictions["Predictions"] != 0].dropna(subset=["Predictions"])
#             print(f"    5.{model_counter}.1 Total number of predictions: {filtered_predictions.shape[0]}")
#             print(filtered_predictions[filtered_predictions["Predictions"]==0])
#             print(filtered_predictions[filtered_predictions["Predictions"]==1])
#             print(filtered_predictions[filtered_predictions["Predictions"]==-1])
#
#             if len(filtered_predictions) > 0:
#                 # Final Thoughts:
#                 # Focus on Recall if you want to capture most trading opportunities, even at the cost of some false positives.
#                 # Focus on Precision if you want to avoid false trades, even at the cost of missing some real ones.
#                 # F1 Score is useful if you want both accuracy & trade capture balance.
#                 # 🔹 If precision is high but recall is low, your model predicts accurately but misses too many trades.
#                 # 🔹 If recall is high but precision is low, your model trades too often and incorrectly.
#                 precision = precision_score(filtered_predictions["Target"],
#                                             filtered_predictions["Predictions"],
#                                             average="weighted",
#                                             zero_division=0)
#                 recall = recall_score(filtered_predictions["Target"],
#                                       filtered_predictions["Predictions"],
#                                       average="weighted",
#                                       zero_division=0)
#                 f1 = f1_score(filtered_predictions["Target"],
#                               filtered_predictions["Predictions"],
#                               average="weighted",
#                               zero_division=0)
#
#                 trades, stats = simulate_trades(
#                     data,
#                     predictions,
#                     profit_perc=profit_perc /100,
#                     stop_loss_perc=stop_loss_perc /100
#                 )
#
#                 model_metrics[model_name] = ModelMetrics(
#                     model_name=model_name,
#                     precision=precision,
#                     recall=recall,
#                     f1=f1,
#                     trading_stats=stats
#                 )
#
#     return predictions, trades, model_metrics


def evaluate_models(data, predictors, start=4800, step=240):
    models = get_models()
    model_metrics, model_predictions, model_trades = {}, {}, {}

    for model_counter, (model_name, model) in enumerate(models.items(), start=1):
        print(f"  5.{model_counter} Evaluating {model_name}...")
        predictions = evaluate_model(model, model_name, data, predictors, start, step, model_counter)

        if predictions is not None:
            filtered_predictions = predictions[predictions["Predictions"] != 0].dropna(subset=["Predictions"])
            trades, model_metrics[model_name] = compute_model_metrics(model_name, filtered_predictions, data, predictions)
            model_predictions[model_name] = predictions
            model_trades[model_name] = trades

    return model_predictions, model_trades, model_metrics


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
    model_predictions, model_trades, model_metrics = evaluate_models(df, predictors)

    print("\n6. Comparing model performances...")
    trading_comparison, ml_metrics_comparison = compare_models_performance(model_metrics)

    print("\n\t6.1 Trading Performance Comparison:")
    print(trading_comparison)
    print("\n\t6.2 ML Metrics Comparison:")
    print(ml_metrics_comparison)

    # Find best model based on return
    best_model_name, best_model_metrics = max(model_metrics.items(), key=lambda x: x[1].trading_stats.perc_return)
    print(f"\n\t6.3 Best performing model (by return): {best_model_name}")
    print(best_model_metrics)

    # Get predictions and trades of the best model
    predictions_of_best_model = model_predictions[best_model_name]
    trades_of_best_model = model_trades[best_model_name]

    print("\n7. Plotting Chart...")
    plot_finplot(df, predictions_of_best_model, trades_of_best_model)
    # print("############### COMMAND TO KILL PROCESS: ################\n"
    #       "ps | grep gold_prediction | awk '{print $1}' | xargs kill\n"
    #       "#########################################################\n")


if __name__ == "__main__":
    main()
