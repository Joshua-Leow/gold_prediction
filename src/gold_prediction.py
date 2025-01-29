import math
import os

import yfinance as yf
import pandas as pd
import finplot as fplt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def get_period(interval):
    if interval == '1d' or interval == '1w':
        return 'max'
    elif interval == '1h':
        return '730d'
    elif interval == '5m':
        return '60d'
    elif interval == '1m':
        return '8d'


def fetch_data(symbol, interval):
    period=get_period(interval)
    df = yf.download(symbol, period=period, interval=interval, ignore_tz=True, progress=False)
    df.to_csv(Path(os.path.join(os.getcwd(), f"data/{symbol}_{interval}.csv")))
    return df

def preprocess_data(df):
    df.index = pd.to_datetime(df.index, utc=True).map(lambda x: x.tz_convert('Singapore'))
    df.columns = df.columns.droplevel(1)

    from config import target_candle
    df["Tomorrow"] = df["Close"].shift(-target_candle)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    df = df.loc["1990-01-01":].copy()
    return df

def get_macd(original_df):
    macd = original_df.Close.ewm(span=12).mean() - original_df.Close.ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    original_df['macd_diff'] = macd - signal
    return original_df

def get_close_ratio_and_trend(df):
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        from config import target_candle
        df[trend_column] = df.shift(target_candle).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_column, trend_column]
    return new_predictors

def final_processing(df):
    df = df.dropna(subset=df.columns[df.columns != "Tomorrow"])
    # df = df.dropna()
    return df

def predict(train, test, predictors, model):
    from config import confidence
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= confidence] = 1
    preds[preds < 1-confidence] = 0
    preds[(preds >= 1-confidence) & (preds < confidence)] = None
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2400, step=240):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i - 10].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

def plot_finplot(df, predictions):
    original_df = df.copy()
    # Add predictions back to original dataframe
    original_df = pd.merge(original_df, predictions, on='Datetime', how='inner')
    # print(original_df)

    # Create plots
    ax, ax2 = fplt.create_plot('GOLD MACD with Trade Signals', rows=2)

    # Plot MACD
    macd = original_df.Close.ewm(span=12).mean() - original_df.Close.ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    original_df['macd_diff'] = macd - signal
    fplt.volume_ocv(original_df[['Open', 'Close', 'macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
    fplt.plot(macd, ax=ax2, legend='MACD')
    fplt.plot(signal, ax=ax2, legend='Signal')

    # Plot candlesticks
    fplt.candlestick_ochl(original_df[['Open', 'Close', 'High', 'Low']], ax=ax)

    # Add trade markers
    buy_signals = original_df[original_df['Predictions'] == 1].index
    sell_signals = original_df[original_df['Predictions'] == 0].index

    # Plot buy signals (green triangles)
    if len(buy_signals) > 0:
        buy_prices = original_df.loc[buy_signals, 'Low'].values * 0.9999
        fplt.plot(pd.Series(index=buy_signals, data=buy_prices), ax=ax, color='#0f0', marker='^',
                  legend='Buy Signal', size=5)

    # Plot sell signals (red triangles)
    if len(sell_signals) > 0:
        sell_prices = original_df.loc[sell_signals, 'High'].values * 1.0001
        fplt.plot(pd.Series(index=sell_signals, data=sell_prices), ax=ax, color='#f00', marker='v',
                  legend='Sell Signal', size=5)

    # Add volume
    axo = ax.overlay()
    fplt.volume_ocv(original_df[['Open', 'Close', 'Volume']], ax=axo)
    fplt.plot(original_df.Volume.ewm(span=24).mean(), ax=axo, color=1)

    # Add hover information
    hover_label = fplt.add_legend('', ax=ax)

    def update_legend_text(x, y):
        row = original_df.loc[pd.to_datetime(x, unit='ns', utc=True).tz_convert('Asia/Singapore')]
        fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.Open < row.Close).all() else 'a00')
        rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
        values = [row.Open, row.Close, row.High, row.Low]

        if 'Predictions' in row and not pd.isna(row.Predictions):
            if row.Predictions == 1:
                signal_type = "BUY"
            elif row.Predictions == 0:
                signal_type = "SELL"
            rawtxt += f' <span style="color:#{"0b0" if signal_type == "BUY" else "a00"}">{signal_type}</span>'
        from config import symbol, interval
        hover_label.setText(rawtxt % tuple([symbol, interval.upper()] + values))


    def update_crosshair_text(x, y, xtext, ytext):
        ytext = '%s (Close%+.2f)' % (ytext, (y - original_df.iloc[x].Close))
        return xtext, ytext

    fplt.set_mouse_callback(update_legend_text, ax=ax, when='hover')
    fplt.add_crosshair_info(update_crosshair_text, ax=ax)
    fplt.show()


def main():
    from config import symbol, interval

    print("  1. Fetching data...")
    df = fetch_data(symbol, interval)

    print("  2. Pre-processing data...")
    df = preprocess_data(df)

    print("  3. Feature Engineering...")
    df = get_macd(df)
    predictors = get_close_ratio_and_trend(df)
    # predictors.append("macd_diff")
    print(predictors)

    print("  4. Final Processing of data...")
    df = final_processing(df)

    print("  5. Preparing model...")
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

    print("  6. Making Predictions...")
    predictions = backtest(df, model, predictors)

    print(predictions["Predictions"].value_counts())
    filtered_predictions = predictions.dropna(subset=["Predictions"])

    print("  7. Getting Precision Score...")
    precision = precision_score(filtered_predictions["Target"], filtered_predictions["Predictions"])
    print("Precision Score:", precision)
    print(predictions["Target"].value_counts() / predictions.shape[0])

    print("  8. Ploting Chart...")
    plot_finplot(df, predictions)
    print("############### COMMAND TO KILL PROCESS: ################\n"
          "ps | grep gold_prediction | awk '{print $1}' | xargs kill\n"
          "#########################################################\n")


if __name__ == "__main__":
    main()