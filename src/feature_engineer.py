

def get_macd_features(df, horizons=[2, 5, 60, 250, 1000]):
    """Calculate MACD features with different horizons"""
    # Calculate basic MACD
    macd = df.Close.ewm(span=12, adjust=False).mean() - df.Close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_diff'] = macd - signal

    new_predictors = []
    # Calculate MACD-based features for different horizons
    for horizon in horizons:
        # Rolling mean of MACD difference
        macd_ma = f'macd_ma_{horizon}'
        df[macd_ma] = df['macd_diff'].rolling(window=horizon, min_periods=1).mean()
        # new_predictors.append(macd_ma)

        # Rolling standard deviation of MACD difference
        macd_std = f'macd_std_{horizon}'
        df[macd_std] = df['macd_diff'].rolling(window=horizon, min_periods=1).std()
        # new_predictors.append(macd_std)

        # MACD momentum (rate of change)
        macd_mom = f'macd_mom_{horizon}'
        df[macd_mom] = df['macd_diff'].pct_change(horizon)
        # new_predictors.append(macd_mom)

        # MACD crossover signals
        macd_cross = f'macd_cross_{horizon}'
        df[macd_cross] = ((df['macd_diff'] > 0) &
                          (df['macd_diff'].shift(1) <= 0)).astype(int)
        new_predictors.append(macd_cross)

    return new_predictors, df


def get_close_ratio_and_trend(df, horizons=[2, 5, 60, 250, 1000]):
    new_predictors = []

    for horizon in horizons:
        # Calculate price ratios using only past data
        rolling_averages = df.Close.rolling(window=horizon, min_periods=1).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Close"] / rolling_averages
        new_predictors.append(ratio_column)

        # Calculate trend using price changes instead of target
        trend_column = f"Trend_{horizon}"
        price_changes = df["Close"].pct_change()
        df[trend_column] = price_changes.rolling(window=horizon, min_periods=1).mean()
        new_predictors.append(trend_column)

    return new_predictors, df
