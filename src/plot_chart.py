import pandas as pd
import finplot as fplt


def plot_finplot(df, predictions, trades):
    original_df = df.copy()
    original_df = pd.merge(original_df, predictions, on='Datetime', how='inner')

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

    # Plot trade markers and profit/loss lines
    for trade in trades:
        if trade.is_closed:
            # Entry marker
            entry_color = '#0f0' if trade.trade_type == "LONG" else '#f00'
            fplt.plot(trade.entry_index, trade.entry_price, ax=ax,
                            marker='^' if trade.trade_type == "LONG" else 'v',
                            color=entry_color, size=10)

            # Exit marker
            profit_color = '#0f0' if trade.profit > 0 else '#f00'
            fplt.plot(trade.exit_index, trade.exit_price, ax=ax,
                            marker='v' if trade.trade_type == "LONG" else '^',
                            color=profit_color, size=10)

            # Connection line
            fplt.add_line((trade.entry_index, trade.entry_price),
                          (trade.exit_index, trade.exit_price),
                          ax=ax, color=profit_color, style='--')

    # Add volume
    axo = ax.overlay()
    fplt.volume_ocv(original_df[['Open', 'Close', 'Volume']], ax=axo)
    fplt.plot(original_df.Volume.ewm(span=24).mean(), ax=axo, color=1)

    # Add hover information
    hover_label = fplt.add_legend('', ax=ax)

    def update_legend_text(x, y):
        timestamp = pd.to_datetime(x, unit='ns', utc=True).tz_convert('Asia/Singapore')
        row = original_df.loc[timestamp]
        fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.Open < row.Close).all() else 'a00')
        rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
        values = [row.Open, row.Close, row.High, row.Low]

        # Add trade information if available
        active_trade = next((t for t in trades if t.entry_index <= timestamp and
                             (not t.is_closed or t.entry_index >= timestamp)), None)
        if active_trade:
            if active_trade.is_closed:
                profit = active_trade.profit
                rawtxt += f' <span style="color:#{"0b0" if profit > 0 else "a00"}">Profit: ${profit:.2f}</span>'
            else:
                rawtxt += ' <span style="color:#0b0">Active Trade</span>'

        from config import symbol, interval
        hover_label.setText(rawtxt % tuple([symbol, interval.upper()] + values))


    def update_crosshair_text(x, y, xtext, ytext):
        ytext = '%s (Close%+.2f)' % (ytext, (y - original_df.iloc[x].Close))
        return xtext, ytext

    fplt.set_mouse_callback(update_legend_text, ax=ax, when='hover')
    fplt.add_crosshair_info(update_crosshair_text, ax=ax)
    fplt.show()
