import pandas as pd
import finplot as fplt


def plot_finplot(df, predictions, trades):
    original_df = df.copy()

    # Ensure predictions is a DataFrame
    if not isinstance(predictions, pd.DataFrame):
        predictions = predictions.to_frame(name="Predictions")

    # Align indices
    original_df.index = pd.to_datetime(original_df.index).tz_convert('Asia/Singapore')
    predictions.index = pd.to_datetime(predictions.index).tz_convert('Asia/Singapore')
    original_df = pd.merge(original_df, predictions, left_index=True, right_index=True, how='inner')

    # Create plots
    ax, ax2 = fplt.create_plot('Trading Chart with MACD', rows=2)

    # Plot MACD
    macd = original_df.Close.ewm(span=12).mean() - original_df.Close.ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    original_df['macd_diff'] = macd - signal
    fplt.volume_ocv(original_df[['Open', 'Close', 'macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
    fplt.plot(macd, ax=ax2, legend='MACD')
    fplt.plot(signal, ax=ax2, legend='Signal')

    # Plot candlesticks
    fplt.candlestick_ochl(original_df[['Open', 'Close', 'High', 'Low']], ax=ax)

    # Plot trade markers and profit/loss lines
    for trade in trades:
        if trade.is_closed:
            # Entry marker
            entry_color = '#0f0' if trade.trade_type == "LONG" else '#f00'
            entry_marker = '^' if trade.trade_type == "LONG" else 'v'
            fplt.plot(trade.entry_index, trade.entry_price, ax=ax,
                            marker=entry_marker, color=entry_color, size=12)

            # Exit marker
            profit_color = '#0f0' if trade.profit > 0 else '#f00'
            exit_marker = 'v' if trade.trade_type == "LONG" else '^'
            fplt.plot(trade.exit_index, trade.exit_price, ax=ax,
                            marker=exit_marker, color=profit_color, size=12)

            # Connection line
            fplt.add_line((trade.entry_index, trade.entry_price),
                          (trade.exit_index, trade.exit_price),
                          ax=ax, color=profit_color, style='--')

            # Plot take profit and stop loss levels
            start_idx = original_df.index.get_indexer([trade.entry_index])[0]
            end_idx = original_df.index.get_indexer([trade.exit_index])[0]

            # Draw take profit line
            fplt.add_line((trade.entry_index, trade.take_profit_val),
                          (trade.exit_index, trade.take_profit_val),
                          ax=ax, color='#0f0', style=':')

            # Draw stop loss line
            fplt.add_line((trade.entry_index, trade.take_stop_loss_val),
                          (trade.exit_index, trade.take_stop_loss_val),
                          ax=ax, color='#f00', style=':')

    # Add volume
    axo = ax.overlay()
    fplt.volume_ocv(original_df[['Open', 'Close', 'Volume']], ax=axo)
    fplt.plot(original_df.Volume.ewm(span=24).mean(), ax=axo, color=1)

    # Add hover information
    hover_label = fplt.add_legend('', ax=ax)

    def update_legend_text(x, y):
        timestamp = pd.to_datetime(x, unit='ns', utc=True).tz_convert('Asia/Singapore')
        try:
            row = original_df.loc[timestamp]
            fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.Open < row.Close).all() else 'a00')
            rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
            values = [row.Open, row.Close, row.High, row.Low]

            # Find active trade at this timestamp
            active_trade = next((t for t in trades
                                 if t.entry_index <= timestamp and
                                 (not t.is_closed or t.exit_index >= timestamp)), None)

            if active_trade:
                if active_trade.is_closed:
                    profit = active_trade.profit
                    rawtxt += f' <span style="color:#{"0b0" if profit > 0 else "a00"}">{active_trade.trade_type} Profit: ${profit:.2f}</span>'
                else:
                    rawtxt += f' <span style="color:#{"0b0" if active_trade.trade_type == "LONG" else "f00"}">Active {active_trade.trade_type}</span>'
                    rawtxt += f' <span style="color:#0b0">TP: {active_trade.take_profit_val:.2f}</span>'
                    rawtxt += f' <span style="color:#f00">SL: {active_trade.take_stop_loss_val:.2f}</span>'

            from config import symbol, interval
            hover_label.setText(rawtxt % tuple([symbol, interval.upper()] + values))

        except (KeyError, IndexError) as e:
            print(f"Error updating legend: {e}")

    def update_crosshair_text(x, y, xtext, ytext):
        try:
            close_price = original_df.iloc[x].Close
            ytext = '%s (Close%+.2f)' % (ytext, (y - close_price))
        except IndexError:
            pass
        return xtext, ytext

    fplt.set_mouse_callback(update_legend_text, ax=ax, when='hover')
    fplt.add_crosshair_info(update_crosshair_text, ax=ax)
    fplt.show()
