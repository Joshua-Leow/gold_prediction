# IGLOF × Gold Prediction — TradingView Pine v6 strategy setup

## What this package is

A pair of **Pine Script trading strategies** combining the IGLOF V3.1 sweep/reclaim → MSS → order-block retest architecture with technical features mentioned in [Joshua-Leow/gold_prediction](https://github.com/Joshua-Leow/gold_prediction): MACD, RSI, Bollinger Bands, ATR, ROC, and Garman–Klass OHLC volatility.

**This is an original Pine trading-rule adaptation, NOT a port of the repository's trained Random Forest / XGBoost / LightGBM / Gradient Boosting model or model predictions.** The repository's Python code and its model evaluation pipeline remain separate. Backtest results are not supplied because the Pine code cannot be run in a live TradingView Pine compiler here.

The two editions are:

- `IGLOF_GoldPrediction_Standard_Strategy.pine`: chart candle/volume proxy, supports typical XAUUSD CFD feeds. No actual trade-side delta or DOM.
- `IGLOF_GoldPrediction_Footprint_Strategy.pine`: uses one TradingView `request.footprint()` call for categorized buying/selling volume, delta, POC, and diagonal row imbalances. Requires eligible **Premium/Ultimate** access and supported data. Not exchange Level 2 DOM, and not automatically imported from a separately displayed footprint chart.

## Install into TradingView

1. Open [TradingView](https://www.tradingview.com/chart/) and find your preferred `XAUUSD` provider or a tradable COMEX gold futures contract.
2. Select a **5-minute chart** for initial testing, optionally compare 15-minute. Use **standard candles** (not synthetic Renko/Heikin Ashi) for realistic Strategy Tester executions.
3. Open **Pine Editor** at the bottom of the chart, create a new blank strategy, and paste the **entire** `.pine` source for your selected edition.
4. Save it as `IGLOF x Gold Prediction - Standard Strategy` or `- Footprint Strategy` and press **Add to chart**.
5. Remove earlier IGLOF indicator editions to reduce overlapping marks.
6. Open the **Strategy Tester** tab → **Overview / Performance Summary / List of Trades**.
7. Open **Settings → Inputs** to configure your session, sweep, technical votes, event timestamps, and order sizing.
8. Open **Settings → Properties** and verify your starting equity, instrument-specific commission, simulated slippage (ticks) and margin settings. The initial source defaults are illustrative only.
9. Enable **Bar Magnifier** under Strategy Properties if your subscription supports it; compare results with and without it. Verify history range and realistic fills.
10. Set alerts using **Create Alert → your strategy → Order fills and alert() function calls** as appropriate. Strategy Tester simulated orders are not live brokerage execution.

## Initial settings for investigation (not validated as profitable)

| Setting | Initial value | Purpose |
|---|---:|---|
| Chart timeframe | 5m | Entry and retest observations |
| Primary liquidity references | Confirmed 1H and 15m | Minimize micro-sweep clutter |
| Session policy | Adaptive | Allows quiet-session entries subject to extra candle/volume rules |
| Relative volume in priority hours | 1.20 | Active-volume preference |
| Relative volume in quiet hours | 0.65 | Does not automatically block quiet sessions |
| Confluence score | 80 / 100 | Required baseline IGLOF confirmation (not win probability) |
| Technical mode | Gate | Enforce repository-inspired technical confirmations |
| Minimum technical votes | 3 / 6 | MACD, RSI, BB, ROC, ATR regime, GK regime |
| Minimum RR | Target 2R | Initial chart trade plan and strategy exit bracket |
| First test sizing | Fixed | Safer to verify order-size semantics before risk sizing |
| Fixed simulated quantity | 1 | **Unit size depends on the instrument** |

### How the six technical votes work

The direction must have the corresponding condition:

- MACD: value above signal for buy, below for sell.
- RSI: at least 51 for buy; at most 49 for sell (both configurable).
- Bollinger middle: close at/above middle for buy, at/below for sell.
- ROC: positive for buy, negative for sell.
- ATR volatility regime: current ATR / long-baseline ATR within configurable bounds (a neutral yes/no vote, not directional).
- Garman–Klass volatility regime: 20-bar OHLC volatility estimate relative to its longer baseline, within configurable bounds (also not directional).

The last two measures are **volatility state filters**; they don't identify buying or selling by themselves. The dashboard displays each row and total of the six votes. The original liquidity / risk hard gates always apply even if technical votes are strong.

## Footprint-specific settings

Use `IGLOF_GoldPrediction_Footprint_Strategy.pine` only with adequate TradingView entitlement and symbol feed. `request.footprint()` returns TradingView-classified lower-timeframe volume data (categorized buy and sell volume, delta, POC and rows); it does not read another chart's displayed footprint automatically and it does **not** provide consolidated CME full depth of market.

If the script reports missing footprint data, do not treat it as neutral or positive. Use the Standard Edition while investigating entitlement and supported symbols. A single `request.footprint()` call is used to respect Pine's documented limit.

## News events

The script **does not pull the future economic release calendar automatically**. In Settings → `04 - Sessions and manual economic events` enable up to three events, choose importance and their exact dates/times in the timestamp picker. HIGH defaults to a 15-minute pre-event and 5-minute post-event block; MEDIUM/LOW can be blocked or warning only according to settings. Use a reputable official calendar (Federal Reserve, BLS, BEA) to confirm each event.

**If the dashboard says “NO EVENTS ENTERED,” the strategy is NOT news-aware.** In that case, use the manual emergency trade lock around relevant releases. Change events weekly; older dates do not protect future trading.

## Position sizes and backtest interpretation

- Starting capital, commission and simulated slippage in the source are **placeholders**. They should be changed through the strategy Properties before interpreting performance.
- Default `Fixed` method submits a fixed `qty` in the TradingView instrument's strategy units.
- `Equity risk` method computes quantity as `(strategy.equity × risk%) / (price stop distance × currency value per point per unit)`, then rounds down to a configurable increment. For example, a standard GC contract has a 100 USD-per-$1 price-point value, while MGC has 10. For XAUUSD CFDs, check your broker's contract specifications and platform symbol sizing.
- If the proposed quantity is below the minimum increment/size, **no order is issued**. The dashboard shows the quantity check.
- Stop and 2R target are based on the close of a confirmed signal candle and are submitted as a bracket together with the market entry. This is a simplified bar-close fill model; gap/slippage can change realized risk and reward.
- TradingView does not necessarily reproduce a specific CFD broker's bid/ask spreads, swaps, fills, commissions, or order restrictions. Use conservative parameters and forward paper trading.
- Indicators such as EMA, MACD and pivot detection consume historical warm-up; backtests on too-short history can be misleading.
- Do not optimize on the entire sample: reserve out-of-sample periods, compare normal vs news days and London/New York vs quiet hours, review max drawdown, trade count and expectancy.

## How to save to your GitHub

The linked repository is publicly viewable but this package has **not** been committed to it. Do not assume you own or can write to `Joshua-Leow/gold_prediction`. To create your own version:

1. Sign in to GitHub.
2. Open <https://github.com/Joshua-Leow/gold_prediction>.
3. Click **Fork** to create a repository under an account you control (or use your existing fork).
4. In your fork, create a `tradingview/` folder by adding a file at `tradingview/IGLOF_GoldPrediction_Standard_Strategy.pine` and paste the source.
5. Add `tradingview/IGLOF_GoldPrediction_Footprint_Strategy.pine` likewise.
6. Add this README as `tradingview/README.md`.
7. Commit to a new branch `feature/iglof-pine-strategy`, then optionally create a pull request to the upstream repository **only if you want to propose the changes to its maintainer**.

If using Git locally, clone **your fork**, not the original repository unless you have permission:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/gold_prediction.git
cd gold_prediction
git switch -c feature/iglof-pine-strategy
mkdir -p tradingview
# Copy both downloaded .pine files and this README into tradingview/
git add tradingview
git commit -m "Add IGLOF gold liquidity and technical Pine strategies"
git push -u origin feature/iglof-pine-strategy
```

## Core constraints

- No Pine ML inference, no direct Python integration, no cross-account copying of a trained model.
- No live COMEX DOM or market-by-order execution feed.
- No automatic economic news calendar.
- No proof of actual institutional ownership of orders or blocks.
- No guarantee of non-repainting on every feed or market setting until tested in TradingView.
- No verification of profitability or Pine compilation from this environment. If you receive a compilation error, record the exact error and line number for correction.

Research sources:
- Repository: https://github.com/Joshua-Leow/gold_prediction
- TradingView footprint documentation: https://www.tradingview.com/pine-script-docs/concepts/other-timeframes-and-data/
- TradingView strategy documentation: https://www.tradingview.com/pine-script-docs/concepts/strategies/
