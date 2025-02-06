# Gold Trading Prediction System

## Overview
This project implements a machine learning-based trading system for gold futures (GC=F), utilizing multiple models and technical indicators to predict price movements. The system includes backtesting capabilities and trading simulation with configurable take-profit and stop-loss levels.

## Features
- Multiple machine learning models:
  - Random Forest
  - XGBoost
  - LightGBM
  - Gradient Boosting
- Technical indicators:
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Garman Klass Volume indicators
  - Average True Range (ATR)
  - Rate of Change (ROC)
- Automated backtesting system
- Trading simulation with configurable parameters
- Performance visualization using FinPlot
- Comprehensive trading statistics

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd gold_prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```
gold_prediction/
├── notebooks/
│   ├── gold_prediction.ipynb
├── src/
│   ├── Trade.py                # Trading logic implementation
│   ├── Stats.py                # Statistics calculation
│   ├── ModelMetrics.py         # Model Metrics dataclass
│   ├── processing.py           # Chart visualization
│   ├── feature_engineering.py  # Chart visualization
│   ├── compare_models.py       # Chart visualization
│   ├── plot_chart.py           # Chart visualization
├── data/
│   └── archive/                # Historical data storage
├── config.py                   # Configuration parameters
├── main.py                     # Main execution file
└── requirements.txt            # Project dependencies
```

## Configuration
Key parameters can be configured in `config.py`:
```python
symbol = 'GC=F'           # Trading symbol
interval = '1h'           # Time interval
confidence = 0.7          # Prediction confidence threshold
target_candle = 7         # Future candle to predict
profit_perc = 1.00        # Take profit percentage
stop_loss_perc = 0.10     # Stop loss percentage

def define_target_labels(df):
    """Function that returns nparray[-1,0,1]
    -1: represents prediction for short trades
     0: represents prediction of neutral price movement
     1: represents prediction for long trades"""
```

## Usage

### Running the System
1. Ensure your configuration is set in `config.py`
2. Run the main script:
```bash
python main.py
```

### Output
The system will:
1. Fetch and preprocess historical data
2. Generate technical indicators
3. Train multiple models
4. Perform backtesting
5. Display performance metrics including:
   - Number of trades
   - Win rate
   - Return percentage
   - Buy and hold comparison
   - Total profit
6. Show a visual chart with trade entries/exits

## Trading Strategy
The system implements a combined strategy using:
1. Technical indicators for feature generation
2. Machine learning models for price movement prediction
3. Confidence thresholds for trade entry
4. Fixed take-profit and stop-loss levels

### Entry Conditions
- Model prediction confidence exceeds threshold
- No active trade currently open

### Exit Conditions
- Take profit level reached
- Stop loss level reached

## Performance Metrics
The system tracks various performance metrics:
- Machine Learning Metrics:
  - Precision
  - Recall
  - F1 Score
- Trading Metrics:
  - Win Rate
  - Return Percentage
  - Total Profit
  - Number of Trades
  - Buy and Hold Comparison

## Visualization
The system uses FinPlot to create interactive charts showing:
- Price candlesticks
- Volume
- MACD indicator
- Trade entry/exit points
- Take profit and stop loss levels

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
Joshua-Leow ✌️

## Disclaimer
This software is for educational purposes only. Trading financial instruments carries risk. Past performance does not guarantee future results. Users should understand the risks involved and consult with financial advisors before making any investment decisions.