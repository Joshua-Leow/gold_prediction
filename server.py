import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import sys

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8081", "http://127.0.0.1:8081"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

def update_config_file(config_data):
    """Update config.py with new values"""
    config_template = f'''
import numpy as np

symbol = '{config_data["symbol"]}'                      # Trading symbol to be downloaded with yfinance
interval = '{config_data["interval"]}'                          # Time interval
confidence = {config_data["confidence"]}                         # Prediction confidence threshold
target_candle = {config_data["targetCandle"]}                       # Future candle to predict
profit_perc = {config_data["profitPerc"]}                       # Take profit percentage
stop_loss_perc = {config_data["stopLossPerc"]}                    # Stop loss percentage
gap_between_trades = {config_data["gapBetweenTrades"]}                   # Number of candles to wait before making the next trade
feature_horizons = {config_data["featureHorizons"]}  # Feature Horizons to be trained with
max_positions = {config_data["maxPositions"]}                       # Max number of open positions at a time
long_bias = {config_data["longBias"]}                          # Long bias (1.0 representing no bias)
leverage = {config_data["leverage"]}                           # leverage multiplier

def define_target_labels(df):
    """
    Defines target labels for a given DataFrame.

    Args:
        df: DataFrame containing financial data with columns: ["Future_High", "Future_Low", "Close"]
        profit_perc: Profit percentage.
        stop_loss_perc: Stop-loss percentage.

    Returns: A Series of target labels (-1, 0, or 1).
    """
    long_condition = (df["Future_High"] > df["Close"] + (df["Close"] * profit_perc / 100)) & \
                     (df["Future_Low"] > df["Close"] - (df["Close"] * stop_loss_perc / 100))

    short_condition = (df["Future_Low"] < df["Close"] - (df["Close"] * profit_perc / 100)) & \
                      (df["Future_High"] < df["Close"] + (df["Close"] * stop_loss_perc / 100))

    return np.where(long_condition, 1,
            np.where(short_condition, -1, 0))
'''
    # Write the new configuration to config.py
    with open('config.py', 'w') as f:
        f.write(config_template.strip())

@app.route('/')
def index():
    return "WebSocket Server Running"


@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    try:
        print("run_backtest function executed")
        global last_position
        last_position = 0

        config_data = request.json
        print("Received config:", config_data)

        # Get the directory of main.py
        script_dir = os.path.dirname(os.path.abspath("main.py"))

        # Create the process with proper environment setup
        process = subprocess.Popen(
            [sys.executable, "main.py"],  # Use sys.executable to ensure correct Python interpreter
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=script_dir,  # Set working directory to script location
            env={**os.environ,  # Include current environment variables
                 'PYTHONPATH': script_dir + os.pathsep + os.environ.get('PYTHONPATH', '')}
        )

        # Clear the log file when starting the server
        with open('prints.log', 'w') as f:
            f.write('')

        # Read output while the process is running
        while True:
            output = process.stdout.readline()
            if output:
                print(output)
                # Optionally write to your log file
                with open('prints.log', 'a') as log_file:
                    log_file.write(output)

            # Check if process has finished
            if process.poll() is not None:
                break

        # Get the return code
        return_code = process.wait()

        if return_code == 0:
            return jsonify({"message": "Backtest completed successfully!",
                            "config": config_data})
        else:
            # Capture any error output
            error_output = process.stderr.read()
            return jsonify({"error": f"Process failed with return code {return_code}. Error: {error_output}",
                            "config": config_data})
    except Exception as e:
        print(f"Failed at run_backtest with error:\n{e}")
        return jsonify({"error": str(e)})


# Global variable to track last read position
last_position = 0


@app.route('/get-logs', methods=['GET'])
def get_logs():
    global last_position
    log_file = 'prints.log'
    new_lines = []

    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            if last_position == 0:
                print("## last position == 1")
                file.seek(0)
            else:
                file.seek(last_position)

            new_lines = file.readlines()
            print(f"New lines: {new_lines}")
            last_position = file.tell()
            print(f"Last position: {last_position}")

    return jsonify(new_lines=new_lines)


if __name__ == "__main__":
    app.run(port=5000, debug=True)