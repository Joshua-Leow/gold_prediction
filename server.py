from flask import Flask, request, jsonify
import subprocess
from flask_socketio import SocketIO
import config

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", engineio_logger=False)

@app.route('/')
def index():
    return "WebSocket Server Running"

@socketio.on('run_backtest')
def run_backtest():
    try:
        print("run_backtest function executed")
        # Run the external script and capture stdout line by line
        process = subprocess.Popen(
            ["python", "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Ensure output is captured as text
        )

        # Stream each line of output to the frontend
        for line in iter(process.stdout.readline, ''):
            print(line)  # Log to server terminal for debugging
            socketio.emit('log', line)  # Send each line to the frontend

        process.stdout.close()
        process.wait()

    except Exception as e:
        print("###################################################################################")
        socketio.emit('log', f"Error: {str(e)}")


import logging
# In-memory log storage
from io import StringIO
log_stream = StringIO()

# Set up logging to the stream
logging.basicConfig(level=logging.INFO, stream=log_stream, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Example route to trigger logs
@app.route('/run-trade-simulation')
def run_trade_simulation():
    logger.info("Trade simulation started")
    # Simulate some logging
    logger.info("Created SHORT trade at 2024-05-20 07:00:00+08:00, Entry price: 2441.30")
    logger.info("Trade Closed at 2024-05-20 09:00:00+08:00, Closing price: 2446.42 (Loss)")
    return jsonify({"status": "Simulation complete"})

# Route to fetch logs
@app.route('/logs')
def get_logs():
    log_stream.seek(0)  # Move cursor to the beginning of the log stream
    return jsonify({"logs": log_stream.read().splitlines()})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
