import os
from flask import Flask, jsonify
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


@app.route('/')
def index():
    return "WebSocket Server Running"


@app.route('/run_backtest', methods=['GET'])
def run_backtest():
    try:
        print("run_backtest function executed")
        global last_position
        last_position = 0

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
            return jsonify({"message": "Backtest completed successfully!"})
        else:
            # Capture any error output
            error_output = process.stderr.read()
            return jsonify({"error": f"Process failed with return code {return_code}. Error: {error_output}"})

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
    # Clear the log file when starting the server
    with open('prints.log', 'w') as f:
        f.write('')
    app.run(port=5000, debug=True)