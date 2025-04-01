from flask import Flask, jsonify
import os
import threading

# Import AI detection functions
from ai_detection import start_detection, stop_detection

app = Flask(__name__)

# Background thread for running detection
detection_thread = None

@app.route('/')
def home():
    return "AI Screen & Age Detection Service is Running!"

@app.route('/start', methods=['GET'])
def start():
    global detection_thread
    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = threading.Thread(target=start_detection, daemon=True)
        detection_thread.start()
        return jsonify({"message": "Detection started!"})
    return jsonify({"message": "Detection is already running!"})

@app.route('/stop', methods=['GET'])
def stop():
    stop_detection()
    return jsonify({"message": "Detection stopped!"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
