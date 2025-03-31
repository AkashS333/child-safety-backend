from flask import Flask
import os  # Import os to read environment variables

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Render Deployment!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # Get PORT from environment
    app.run(host="0.0.0.0", port=port)  # Bind to 0.0.0.0 to work on Render
