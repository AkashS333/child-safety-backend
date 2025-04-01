from flask import Flask, jsonify, request
import os
import cv2
import time
import threading
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import ImageGrab, Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from twilio.rest import Client
from supabase import create_client, Client
import uuid

# Initialize Flask App
app = Flask(__name__)

# Twilio setup
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OWNER_PHONE_NUMBER = os.getenv("OWNER_PHONE_NUMBER")
TWILIO_PHONE_NUMBER = "+18149928399"

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load AI models
print("[INFO] Loading AI models...")
age_pipe = pipeline("image-classification", model="dima806/fairface_age_image_detection")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Labels for CLIP detection
harmful_labels = ["violence", "adult content", "weapons", "abuse", "explicit language", "drugs", "18+ content"]
safe_labels = ["education", "kids", "family friendly", "games", "tutorial"]

# Global control variable
running = False

# Function to capture screen
def capture_screen():
    screenshot = ImageGrab.grab()
    filename = f"screenshot_{uuid.uuid4()}.png"
    screenshot.save(filename)
    return filename

# Upload image to Supabase Storage
def upload_to_supabase(file_path):
    with open(file_path, "rb") as file:
        file_name = os.path.basename(file_path)
        res = supabase.storage.from_("screenshots").upload(file_name, file, {"content-type": "image/png"})
    
    os.remove(file_path)
    return f"{SUPABASE_URL}/storage/v1/object/public/screenshots/{file_name}"

# Function to slice image
def slice_image(image_path, grid_size=(3, 3)):
    img = Image.open(image_path)
    img_width, img_height = img.size
    slice_width = img_width // grid_size[0]
    slice_height = img_height // grid_size[1]

    slices = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            left, upper = i * slice_width, j * slice_height
            right, lower = left + slice_width, upper + slice_height
            slices.append(img.crop((left, upper, right, lower)))
    
    return slices

# Detect harmful content
def detect_harmful_content(image_path):
    try:
        slices = slice_image(image_path, grid_size=(3, 3))
        for img in slices:
            inputs = clip_processor(text=harmful_labels + safe_labels, images=img, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
            label_probs = dict(zip(harmful_labels + safe_labels, probs))
            best_label, confidence = max(label_probs.items(), key=lambda x: x[1])

            if best_label in harmful_labels and confidence > 0.4:
                return best_label, True
        return best_label, False
    except Exception as e:
        return "Error", False

# Detect age from webcam
def predict_age_from_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return "Unknown"

    image_path = f"webcam_{uuid.uuid4()}.png"
    cv2.imwrite(image_path, frame)
    
    try:
        img = Image.open(image_path)
        results = age_pipe(img)
        detected_ages = [int(result['label'].split('-')[0]) for result in results]
        return round(sum(detected_ages) / len(detected_ages))
    except Exception:
        return "Unknown"

# Send alert via Twilio
def send_alert(message):
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=OWNER_PHONE_NUMBER
        )
    except Exception as e:
        print(f"[ERROR] Failed to send alert: {e}")

# Log bad content in Supabase
def log_bad_content(user_id, age, content, image_path):
    now = datetime.utcnow().isoformat()
    image_url = upload_to_supabase(image_path)
    
    data = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "screenshot_id": str(uuid.uuid4()),
        "alert_type": content,
        "severity": "high",
        "message": f"Harmful content detected: {content}",
        "is_read": False,
        "created_at": now,
    }

    supabase.table("alerts").insert(data).execute()
    send_alert(f"🚨 Alert! Harmful content detected: {content}")

# Log good content in Supabase
def log_good_content(user_id, age, content):
    now = datetime.utcnow().isoformat()
    data = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "date": now[:10],
        "screen_time": 0,
        "app_usage": {},
        "content_safety": {"label": content, "safe": True},
        "created_at": now
    }
    
    supabase.table("analytics_data").insert(data).execute()

# Continuous monitoring (Runs in a separate thread)
def monitor(user_id):
    global running
    while running:
        screenshot_file = capture_screen()
        age = predict_age_from_webcam()
        content, is_harmful = detect_harmful_content(screenshot_file)

        if is_harmful:
            log_bad_content(user_id, age, content, screenshot_file)
        else:
            os.remove(screenshot_file)
            log_good_content(user_id, age, content)

        time.sleep(10)

# API Endpoints
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "AI Child Safety System Running!"})

@app.route("/start", methods=["POST"])
def start_monitoring():
    global running
    if running:
        return jsonify({"message": "Monitoring already running!"})
    
    user_id = request.json.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    running = True
    thread = threading.Thread(target=monitor, args=(user_id,))
    thread.start()
    
    return jsonify({"message": "Started monitoring!"})

@app.route("/stop", methods=["POST"])
def stop_monitoring():
    global running
    running = False
    return jsonify({"message": "Monitoring stopped!"})

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"monitoring": running})

# Run Flask App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
