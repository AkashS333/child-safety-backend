from fastapi import FastAPI, HTTPException, BackgroundTasks
import os
import cv2
import time
import threading
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from twilio.rest import Client
from supabase import create_client, Client
import uuid

app = FastAPI()

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

harmful_labels = ["violence", "adult content", "weapons", "abuse", "explicit language", "drugs", "18+ content"]
safe_labels = ["education", "kids", "family friendly", "games", "tutorial"]

running = False

# Function to capture screen using OpenCV
def capture_screen():
    cap = cv2.VideoCapture(0)  # Use webcam instead
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    filename = f"screenshot_{uuid.uuid4()}.png"
    cv2.imwrite(filename, frame)
    return filename

# Upload image to Supabase Storage
def upload_to_supabase(file_path):
    with open(file_path, "rb") as file:
        file_name = os.path.basename(file_path)
        res = supabase.storage.from_("screenshots").upload(file_name, file, {"content-type": "image/png"})
    os.remove(file_path)
    return f"{SUPABASE_URL}/storage/v1/object/public/screenshots/{file_name}"

# Detect harmful content
def detect_harmful_content(image_path):
    try:
        img = Image.open(image_path)
        inputs = clip_processor(text=harmful_labels + safe_labels, images=img, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
        label_probs = dict(zip(harmful_labels + safe_labels, probs))
        best_label, confidence = max(label_probs.items(), key=lambda x: x[1])
        
        return best_label, confidence > 0.4
    except Exception as e:
        return "Error", False

# Send Twilio Alert
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
def log_bad_content(user_id, content, image_path):
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

# Continuous monitoring
def monitor(user_id):
    global running
    while running:
        screenshot_file = capture_screen()
        if screenshot_file:
            content, is_harmful = detect_harmful_content(screenshot_file)
            if is_harmful:
                log_bad_content(user_id, content, screenshot_file)
            else:
                os.remove(screenshot_file)
        time.sleep(10)

@app.get("/")
def home():
    return {"message": "AI Child Safety System Running!"}

@app.post("/start")
def start_monitoring(background_tasks: BackgroundTasks, user_id: str):
    global running
    if running:
        return {"message": "Monitoring already running!"}
    
    running = True
    background_tasks.add_task(monitor, user_id)
    return {"message": "Started monitoring!"}

@app.post("/stop")
def stop_monitoring():
    global running
    running = False
    return {"message": "Monitoring stopped!"}

@app.get("/status")
def get_status():
    return {"monitoring": running}
