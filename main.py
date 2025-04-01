from fastapi import FastAPI, BackgroundTasks
import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import ImageGrab, Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from twilio.rest import Client

# Initialize FastAPI
app = FastAPI(title="AI Child Safety System")

# Twilio setup
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OWNER_PHONE_NUMBER = os.getenv("OWNER_PHONE_NUMBER")
TWILIO_PHONE_NUMBER = "+18149928399"

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load AI models
print("[INFO] Loading AI models...")
age_pipe = pipeline("image-classification", model="dima806/fairface_age_image_detection")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Labels for CLIP detection
harmful_labels = ["violence", "adult content", "weapons", "abuse", "explicit language", "drugs", "18+ content"]
safe_labels = ["education", "kids", "family friendly", "games", "tutorial"]

# Log file paths
BAD_CONTENT_LOG = "alert_log.xlsx"
GOOD_CONTENT_LOG = "good_content.xlsx"

# Global control variable
running = True


# Function to capture screen
def capture_screen():
    screenshot = ImageGrab.grab()
    filename = "screenshot.png"
    screenshot.save(filename)
    return filename


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

    image_path = "webcam_capture.png"
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


# Log bad content
def log_bad_content(age, content, image_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = pd.DataFrame([{"Age": age, "Content": content, "Timestamp": now}])
    log_entry.to_excel(BAD_CONTENT_LOG, index=False, mode='a', header=False)
    send_alert(f"🚨 Alert! Harmful content detected: {content}")


# Log good content
def log_good_content(age, content):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = pd.DataFrame([{"Age": age, "Content": content, "Timestamp": now}])
    log_entry.to_excel(GOOD_CONTENT_LOG, index=False, mode='a', header=False)


# Continuous monitoring
def monitor():
    global running
    while running:
        screenshot_file = capture_screen()
        age = predict_age_from_webcam()
        content, is_harmful = detect_harmful_content(screenshot_file)

        if is_harmful:
            log_bad_content(age, content, screenshot_file)
        else:
            os.remove(screenshot_file)
            log_good_content(age, content)

        time.sleep(10)


# API Endpoints
@app.get("/")
def home():
    return {"message": "AI Child Safety System Running!"}


@app.post("/start")
def start_monitoring(background_tasks: BackgroundTasks):
    global running
    if running:
        return {"message": "Monitoring already running!"}
    running = True
    background_tasks.add_task(monitor)
    return {"message": "Started monitoring!"}


@app.post("/stop")
def stop_monitoring():
    global running
    running = False
    return {"message": "Monitoring stopped!"}


@app.get("/status")
def get_status():
    return {"monitoring": running}


# Run FastAPI with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
