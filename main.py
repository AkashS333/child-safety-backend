# Install necessary packages
import cv2
import numpy as np
import torch
import time
import os
import pandas as pd
from datetime import datetime
from PIL import ImageGrab, Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from twilio.rest import Client

# Twilio setup (Only sends alerts for harmful content)
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
harmful_labels = ["violence", "adult content", "weapons", "abuse", "explicit language", "drugs", "18+ content", "risky language", "gore"]
safe_labels = ["education", "programming", "learning", "cartoons", "kids", "family friendly", "games", "tutorial", "documentation"]

# Log file paths
BAD_CONTENT_LOG = "alert_log.xlsx"
GOOD_CONTENT_LOG = "good_content.xlsx"

# Folder to save bad content
BAD_CONTENT_FOLDER = r"C:\Users\Akash.s\Downloads\note\note 4\unsisy\bad"

# Capture interval (in seconds)
CAPTURE_INTERVAL = 10

# Global control variable
running = True
bad_content_counter = 1


# Function to capture the entire screen
def capture_screen():
    print("[INFO] Capturing screen...")
    screenshot = ImageGrab.grab()
    filename = "screenshot.png"
    screenshot.save(filename)
    return filename


# Function to slice an image into smaller sections
def slice_image(image_path, grid_size=(3, 3)):
    img = Image.open(image_path)
    img_width, img_height = img.size
    slice_width = img_width // grid_size[0]
    slice_height = img_height // grid_size[1]

    slices = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            left = i * slice_width
            upper = j * slice_height
            right = left + slice_width
            lower = upper + slice_height
            sliced_img = img.crop((left, upper, right, lower))
            slices.append(sliced_img)

    return slices


# Updated function to detect harmful content with image slicing
def detect_harmful_content(image_path):
    try:
        slices = slice_image(image_path, grid_size=(3, 3))  # 3x3 grid
        detected_harmful = False
        detected_label = "Unknown"

        for img in slices:
            inputs = clip_processor(text=harmful_labels + safe_labels, images=img, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
            label_probs = dict(zip(harmful_labels + safe_labels, probs))
            best_label, confidence = max(label_probs.items(), key=lambda x: x[1])

            if best_label in harmful_labels and confidence > 0.4:
                print(f"[ALERT] Harmful content detected: {best_label} (Confidence: {confidence:.2f})")
                detected_harmful = True
                detected_label = best_label
                break  # Stop checking if we find harmful content

        if not detected_harmful:
            detected_label = max(label_probs, key=label_probs.get)  # Pick the most probable safe label

        return detected_label, detected_harmful
    except Exception as e:
        print(f"[ERROR] Content detection failed: {str(e)}")
        return "Unknown", False


# Function to detect age from webcam
def predict_age_from_webcam():
    print("[INFO] Capturing webcam image...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return "Unknown"

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Failed to capture image from webcam")
        return "Unknown"

    image_path = "webcam_capture.png"
    cv2.imwrite(image_path, frame)

    try:
        img = Image.open(image_path)
        results = age_pipe(img)

        detected_ages = [int(result['label'].split('-')[0]) for result in results]
        avg_age = round(sum(detected_ages) / len(detected_ages))

        if avg_age < 10:
            print(f"[WARNING] Detected unrealistically low age ({avg_age}), retrying...")
            return predict_age_from_webcam()  # Retry once

        print(f"[INFO] Detected Age: {avg_age}")
        return avg_age
    except Exception as e:
        print(f"[ERROR] Age detection failed: {str(e)}")
        return "Unknown"


# Function to send an alert via Twilio (Only for harmful content)
def send_alert(message):
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=OWNER_PHONE_NUMBER
        )
        print(f"[ALERT SENT] {message}")
    except Exception as e:
        print(f"[ERROR] Failed to send alert: {str(e)}")


# Function to log bad content and save screenshot
def log_bad_content(age, content, image_path):
    global bad_content_counter
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    bad_screenshot_path = os.path.join(BAD_CONTENT_FOLDER, f"bad_content_{bad_content_counter}.png")
    bad_content_counter += 1
    os.rename(image_path, bad_screenshot_path)

    log_entry = pd.DataFrame([{"Age": age, "Content": content, "Timestamp": now, "File": bad_screenshot_path}])

    if os.path.exists(BAD_CONTENT_LOG):
        existing_data = pd.read_excel(BAD_CONTENT_LOG)
        log_entry = pd.concat([existing_data, log_entry], ignore_index=True)

    log_entry.to_excel(BAD_CONTENT_LOG, index=False)
    print(f"[LOGGED] Bad Content: Age: {age}, Content: {content} at {now}")


# Function to log good content
def log_good_content(age, content):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = pd.DataFrame([{"Age": age, "Content": content, "Timestamp": now}])

    if os.path.exists(GOOD_CONTENT_LOG):
        existing_data = pd.read_excel(GOOD_CONTENT_LOG)
        log_entry = pd.concat([existing_data, log_entry], ignore_index=True)

    log_entry.to_excel(GOOD_CONTENT_LOG, index=False)
    print(f"[LOGGED] Good Content: Age: {age}, Content: {content} at {now}")


# Function to stop detection
def stop_detection():
    global running
    running = False
    print("[INFO] Detection stopped.")


# Main screen detection loop
def start_detection():
    global running
    print(f"[INFO] Starting screen detection every {CAPTURE_INTERVAL} seconds...")
    running = True
    try:
        while running:
            screenshot_file = capture_screen()
            age = predict_age_from_webcam()
            content, is_harmful = detect_harmful_content(screenshot_file)

            if is_harmful:
                log_bad_content(age, content, screenshot_file)
                send_alert(f"🚨 Alert! Harmful content detected: {content}")
            else:
                os.remove(screenshot_file)
                log_good_content(age, content)

            time.sleep(CAPTURE_INTERVAL)
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")


# Menu for controlling the app
def main():
    while True:
        print("\n⚡ AI Screen & Age Monitor")
        print("1. Start Screen & Age Detection")
        print("2. Stop Detection")
        print("3. Set Capture Interval (seconds)")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")
        if choice == "1":
            start_detection()
        elif choice == "2":
            stop_detection()
        elif choice == "3":
            global CAPTURE_INTERVAL
            CAPTURE_INTERVAL = int(input("Enter new capture interval: "))
        elif choice == "4":
            stop_detection()
            break


if __name__ == "__main__":
    main()
