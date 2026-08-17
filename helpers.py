import os
import json
import subprocess
import time
import threading
import cv2
from config import (STREAMS, OVERRIDE_F, SETTINGS_F, PRE_ROLL_SEC, 
                    POST_ROLL_SEC, TARGET_FPS, CONFIDENCE_THRESHOLD, DETECTION_CLASSES)

LATEST_FRAMES = {}

def update_thumbnails():
    """Hintergrund-Thread: Zieht kontinuierlich 1 Frame aus den Streams in den RAM"""
    while True:
        for s in STREAMS:
            url = s.get("url")
            name = s["name"]
            if not url:
                continue
            try:
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    frame = cv2.resize(frame, (640, 360))
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        LATEST_FRAMES[name] = buffer.tobytes()
            except Exception:
                pass
        time.sleep(0.25)

def start_thumbnail_thread():
    threading.Thread(target=update_thumbnails, daemon=True).start()

def is_pipeline_running():
    result = subprocess.run(['pgrep', '-f', 'recorder_pipeline.py'], capture_output=True)
    return result.returncode == 0

def load_overrides():
    overrides = {s["name"]: ("ON" if s["enabled"] else "OFF") for s in STREAMS}
    if os.path.exists(OVERRIDE_F):
        try:
            with open(OVERRIDE_F, 'r') as f:
                overrides.update(json.load(f))
        except (json.JSONDecodeError, IOError):
            pass
    return overrides

def load_settings():
    settings = {
        "PRE_ROLL_SEC": PRE_ROLL_SEC, "POST_ROLL_SEC": POST_ROLL_SEC,
        "TARGET_FPS": TARGET_FPS, "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
        "DETECTION_CLASSES": DETECTION_CLASSES
    }
    if os.path.exists(SETTINGS_F):
        try:
            with open(SETTINGS_F, 'r') as f:
                settings.update(json.load(f))
        except (json.JSONDecodeError, IOError):
            pass
    return settings

def save_overrides(data):
    with open(OVERRIDE_F, 'w') as f:
        json.dump(data, f, indent=4)

def format_size(size_bytes):
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"
