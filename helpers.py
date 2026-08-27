import os
import json
import subprocess
import time
import threading
import cv2
from config import (STREAMS, OVERRIDE_F, SETTINGS_F, PRE_ROLL_SEC,
                    POST_ROLL_SEC, TARGET_FPS, CONFIDENCE_THRESHOLD, DETECTION_CLASSES)

LATEST_FRAMES = {}

def _stream_worker(name, url):
    """Dauerhafter Worker für genau einen Stream, hält die Verbindung offen."""
    while True:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        last_encode_time = 0

        while cap.isOpened():
            # Frames durchgehend auslesen, damit der Puffer nicht vollläuft
            ret, frame = cap.read()
            if not ret:
                break  # Bricht ab, wenn der Stream abreißt -> Reconnect
            
            current_time = time.time()
            # Nur alle ~0.2 Sekunden (5 fps) encoden, um CPU zu sparen
            if current_time - last_encode_time >= 0.2:
                frame_resized = cv2.resize(frame, (640, 360))
                ret_enc, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret_enc:
                    LATEST_FRAMES[name] = buffer.tobytes()
                last_encode_time = current_time

        if cap is not None:
            cap.release()
        time.sleep(2)  # Kurze Pause vor dem Reconnect (falls Kamera offline)

def start_thumbnail_thread():
    """Startet für JEDEN Stream einen eigenen Hintergrund-Thread."""
    for s in STREAMS:
        url = s.get("url")
        name = s["name"]
        if url:
            threading.Thread(target=_stream_worker, args=(name, url), daemon=True).start()

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
