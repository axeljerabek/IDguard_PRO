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
        last_setting_check = 0
        encode_interval = 1.0  # Fallback (1 FPS)

        while cap.isOpened():
            # grab() holt das Frame-Paket vom Netzwerk, DEKODIERT es aber noch nicht (spart CPU)
            if not cap.grab():
                break

            current_time = time.time()

            # Settings alle 2 Sekunden neu auslesen
            if current_time - last_setting_check > 2.0:
                settings = load_settings()
                try:
                    thumbnail_fps = float(settings.get('THUMBNAIL_FPS', 1.0))
                    if thumbnail_fps <= 0:
                        thumbnail_fps = 1.0
                except (ValueError, TypeError):
                    thumbnail_fps = 1.0

                encode_interval = 1.0 / thumbnail_fps
                last_setting_check = current_time

            # Prüfen, ob das Intervall für den nächsten Thumbnail-Frame erreicht ist
            if current_time - last_encode_time >= encode_interval:
                # Erst hier wird exakt dieses EINE Bild auf der CPU dekodiert
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    frame_resized = cv2.resize(frame, (640, 360))
                    ret_enc, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret_enc:
                        LATEST_FRAMES[name] = buffer.tobytes()
                last_encode_time = current_time
            else:
                # Kurze Drosselung, um die while-Schleife nicht mit maximaler Frequenz laufen zu lassen
                time.sleep(0.01)

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
