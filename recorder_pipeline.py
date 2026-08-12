import os
import sys
import time
import datetime
import threading
from collections import deque

# 1. PATH RESOLUTION
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)

try:
    from config import (
        STREAMS, ALERTS_DIR, MODEL_PATH, PRE_ROLL_SEC, 
        POST_ROLL_SEC, TARGET_FPS, DETECTION_CLASSES, 
        get_stream_log_function as get_stream_logger,
        system_logger
    )
except ImportError:
    try:
        from config import get_stream_logger, system_logger
    except:
        print("❌ CRITICAL ERROR: Could not load config.py")
        sys.exit(1)

# 2. Class Definition for the Camera Agent
class CameraAgent(threading.Thread):
    def __init__(self, stream_info):
        super().__init__()
        self.name = stream_info["name"]
        self.url = stream_info.get("url", "")
        self.enabled = stream_info.get("enabled", False)
        try: 
            from config import get_stream_logger as gs; self.logger = gs(self.name)
        except:
            import logging
            self.logger = logging.getLogger(self.name)
            
        self.daemon = True 
        self._stop_event = threading.Event()

    def run(self):
        """The primary execution loop for each camera thread."""
        print(f"🚀 [Thread Start] Initializing agent: {self.name}")
        
        try:
            import cv2
            from ultralytics import YOLO
        except ImportError as e:
            self.logger.error(f"❌ Dependency Error in {self.name}: {e}")
            return

        # 1. Initialize AI Engine (YOLO)
        detector = None
        if os.path.exists(MODEL_PATH) and MODEL_PATH:
            try:
                detector = YOLO(MODEL_PATH)
                self.logger.info("✅ AI Model loaded successfully.")
            except Exception as e:
                self.logger.error(f"❌ Failed to load model: {e}")
        else:
            self.logger.warning("⚠️ No valid YOLO path found; running in VISION-ONLY mode.")

        # 2. Main Loop with Robust Connection Retry Logic
        cap = None
        state = "IDLE"
        buffer_size = int((PRE_ROLL_SEC + POST_ROLL_SEC) * TARGET_FPS)
        frame_buffer = deque(maxlen=buffer_size)
        writer = None
        post_roll_end_time = 0

        try:
            while not self._stop_event.is_set():
                loop_start = time.time()

                # --- CONNECTION MANAGEMENT (RETRY LOGIC) ---
                if cap is None or not cap.isOpened():
                    self.logger.info(f"🔗 Attempting connection to RTMP: {self.url}")
                    try:
                        temp_cap = cv2.VideoCapture(self.url)
                        if temp_cap and temp_cap.isOpened():
                            cap = temp_cap
                            self.logger.info(f"✅ [CONNECTED] '{self.name}' established stream at {self.url}")
                        else:
                            raise ConnectionError("Could not open stream via OpenCV.")
                    except Exception as e:
                        self.logger.error(f"❌ [CONNECTION FAILED] '{self.name}': {e}. Retrying in 5s...")
                        if cap: cap.release()
                        cap = None
                        time.sleep(5)
                        continue # This keeps the thread alive!

                # --- STREAM PROCESSING ---
                ret, frame = cap.read()

                if not ret:
                    self.logger.error(f"⚠️ [STREAM LOST] '{self.name}' lost connection to {self.url}. Retrying in 5s...")
                    cap.release()
                    cap = None
                    time.sleep(5)
                    continue

                # A. Update rolling pre-roll buffer
                frame_copy = frame.copy()
                frame_buffer.append(frame_copy)

                # B. Perform Detection (if model is available)
                person_detected = False
                if detector:
                    results = detector(frame, verbose=False, classes=DETECTION_CLASSES)
                    person_detected = len(results[0].boxes) > 0

                # C. STATE MACHINE LOGIC
                if state == "ID_STATE": # Using a safe check via string
                    pass # We'll use the logic below directly to avoid any confusion
                
                # Unified State Logic for clarity and stability
                if state == "IDLE":
                    if person_detected:
                        state = "RECORDING"
                        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_file_path = os.path.join(ALERTS_DIR, f"{self.name}_EVENT_{ts_str}.mp4")
                        self.logger.warning("🚨 [DETECTED] Person found! Starting continuous recording.")
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(video_file_path, fourcc, TARGET_FPS, (w, h))
                        for hist_frame in frame_buffer: 
                            writer.write(hist_frame)

                elif state == "RECORDING":
                    if person_detected:
                        if writer: writer.write(frame)
                    else:
                        state = "POST_ROLL"
                        post_roll_end_time = time.time() + POST_ROLL_SEC
                        self.logger.info(f"🏠 [GONE] Person departed. Monitoring area for {POST_ROLL_SEC}s extra.")

                elif state == "POST_ROLL":
                    if writer: 
                        writer.write(frame)
                    if time.time() > post_roll_end_time:
                        self.logger.info(f"✅ Session ended for {self.name}. Closing file.")
                        if writer:
                            writer.release()
                            writer = None
                        state = "IDLE"

                # D. Frame Rate Control (Preventing CPU overload)
                elapsed = time.time() - loop_start
                sleep_duration = max(0, (1.0 / TARGET_FPS) - elapsed)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

        except Exception as e:
            self.logger.error(f"💥 Thread Crash [{self.name}]: {e}")
        finally:
            if writer: writer.release()
            if cap: cap.release()
            self.logger.info("🛑 Agent thread shutting down.")

    def stop_agent(self):
        self._stop_event.set()

# ---------------------------------------------------------
# MAIN ORCHESTRATOR (The Launcher)
# ---------------------------------------------------------
if __name__ == "__main__":
    system_logger = get_stream_logger("SYSTEM")
    system_logger.info("🚀 [MASTER] Initializing Multi-Agent Pipeline...")

    all_agents = []
    for stream in STREAMS:
        if stream["enabled"]:
            agent_thread = CameraAgent(stream)
            agent_thread.start() 
            all_agents.append(agent_thread)
            system_logger.info(f"📡 [MASTER] Launched Thread Worker for: {stream['name']}")
        else:
            system_logger.info(f"⏭️ [MASTER] Skipping Disabled Stream: {stream['name']}")

    if not all_agents:
        system_logger.error("❌ No active streams found! Exiting.")
        sys.exit(1)

    system_logger.info("[MASTER] All threads running parallelly. Monitoring ACTIVE.")

    try:
        while True:
            alive_count = sum(1 for t in all_agents if t.is_alive())
            if alive_count < len(all_agents):
                system_logger.warning(f"⚠️ ALERT: {len(all_agents) - alive_count} camera thread(s) have died/stopped!")
            time.sleep(15)
    except KeyboardInterrupt:
        system_logger.info("[MASTER] Shutdown signal received.")

    for a in all_agents:
        a.stop_agent()
    system_logger.info("[MASTER] Pipeline shutdown complete.")
