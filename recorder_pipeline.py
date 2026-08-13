import os
import sys
import time
import datetime
import multiprocessing
from collections import deque

# CPU-Thread-Wildwuchs von PyTorch/OpenBLAS global drosseln
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

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
class CameraAgent(multiprocessing.Process):
    def __init__(self, stream_info):
        super().__init__()
        self.name = stream_info["name"]
        self.url = stream_info.get("url", "")
        self.enabled = stream_info.get("enabled", False)

        self.daemon = True
        self._stop_event = multiprocessing.Event()

    def run(self):
        """The primary execution loop for each camera process using PyAV and CUDA GPU."""
        try:
            from config import get_stream_logger as gs
            self.logger = gs(self.name)
        except:
            import logging
            self.logger = logging.getLogger(self.name)

        print(f"🚀 [Process Start] Initializing agent: {self.name}")

        try:
            import av
            import torch
            
            # Umgeht den cuDNN Sublibrary Mismatch. CUDA läuft voll weiter!
            torch.backends.cudnn.enabled = False
            torch.set_num_threads(2)
            
            from ultralytics import YOLO
        except ImportError as e:
            self.logger.error(f"❌ Dependency Error in {self.name}: {e}")
            return

        # 1. Initialize AI Engine (YOLO) auf CUDA GPU
        detector = None
        device_target = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if os.path.exists(MODEL_PATH) and MODEL_PATH:
            try:
                detector = YOLO(MODEL_PATH)
                if device_target == "cuda:0":
                    detector.to("cuda:0")
                    self.logger.info(f"✅ AI Model loaded successfully on CUDA GPU ({torch.cuda.get_device_name(0)}).")
                else:
                    self.logger.warning("⚠️ CUDA not available, falling back to CPU.")
            except Exception as e:
                self.logger.error(f"❌ Failed to load model on CUDA: {e}")
        else:
            self.logger.warning("⚠️ No valid YOLO path found; running in VISION-ONLY mode.")

        # Pre-Roll Puffer
        video_buffer_size = int(PRE_ROLL_SEC * TARGET_FPS)
        audio_buffer_size = int(PRE_ROLL_SEC * 50)  # ca. 50 Audio-Pakete/Sekunde

        video_buffer = deque(maxlen=video_buffer_size)
        audio_buffer = deque(maxlen=audio_buffer_size)

        state = "IDLE"
        post_roll_end_time = 0
        container = None
        out_container = None
        out_video = None
        out_audio = None

        def close_writer():
            nonlocal out_container, out_video, out_audio
            if out_container:
                try:
                    if out_video:
                        for packet in out_video.encode(None):
                            out_container.mux(packet)
                    if out_audio:
                        for packet in out_audio.encode(None):
                            out_container.mux(packet)
                    out_container.close()
                except Exception as e:
                    self.logger.error(f"❌ Error closing output file: {e}")
                finally:
                    out_container = None
                    out_video = None
                    out_audio = None

        def encode_video_frame(img_bgr):
            if not out_container or not out_video:
                return
            try:
                av_frame = av.VideoFrame.from_ndarray(img_bgr, format="bgr24")
                av_frame.pts = None
                for packet in out_video.encode(av_frame):
                    out_container.mux(packet)
            except Exception as e:
                self.logger.error(f"❌ Video encoding error: {e}")

        def encode_audio_frame(a_frame):
            if not out_container or not out_audio:
                return
            try:
                a_frame.pts = None
                for packet in out_audio.encode(a_frame):
                    out_container.mux(packet)
            except Exception as e:
                self.logger.error(f"❌ Audio encoding error: {e}")

        try:
            while not self._stop_event.is_set():
                # --- VERBINDUNGSAUFBAU VIA PYAV ---
                if container is None:
                    self.logger.info(f"🔗 Attempting connection to RTMP: {self.url}")
                    try:
                        container = av.open(
                            self.url,
                            options={"rtmp_live": "live", "rw_timeout": "5000000"}
                        )
                        self.logger.info(f"✅ [CONNECTED] '{self.name}' established stream at {self.url}")
                    except Exception as e:
                        self.logger.error(f"❌ [CONNECTION FAILED] '{self.name}': {e}. Retrying in 5s...")
                        container = None
                        time.sleep(5)
                        continue

                # --- DEMUXING & FRAME PROCESSING ---
                try:
                    for packet in container.demux():
                        if self._stop_event.is_set():
                            break

                        # VIDEO FRAME PROCESSING
                        if packet.stream.type == 'video':
                            for frame in packet.decode():
                                loop_start = time.time()
                                img_bgr = frame.to_ndarray(format='bgr24')

                                # 1. Pre-Roll Puffer
                                video_buffer.append(img_bgr.copy())

                                # 2. YOLO Detection auf GPU (CUDA)
                                person_detected = False
                                if detector:
                                    results = detector(img_bgr, verbose=False, classes=DETECTION_CLASSES, device=device_target)
                                    person_detected = len(results[0].boxes) > 0

                                # 3. STATE MACHINE
                                if state == "IDLE":
                                    if person_detected:
                                        state = "RECORDING"
                                        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                        video_file_path = os.path.join(ALERTS_DIR, f"{self.name}_EVENT_{ts_str}.mp4")
                                        self.logger.warning("🚨 [DETECTED] Person found! Starting recording (NVENC + Audio).")

                                        h, w = img_bgr.shape[:2]
                                        try:
                                            out_container = av.open(video_file_path, mode='w')
                                            
                                            # NVENC mit Libx264 Fallback
                                            try:
                                                out_video = out_container.add_stream('h264_nvenc', rate=TARGET_FPS)
                                            except Exception:
                                                self.logger.warning("⚠️ NVENC unavailable, falling back to libx264.")
                                                out_video = out_container.add_stream('libx264', rate=TARGET_FPS)

                                            out_video.width = w
                                            out_video.height = h
                                            out_video.pix_fmt = 'yuv420p'

                                            # Audio Stream aus RTMP einrichten (falls vorhanden)
                                            audio_in_stream = next((s for s in container.streams if s.type == 'audio'), None)
                                            if audio_in_stream:
                                                out_audio = out_container.add_stream('aac')

                                            # Pre-Roll Video & Audio rausschreiben
                                            for hist_v in video_buffer:
                                                encode_video_frame(hist_v)
                                            for hist_a in audio_buffer:
                                                encode_audio_frame(hist_a)

                                        except Exception as e:
                                            self.logger.error(f"❌ Failed to initialize video writer: {e}")
                                            close_writer()
                                            state = "IDLE"

                                elif state == "RECORDING":
                                    if person_detected:
                                        encode_video_frame(img_bgr)
                                    else:
                                        state = "POST_ROLL"
                                        post_roll_end_time = time.time() + POST_ROLL_SEC
                                        self.logger.info(f"🏠 [GONE] Person departed. Monitoring for {POST_ROLL_SEC}s extra.")
                                        encode_video_frame(img_bgr)

                                elif state == "POST_ROLL":
                                    if person_detected:
                                        state = "RECORDING"
                                        self.logger.info("🚨 [DETECTED] Person returned! Resuming recording.")
                                        encode_video_frame(img_bgr)
                                    else:
                                        encode_video_frame(img_bgr)
                                        if time.time() > post_roll_end_time:
                                            self.logger.info(f"✅ Session ended for {self.name}. Closing file.")
                                            close_writer()
                                            state = "IDLE"

                                # Frame-Rate Kontrolle
                                elapsed = time.time() - loop_start
                                sleep_duration = max(0, (1.0 / TARGET_FPS) - elapsed)
                                if sleep_duration > 0:
                                    time.sleep(sleep_duration)

                        # AUDIO FRAME PROCESSING
                        elif packet.stream.type == 'audio':
                            for a_frame in packet.decode():
                                audio_buffer.append(a_frame)
                                if state in ["RECORDING", "POST_ROLL"]:
                                    encode_audio_frame(a_frame)

                except Exception as e:
                    self.logger.error(f"⚠️ [STREAM LOST] '{self.name}': {e}. Retrying in 5s...")
                    close_writer()
                    state = "IDLE"
                    if container:
                        try: container.close()
                        except: pass
                    container = None
                    time.sleep(5)

        except Exception as e:
            self.logger.error(f"💥 Process Crash [{self.name}]: {e}")
        finally:
            close_writer()
            if container:
                try: container.close()
                except: pass
            self.logger.info("🛑 Agent process shutting down.")

    def stop_agent(self):
        self._stop_event.set()


# ---------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------
if __name__ == "__main__":
    system_logger = get_stream_logger("SYSTEM")
    system_logger.info("🚀 [MASTER] Initializing Multi-Agent Pipeline...")

    all_agents = []
    for stream in STREAMS:
        if stream.get("enabled", False):
            agent_proc = CameraAgent(stream)
            agent_proc.start()
            all_agents.append(agent_proc)
            system_logger.info(f"📡 [MASTER] Launched Process Worker for: {stream['name']}")
        else:
            system_logger.info(f"⏭️ [MASTER] Skipping Disabled Stream: {stream['name']}")

    if not all_agents:
        system_logger.error("❌ No active streams found! Exiting.")
        sys.exit(1)

    system_logger.info("[MASTER] All processes running in parallel. Monitoring ACTIVE.")

    try:
        while True:
            alive_count = sum(1 for p in all_agents if p.is_alive())
            if alive_count < len(all_agents):
                system_logger.warning(f"⚠️ ALERT: {len(all_agents) - alive_count} camera process(es) died!")
            time.sleep(15)
    except KeyboardInterrupt:
        system_logger.info("[MASTER] Shutdown signal received.")

    for a in all_agents:
        a.stop_agent()
        a.join(timeout=2)
        if a.is_alive():
            a.terminate()

    system_logger.info("[MASTER] Pipeline shutdown complete.")
