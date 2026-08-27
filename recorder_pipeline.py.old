import os
import sys
import time
import signal
import threading
import datetime
import multiprocessing
from collections import deque
from fractions import Fraction

# CPU-Thread-Wildwuchs von PyTorch/OpenBLAS global drosseln
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

# Fix 4: cuDNN war hart deaktiviert (Kommentar: "cuDNN Sublibrary Mismatch"),
# was auf der RTX 5090 spürbar Performance kostet, da cuDNN die
# GPU-beschleunigten Convolutions übernimmt. Jetzt per ENV-Var umschaltbar,
# damit sich ohne Codeänderung testen lässt, ob ein aktuelles PyTorch/cuDNN-
# Paar das eigentliche Versionsproblem behebt: DISABLE_CUDNN=0 python3 ...
DISABLE_CUDNN = os.environ.get("DISABLE_CUDNN", "0") == "1"

# 1. PATH RESOLUTION
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)

try:
    from config import (
        STREAMS, ALERTS_DIR, MODEL_PATH, PRE_ROLL_SEC,
        POST_ROLL_SEC, TARGET_FPS, DETECTION_CLASSES, CONFIDENCE_THRESHOLD,
        get_stream_logger, system_logger, YOLO_VERSION
    )
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not load config.py: {e}")
    sys.exit(1)


class GracefulShutdown(BaseException):
    """Eigene Exception statt Exception-Basisklasse, damit sie nicht versehentlich
    vom generischen 'except Exception' als Crash geloggt wird (siehe Fix 2)."""
    pass


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
            from config import get_stream_logger as gs, YOLO_VERSION
            self.logger = gs(self.name)
        except Exception:
            import logging
            self.logger = logging.getLogger(self.name)
            YOLO_VERSION = "v10"  # Fallback

        # Fix 2: SIGTERM (Standard-Signal von `pkill`/`kill`, siehe stop.sh) hat
        # ohne eigenen Handler die Default-Disposition "sofort beenden" — das
        # überspringt jeden Python-Code inkl. finally-Blöcken. Ohne diesen
        # Handler wird eine laufende Aufnahme beim Stoppen NIE sauber
        # geschlossen (kein Flush, potenziell kaputte MP4-Datei).
        def _handle_signal(signum, frame):
            self._stop_event.set()
            raise GracefulShutdown()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        print(f"🚀 [Process Start] Initializing agent: {self.name} (Using YOLO {YOLO_VERSION})")

        try:
            import av
            import torch

            # Umgeht den cuDNN Sublibrary Mismatch, kostet dafür GPU-Performance.
            # Siehe DISABLE_CUDNN-Kommentar oben.
            torch.backends.cudnn.enabled = not DISABLE_CUDNN
            torch.set_num_threads(2)

            from ultralytics import YOLO
        except ImportError as e:
            self.logger.error(f"❌ Dependency Error in {self.name}: {e}")
            return

        # 1. Initialize AI Engine (YOLO v10, v12 or v26) auf CUDA GPU
        detector = None
        device_target = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Fix 1: Reihenfolge war `os.path.exists(MODEL_PATH) and MODEL_PATH` —
        # os.path.exists(None) wirft TypeError, wenn MODEL_PATH mal None/leer
        # ist, BEVOR die and-Kurzschlussauswertung das prüfen kann. Erst auf
        # MODEL_PATH prüfen, dann erst exists() aufrufen.
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            try:
                detector = YOLO(MODEL_PATH)
                if device_target == "cuda:0":
                    detector.to("cuda:0")
                    self.logger.info(f"✅ AI Model (YOLO {YOLO_VERSION}) loaded successfully on CUDA GPU ({torch.cuda.get_device_name(0)}).")
                else:
                    self.logger.warning("⚠️ CUDA not available, falling back to CPU.")
            except Exception as e:
                self.logger.error(f"❌ Failed to load model ({YOLO_VERSION}) on CUDA: {e}")
        else:
            self.logger.warning("⚠️ No valid YOLO path found; running in VISION-ONLY mode.")

        # Fix 5: Gemeinsamer Pre-Roll Puffer für Video und Audio, jetzt primär
        # nach Timestamp statt nach geschätzter Item-Anzahl getrimmt. Die alte
        # Formel `(TARGET_FPS + 50) * PRE_ROLL_SEC` war nur eine Annahme über
        # die Audio-Frame-Rate — kam mehr Audio rein als angenommen, flogen
        # Video-Frames früher raus als der eingestellte PRE_ROLL_SEC vorsah,
        # ohne dass das irgendwo sichtbar wurde. maxlen bleibt als grobzügiges
        # Sicherheitsnetz gegen Speicher-Runaway, trimmt aber nicht mehr aktiv.
        safety_cap = int((TARGET_FPS + 60) * PRE_ROLL_SEC) * 3
        av_buffer = deque(maxlen=safety_cap)

        def trim_buffer():
            cutoff = time.time() - PRE_ROLL_SEC
            while av_buffer and av_buffer[0][2] < cutoff:
                av_buffer.popleft()

        state = "IDLE"
        post_roll_end_time = 0
        container = None
        out_container = None
        out_video = None
        out_audio = None
        resampler = None
        video_frame_count = 0

        def close_writer():
            nonlocal out_container, out_video, out_audio, resampler, video_frame_count
            if out_container:
                try:
                    # Flush Resampler zuerst
                    if out_audio and resampler:
                        try:
                            for rf in resampler.resample(None):
                                rf.pts = None
                                for packet in out_audio.encode(rf):
                                    out_container.mux(packet)
                        except Exception as e:
                            self.logger.warning(f"⚠️ Audio resampler flush error: {e}")

                    # Flush Encoders
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
                    resampler = None
                    video_frame_count = 0
                    av_buffer.clear()

        def encode_video_frame(img_bgr):
            nonlocal video_frame_count
            if not out_container or not out_video:
                return
            try:
                av_frame = av.VideoFrame.from_ndarray(img_bgr, format="bgr24")
                av_frame.pts = video_frame_count
                video_frame_count += 1

                for packet in out_video.encode(av_frame):
                    out_container.mux(packet)
            except Exception as e:
                self.logger.error(f"❌ Video encoding error: {e}")

        def encode_audio_frame(a_frame):
            nonlocal resampler
            if not out_container or not out_audio:
                return
            try:
                if resampler is None:
                    resampler = av.AudioResampler(
                        format=out_audio.format.name,
                        layout=out_audio.layout.name,
                        rate=out_audio.rate
                    )

                resampled_frames = resampler.resample(a_frame)
                for rf in resampled_frames:
                    rf.pts = None
                    for packet in out_audio.encode(rf):
                        out_container.mux(packet)
            except Exception as e:
                self.logger.error(f"❌ Audio encoding error: {e}")

        def write_buffered_item(item_type, data):
            if item_type == "video":
                encode_video_frame(data)
            elif item_type == "audio":
                encode_audio_frame(data)

        try:
            while not self._stop_event.is_set():
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

                try:
                    for packet in container.demux():
                        if self._stop_event.is_set():
                            break

                        # VIDEO FRAME PROCESSING
                        if packet.stream.type == 'video':
                            for frame in packet.decode():
                                img_bgr = frame.to_ndarray(format='bgr24')
                                now = time.time()

                                av_buffer.append(("video", img_bgr.copy(), now))
                                trim_buffer()

                                # Fix 6: "person_detected"/"Person" umbenannt, da
                                # DETECTION_CLASSES inzwischen beliebige Objektarten
                                # sein können (Tiere, Pakete, ...) — die alten Logs
                                # sagten irreführend "Person found", auch wenn z.B.
                                # eine Katze erkannt wurde.
                                target_detected = False
                                if detector:
                                    # HIER wird die CONFIDENCE_THRESHOLD direkt genutzt
                                    results = detector(img_bgr, verbose=False, classes=DETECTION_CLASSES, conf=CONFIDENCE_THRESHOLD, device=device_target)
                                    target_detected = len(results[0].boxes) > 0

                                if state == "IDLE":
                                    if target_detected:
                                        state = "RECORDING"
                                        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                        os.makedirs(ALERTS_DIR, exist_ok=True)
                                        video_file_path = os.path.join(ALERTS_DIR, f"{self.name}_EVENT_{ts_str}.mp4")
                                        self.logger.warning(f"🚨 [DETECTED] Target object found! Starting recording (YOLO {YOLO_VERSION}, NVENC + Audio).")

                                        h, w = img_bgr.shape[:2]
                                        try:
                                            out_container = av.open(video_file_path, mode='w')

                                            try:
                                                out_video = out_container.add_stream('h264_nvenc', rate=TARGET_FPS)
                                            except Exception:
                                                self.logger.warning("⚠️ NVENC unavailable, falling back to libx264.")
                                                out_video = out_container.add_stream('libx264', rate=TARGET_FPS)

                                            out_video.width = w
                                            out_video.height = h
                                            out_video.pix_fmt = 'yuv420p'
                                            out_video.time_base = Fraction(1, TARGET_FPS)

                                            audio_in_stream = next((s for s in container.streams if s.type == 'audio'), None)
                                            if audio_in_stream:
                                                out_audio = out_container.add_stream('aac')
                                                out_audio.rate = audio_in_stream.rate if audio_in_stream.rate else 44100
                                                out_audio.layout = audio_in_stream.layout.name if audio_in_stream.layout else 'stereo'

                                            for item_type, data, _ts in av_buffer:
                                                write_buffered_item(item_type, data)

                                        except Exception as e:
                                            self.logger.error(f"❌ Failed to initialize video writer: {e}")
                                            close_writer()
                                            state = "IDLE"

                                elif state == "RECORDING":
                                    if target_detected:
                                        encode_video_frame(img_bgr)
                                    else:
                                        state = "POST_ROLL"
                                        post_roll_end_time = time.time() + POST_ROLL_SEC
                                        self.logger.info(f"🏠 [GONE] Target object left frame. Monitoring for {POST_ROLL_SEC}s extra.")
                                        encode_video_frame(img_bgr)

                                elif state == "POST_ROLL":
                                    if target_detected:
                                        state = "RECORDING"
                                        self.logger.info("🚨 [DETECTED] Target object returned! Resuming recording.")
                                        encode_video_frame(img_bgr)
                                    else:
                                        encode_video_frame(img_bgr)
                                        if time.time() > post_roll_end_time:
                                            self.logger.info(f"✅ Session ended for {self.name}. Closing file.")
                                            close_writer()
                                            state = "IDLE"

                        # AUDIO FRAME PROCESSING
                        elif packet.stream.type == 'audio':
                            for a_frame in packet.decode():
                                now = time.time()
                                av_buffer.append(("audio", a_frame, now))
                                trim_buffer()
                                if state in ["RECORDING", "POST_ROLL"]:
                                    encode_audio_frame(a_frame)

                except GracefulShutdown:
                    raise
                except Exception as e:
                    self.logger.error(f"⚠️ [STREAM LOST] '{self.name}': {e}. Retrying in 5s...")
                    close_writer()
                    state = "IDLE"
                    if container:
                        try:
                            container.close()
                        except Exception:
                            pass
                    container = None
                    time.sleep(5)

        except GracefulShutdown:
            self.logger.info(f"🛑 [{self.name}] Shutdown-Signal empfangen, schließe sauber ab...")
        except Exception as e:
            self.logger.error(f"💥 Process Crash [{self.name}]: {e}")
        finally:
            close_writer()
            if container:
                try:
                    container.close()
                except Exception:
                    pass
            self.logger.info("🛑 Agent process shutting down.")

    def stop_agent(self):
        self._stop_event.set()


# ---------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------
if __name__ == "__main__":
    system_logger = get_stream_logger("SYSTEM")
    system_logger.info("🚀 [MASTER] Initializing Multi-Agent Pipeline...")

    # Fix 2 (Master-Seite): stop.sh killt per `pkill -f "recorder_pipeline.py"`,
    # was auch den Master-Prozess selbst trifft (SIGTERM, nicht SIGINT) — ohne
    # eigenen Handler wäre `except KeyboardInterrupt` unten wirkungslos und die
    # Cleanup-Schleife für alle Worker würde nie laufen.
    shutdown_requested = threading.Event()

    def _handle_master_signal(signum, frame):
        system_logger.info(f"[MASTER] Signal {signum} empfangen — fahre Pipeline sauber herunter...")
        shutdown_requested.set()

    signal.signal(signal.SIGTERM, _handle_master_signal)
    signal.signal(signal.SIGINT, _handle_master_signal)

    # Fix 3: statt nur (proc) merken wir uns auch die Stream-Config, damit ein
    # abgestürzter Worker mit denselben Settings automatisch neu gestartet
    # werden kann, statt nur eine Warnung zu loggen.
    agents = []
    for stream in STREAMS:
        if stream.get("enabled", False):
            agent_proc = CameraAgent(stream)
            agent_proc.start()
            agents.append({'process': agent_proc, 'stream': stream})
            system_logger.info(f"📡 [MASTER] Launched Process Worker for: {stream['name']}")
        else:
            system_logger.info(f"⏭️ [MASTER] Skipping Disabled Stream: {stream['name']}")

    if not agents:
        system_logger.error("❌ No active streams found! Exiting.")
        sys.exit(1)

    system_logger.info("[MASTER] All processes running in parallel. Monitoring ACTIVE.")

    while not shutdown_requested.is_set():
        for entry in agents:
            proc = entry['process']
            if not proc.is_alive():
                stream = entry['stream']
                exitcode = proc.exitcode
                system_logger.warning(
                    f"⚠️ [MASTER] Worker '{stream['name']}' ist beendet (exitcode={exitcode}) — starte automatisch neu..."
                )
                new_proc = CameraAgent(stream)
                new_proc.start()
                entry['process'] = new_proc

        # Interruptible statt time.sleep(15): reagiert sofort auf ein Signal
        # statt bis zu 15s zu blockieren.
        shutdown_requested.wait(15)

    system_logger.info("[MASTER] Shutting down all workers...")
    for entry in agents:
        proc = entry['process']
        proc.stop_agent()
        proc.join(timeout=5)
        if proc.is_alive():
            system_logger.warning(f"⚠️ [MASTER] '{entry['stream']['name']}' reagiert nicht — erzwinge Terminate.")
            proc.terminate()
            proc.join(timeout=2)

    system_logger.info("[MASTER] Pipeline shutdown complete.")
