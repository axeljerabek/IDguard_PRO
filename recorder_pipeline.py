import json
import os
import sys
import subprocess
import time
import signal
import random
import threading
import datetime
import multiprocessing
from collections import deque
from fractions import Fraction

try:
    from audio_trigger import AudioTrigger
except ImportError:
    AudioTrigger = None  # Optionales Feature — Pipeline läuft unverändert ohne es

# CPU-Thread-Wildwuchs von PyTorch/OpenBLAS global drosseln
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

# Fix 4: cuDNN war hart deaktiviert (Kommentar: "cuDNN Sublibrary Mismatch"),
# was auf einer aktuellen GPU spürbar Performance kostet, da cuDNN die
# GPU-beschleunigten Convolutions übernimmt. Jetzt wird cuDNN standardmäßig
# versucht — mit einem automatischen Selbsttest beim Modell-Laden (siehe
# unten): schlägt der fehl, wird cuDNN pro Prozess automatisch deaktiviert
# und neu geladen, statt manuell raten zu müssen. DISABLE_CUDNN=1 erzwingt
# weiterhin das alte, garantiert sichere Verhalten ohne jeden Selbsttest.
DISABLE_CUDNN = os.environ.get("DISABLE_CUDNN", "0") == "1"

# 1. PATH RESOLUTION
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)

try:
    from config import (
        STREAMS, ALERTS_DIR, MODEL_PATH, PRE_ROLL_SEC, SETTINGS_F,
        POST_ROLL_SEC, TARGET_FPS, DETECTION_CLASSES, CONFIDENCE_THRESHOLD,
        get_stream_logger, system_logger, YOLO_VERSION
    )
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not load config.py: {e}")
    sys.exit(1)

# Für den REC-Indikator im Dashboard: aktueller Zustand pro Stream, von
# web_ui.py gelesen (kein *.mp4-Glob-Konflikt durch führenden Punkt).
STATUS_DIR = os.path.join(ALERTS_DIR, '.status')
os.makedirs(STATUS_DIR, exist_ok=True)

def _load_filmstrip_settings():
    """FILMSTRIP_COUNT=0 -> Feature aus. Live aus der Settings-Datei gelesen,
    kein Pipeline-Neustart bei Änderung nötig."""
    try:
        with open(SETTINGS_F) as f:
            d = json.load(f)
        return int(d.get('FILMSTRIP_COUNT', 0)), float(d.get('FILMSTRIP_INTERVAL_SEC', 2.0))
    except Exception:
        return 0, 2.0

def _postprocessing_enabled():
    """Ob überhaupt Grund besteht, postprocess.py zu starten — Vision-Analyse,
    Transkription ODER Gesichtserkennung reicht schon, postprocess.py
    entscheidet dann selbst pro Schritt anhand seines eigenen Enabled-Flags
    weiter."""
    try:
        with open(SETTINGS_F) as f:
            s = json.load(f)
        return (bool(s.get('AI_ANALYSIS_ENABLED', False))
                or bool(s.get('TRANSCRIPTION_ENABLED', False))
                or bool(s.get('FACE_RECOGNITION_ENABLED', False)))
    except Exception:
        return False

def _load_settings_dict():
    """Komplettes Settings-Dict, roh — für Features wie AudioTrigger, die
    mehrere Werte auf einmal live abfragen wollen."""
    try:
        with open(SETTINGS_F) as f:
            return json.load(f)
    except Exception:
        return {}

def _write_state(name, state):
    try:
        with open(os.path.join(STATUS_DIR, f'{name}.json'), 'w') as f:
            json.dump({'state': state}, f)
    except Exception:
        pass


class GracefulShutdown(BaseException):
    """Eigene Exception statt Exception-Basisklasse, damit sie nicht versehentlich
    vom generischen 'except Exception' als Crash geloggt wird (siehe Fix 2)."""
    pass


# 2. Class Definition for the Camera Agent
class CameraAgent(multiprocessing.Process):
    def __init__(self, stream_info, half_precision=True):
        super().__init__()
        self.name = stream_info["name"]
        self.url = stream_info.get("url", "")
        self.enabled = stream_info.get("enabled", False)
        # Default True — bestehende streams.json-Einträge von vor diesem
        # Feature haben das Feld noch nicht, sollen sich aber nicht plötzlich
        # stumm schalten.
        self.audio_enabled = stream_info.get("audio_enabled", True)
        # Vom Master anhand der tatsächlich verbauten GPU bestimmt (siehe
        # detect_gpu_profile) — nicht pro Worker neu geraten.
        self.half_precision_allowed = half_precision

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
            import numpy as np
            import cv2  # bereits Ultralytics-Abhängigkeit, für Trigger-Screenshots genutzt

            torch.set_num_threads(2)
            cv2.setNumThreads(2)  # sonst nutzt cv2 (Resize/JPEG-Encode für Thumbnails,
            # Filmstrip, Shared-Frames) unkontrolliert alle Kerne — pro Kamera-Prozess
            # echte CPU-Konkurrenz mit den bereits gedeckelten Torch/OMP-Threads.

            from ultralytics import YOLO
        except ImportError as e:
            self.logger.error(f"❌ Dependency Error in {self.name}: {e}")
            return

        # 1. Initialize AI Engine (YOLO v10, v12 or v26) auf CUDA GPU
        detector = None
        device_target = "cuda:0" if torch.cuda.is_available() else "cpu"
        half_enabled = False

        def _load_and_selftest(use_cudnn, use_half):
            """Lädt das Modell mit gegebenem cuDNN-/FP16-Zustand und führt einen
            winzigen Dummy-Inferenzlauf aus. Deckt cuDNN- oder FP16-
            Versionskonflikte sofort beim Start auf, statt erst mitten im
            Stream-Loop zu crashen."""
            torch.backends.cudnn.enabled = use_cudnn
            m = YOLO(MODEL_PATH)
            if device_target == "cuda:0":
                m.to("cuda:0")
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            # 'half=' ist in aktuellen Ultralytics-Versionen deprecated
            # (Warnung: "Use 'quantize' instead") — quantize=16 ist das
            # Äquivalent für FP16. Nur mitgeben wenn gewünscht, statt
            # quantize=None zu raten.
            if use_half:
                m(dummy, verbose=False, device=device_target, quantize=16)
            else:
                m(dummy, verbose=False, device=device_target)
            return m

        # Fix 1: Reihenfolge war `os.path.exists(MODEL_PATH) and MODEL_PATH` —
        # os.path.exists(None) wirft TypeError, wenn MODEL_PATH mal None/leer
        # ist, BEVOR die and-Kurzschlussauswertung das prüfen kann. Erst auf
        # MODEL_PATH prüfen, dann erst exists() aufrufen.
        def _try_load_model():
            """Versucht das Modell zu laden — von voller Performance (cuDNN +
            FP16) stufenweise abwärts bis zu einer Kombination, die auf DIESER
            Hardware tatsächlich funktioniert. Geht über jede GPU-Generation
            von RTX 2060 bis RTX 5090 sicher, ohne dass man vorher wissen
            muss, welche Kombination auf der jeweiligen Maschine läuft."""
            if device_target != "cuda:0":
                # Reines CPU-Vision: kein cuDNN/FP16 relevant
                try:
                    m = YOLO(MODEL_PATH)
                    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                    m(dummy, verbose=False, device=device_target)
                    self.logger.warning("⚠️ CUDA not available, falling back to CPU.")
                    return m, False
                except Exception as e:
                    self.logger.error(f"❌ Failed to load model ({YOLO_VERSION}) on CPU: {e}")
                    return None, False

            attempts = []
            if not DISABLE_CUDNN:
                if self.half_precision_allowed:
                    attempts.append((True, True))
                attempts.append((True, False))
            if self.half_precision_allowed:
                attempts.append((False, True))
            attempts.append((False, False))

            last_error = None
            for use_cudnn, use_half in attempts:
                try:
                    m = _load_and_selftest(use_cudnn, use_half)
                    self.logger.info(
                        f"✅ AI Model (YOLO {YOLO_VERSION}) auf CUDA GPU geladen "
                        f"({torch.cuda.get_device_name(0)}), cuDNN={'aktiv' if use_cudnn else 'inaktiv'}, "
                        f"FP16={'aktiv' if use_half else 'inaktiv'}."
                    )
                    return m, use_half
                except Exception as e:
                    last_error = e
                    self.logger.warning(
                        f"⚠️ Selbsttest fehlgeschlagen mit cuDNN={'an' if use_cudnn else 'aus'}/"
                        f"FP16={'an' if use_half else 'aus'} ({e}) — versuche schwächere Kombination..."
                    )
                    if device_target == "cuda:0":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

            self.logger.error(f"❌ Failed to load model ({YOLO_VERSION}) in jeder getesteten Kombination: {last_error}")
            return None, False

        if MODEL_PATH and os.path.exists(MODEL_PATH):
            detector, half_enabled = _try_load_model()
        else:
            self.logger.warning("⚠️ No valid YOLO path found; running in VISION-ONLY mode.")

        # Einmaliger echter NVENC-Test beim Start dieses Kamera-Prozesses.
        # add_stream('h264_nvenc', ...) allein ist KEIN zuverlässiger
        # Verfügbarkeits-Check — PyAV/ffmpeg öffnen den eigentlichen Encoder
        # (avcodec_open2) oft erst beim ERSTEN echten encode()-Aufruf, nicht
        # schon bei add_stream(). Ohne diesen Test hier würde add_stream()
        # klaglos durchlaufen, die Pipeline meldet fälschlich "NVENC aktiv",
        # und der eigentliche Fehler taucht erst still mitten in der
        # laufenden Aufnahme auf — bei JEDEM einzelnen Frame erneut, weit
        # außerhalb des dafür vorgesehenen Fallback-try/except (genau das
        # Muster, das im Docker-Container ohne funktionierenden NVENC-Zugriff
        # beobachtet wurde). Gleiche Philosophie wie _load_and_selftest oben:
        # echt testen statt nur hoffen.
        def _probe_nvenc():
            try:
                import tempfile
                # 320x240, nicht 64x64: manche NVENC-Generationen/Treiber
                # lehnen zu kleine Auflösungen ab (unterhalb einer nicht
                # überall gleich dokumentierten Mindestgröße), was den Test
                # fälschlich als "NVENC kaputt" melden würde, obwohl der
                # Encoder bei den tatsächlichen Aufnahme-Auflösungen (720p/
                # 1080p) einwandfrei funktionieren könnte.
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as tmp:
                    probe_container = av.open(tmp.name, mode='w')
                    probe_stream = probe_container.add_stream('h264_nvenc', rate=30)
                    probe_stream.width = 320
                    probe_stream.height = 240
                    probe_stream.pix_fmt = 'yuv420p'
                    probe_frame = av.VideoFrame(width=320, height=240, format='yuv420p')
                    list(probe_stream.encode(probe_frame))
                    list(probe_stream.encode(None))  # flush
                    probe_container.close()
                return True
            except Exception as e:
                self.logger.warning(f"⚠️ [{self.name}] NVENC-Test fehlgeschlagen, nutze libx264 (CPU-Encoding) für diese Kamera: {e}")
                return False

        nvenc_available = _probe_nvenc()
        if nvenc_available:
            self.logger.info(f"🎮 [{self.name}] NVENC-Test erfolgreich — Aufnahmen nutzen GPU-Encoding.")

        # Optionaler Audio-Trigger (CLAP, mehrere frei wählbare Kategorien
        # gleichzeitig). Läuft immer als Hintergrund-Thread, prüft aber selbst
        # laufend AUDIO_TRIGGER_ENABLED aus den Settings — so greift Ein/Aus
        # und Kategorien-Änderung live, ohne Pipeline-Neustart. Lädt das
        # eigentliche Modell erst lazy beim ersten Aktivieren.
        audio_trigger = None
        if AudioTrigger is not None and self.audio_enabled:
            try:
                audio_trigger = AudioTrigger(self.logger, self.name)
                audio_trigger.start(lambda: _load_settings_dict())
            except Exception as e:
                self.logger.warning(f"⚠️ [{self.name}] Audio-Trigger konnte nicht gestartet werden: {e}")
                audio_trigger = None
        elif AudioTrigger is not None and not self.audio_enabled:
            self.logger.info(f"🔇 [{self.name}] Audio-Erkennung für diese Kamera deaktiviert.")

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

        # Sicherheitsnetz für das wichtigste Kriterium: die Pipeline MUSS
        # aufzeichnen, sobald ein Trigger erkannt wird. Konnte das Modell beim
        # Start nicht geladen werden (z.B. kurzzeitiger GPU/Treiber-Hänger),
        # würde ohne diesen Retry NIE wieder etwas erkannt — der Stream liefe
        # bis zum manuellen Neustart nur im "VISION-ONLY"-Blindflug weiter.
        MODEL_RETRY_INTERVAL = 60  # Sekunden zwischen Nachlade-Versuchen
        last_model_retry = time.time()

        # Frame-Rate-Drosselung: bisher wurde JEDER vom Quell-Stream gelieferte
        # Frame durch YOLO gejagt und encodiert, unabhängig von TARGET_FPS.
        # Liefert die Kamera nativ z.B. 25-30fps, aber TARGET_FPS steht auf 15,
        # lief die Inferenz also fast doppelt so oft wie nötig — UND die
        # Ausgabedatei bekam PTS als wäre sie exakt mit TARGET_FPS aufgenommen,
        # obwohl tatsächlich mit Quell-FPS geschrieben wurde (falsche
        # Wiedergabegeschwindigkeit/-dauer). Jetzt werden überzählige
        # Quell-Frames VOR der (teuren) BGR-Konvertierung übersprungen.
        frame_interval = 1.0 / TARGET_FPS if TARGET_FPS and TARGET_FPS > 0 else 0.0
        last_processed_time = 0.0

        # Geteilter Live-Frame für web_ui.py/helpers.py: schreibt periodisch (Rate
        # aus THUMBNAIL_FPS) ein JPEG, damit die Web-UI NICHT mehr selbst eine
        # zweite RTMP-Verbindung pro Kamera aufmachen und decodieren muss.
        FRAMES_DIR = os.path.join(ALERTS_DIR, '.frames')
        os.makedirs(FRAMES_DIR, exist_ok=True)
        shared_frame_next_time = 0
        shared_frame_interval = 1.0
        shared_frame_last_check = 0
        show_boxes_live = True

        def write_shared_frame(img_bgr, results=None):
            nonlocal shared_frame_next_time, shared_frame_interval, shared_frame_last_check, show_boxes_live
            now2 = time.time()
            if now2 - shared_frame_last_check > 5.0:
                try:
                    with open(SETTINGS_F) as f:
                        d = json.load(f)
                    fps = float(d.get('THUMBNAIL_FPS', 1.0))
                    shared_frame_interval = 1.0 / fps if fps > 0 else 1.0
                    show_boxes_live = bool(d.get('SHOW_DETECTION_BOXES', True))
                except Exception:
                    shared_frame_interval = 1.0
                shared_frame_last_check = now2
            if now2 < shared_frame_next_time:
                return
            try:
                source = img_bgr
                if show_boxes_live and results:
                    # Fällt bei JEDEM Problem sofort aufs Rohbild zurück — die
                    # Live-Vorschau ist rein kosmetisch, darf nie etwas anderes stören.
                    try:
                        source = results[0].plot()
                    except Exception:
                        source = img_bgr
                small = cv2.resize(source, (640, max(1, int(source.shape[0] * 640 / source.shape[1]))))
                ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    tmp = os.path.join(FRAMES_DIR, f'.{self.name}.tmp')
                    with open(tmp, 'wb') as f:
                        f.write(buf.tobytes())
                    os.replace(tmp, os.path.join(FRAMES_DIR, f'{self.name}.jpg'))
                shared_frame_next_time = now2 + shared_frame_interval
            except Exception:
                pass

        state = "IDLE"
        post_roll_end_time = 0
        container = None
        out_container = None
        out_video = None
        out_audio = None
        resampler = None
        video_frame_count = 0
        recording_start_time = 0
        last_pts = -1
        audio_samples_written = 0
        audio_first_frame_time = None
        _debug_audio_calls = 0  # TEMPORÄR — für die Diagnose, danach wieder raus

        # Filmstrip (Hover-Scrub-Vorschau + AI-taugliche Großbilder): pro
        # Recording neu gesetzt, siehe RECORDING-Start weiter unten.
        fs_small_dir = None
        fs_large_dir = None
        filmstrip_count_target = 0
        filmstrip_interval = 2.0
        filmstrip_taken_total = 0   # ALLE seit Recording-Start gesehenen Kandidaten (fürs Reservoir Sampling)
        filmstrip_timestamps = {}  # slot_idx (str) -> Sekunden seit Recording-Start, für korrekte Zeitreihenfolge trotz Slot-Überschreibung
        filmstrip_next_time = 0

        def close_writer():
            nonlocal out_container, out_video, out_audio, resampler, video_frame_count, last_pts, audio_samples_written, audio_first_frame_time
            if out_container:
                try:
                    # Flush Resampler zuerst — explizite, fortlaufende PTS statt
                    # None, damit am Dateiende keine Unstetigkeit zur sonst
                    # überall expliziten Audio-PTS-Vergabe entsteht.
                    if out_audio and resampler:
                        try:
                            for rf in resampler.resample(None):
                                rf.pts = audio_samples_written
                                audio_samples_written += rf.samples
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
                    last_pts = -1
                    audio_samples_written = 0
                    audio_first_frame_time = None
                    av_buffer.clear()

        def encode_video_frame(img_bgr, ts=None):
            nonlocal video_frame_count, last_pts
            if not out_container or not out_video:
                return
            try:
                t = ts if ts is not None else time.time()
                elapsed = max(0.0, t - recording_start_time)
                # Zwei Ziele gleichzeitig, die sich mit EINER reinen Wall-Clock-
                # PTS-Berechnung gegenseitig im Weg standen:
                # 1) Ein echter Stall (Netz-Hänger) muss im Video als Lücke
                #    sichtbar bleiben, nicht einfach verschluckt werden.
                # 2) Normales, kleines Verarbeitungs-Jitter (YOLO-Inferenz,
                #    Filmstrip-I/O, GPU-Konkurrenz zwischen mehreren Kamera-
                #    Prozessen) darf NICHT zu ungleichmäßigem PTS führen — das
                #    erzeugt sichtbares Ruckeln, obwohl die Quelle sauber
                #    30fps liefert.
                # Lösung, analog zum Audio-PTS-Gap-Fix: normal einen reinen
                # Frame-Zähler hochzählen (glatt, jitter-unempfindlich), nur
                # bei einer ECHTEN Abweichung (>0.5s zwischen erwarteter
                # Zähler-Position und tatsächlicher Wall-Clock-Zeit) auf die
                # Wall-Clock-Zeit springen, damit der Stall sichtbar bleibt.
                expected_pts = video_frame_count
                wall_clock_pts = int(elapsed * TARGET_FPS)
                if wall_clock_pts - expected_pts > TARGET_FPS * 0.5:
                    pts = wall_clock_pts
                else:
                    pts = expected_pts
                if pts <= last_pts:
                    pts = last_pts + 1
                last_pts = pts

                av_frame = av.VideoFrame.from_ndarray(img_bgr, format="bgr24")
                av_frame.pts = pts
                video_frame_count += 1

                for packet in out_video.encode(av_frame):
                    out_container.mux(packet)
            except Exception as e:
                self.logger.error(f"❌ Video encoding error: {e}")

        def capture_filmstrip(img_bgr, results=None):
            """Small (Hover-Scrub, MIT Boxen) + Large (KI-Analyse, bewusst ROH ohne
            Boxen — sauberere Eingabe für Ollama) Frames im Intervall.

            Hybrid aus Reservoir Sampling + garantiertem Ende-Slot:
            - Der LETZTE Slot wird bei JEDEM Aufruf überschrieben — zeigt also
              immer den zuletzt aufgenommenen Frame. Garantiert, dass das Ende
              einer Aktion nie fehlt, egal wie lange sie dauert.
            - Die übrigen Slots nutzen Reservoir Sampling (Algorithm R): jeder
              Kandidat hat eine mit der Zeit abnehmende Chance, einen davon zu
              ersetzen — Ergebnis: gleichmäßige Verteilung über die gesamte
              bisherige Dauer, egal ob 10 Sekunden oder 30 Minuten.
            Reines Reservoir Sampling allein GARANTIERT die Ende-Abdeckung nicht
            (nur im statistischen Mittel) — deshalb der feste Ende-Slot zusätzlich.
            """
            nonlocal filmstrip_taken_total, filmstrip_next_time
            if not fs_small_dir or filmstrip_count_target <= 0:
                return
            now = time.time()
            if now < filmstrip_next_time:
                return
            try:
                reservoir_size = filmstrip_count_target - 1 if filmstrip_count_target >= 2 else filmstrip_count_target
                end_slot = filmstrip_count_target - 1 if filmstrip_count_target >= 2 else None

                filmstrip_taken_total += 1
                slot = None
                if reservoir_size > 0:
                    if filmstrip_taken_total <= reservoir_size:
                        slot = filmstrip_taken_total - 1
                    else:
                        j = random.randint(0, filmstrip_taken_total - 1)
                        if j < reservoir_size:
                            slot = j

                h, w = img_bgr.shape[:2]
                annotated = img_bgr
                if results:
                    try:
                        annotated = results[0].plot()
                    except Exception:
                        annotated = img_bgr
                small_full = cv2.resize(annotated, (560, max(1, int(h * 560 / w))))
                large_full = img_bgr if w <= 1280 else cv2.resize(img_bgr, (1280, max(1, int(h * 1280 / w))))

                slots_to_write = set()
                if slot is not None:
                    slots_to_write.add(slot)
                if end_slot is not None:
                    slots_to_write.add(end_slot)

                for s in slots_to_write:
                    cv2.imwrite(os.path.join(fs_small_dir, f'{s:04d}.jpg'), small_full)
                    cv2.imwrite(os.path.join(fs_large_dir, f'{s:04d}.jpg'), large_full)
                    filmstrip_timestamps[str(s)] = round(now - recording_start_time, 2)

                if slots_to_write:
                    ts_path = os.path.join(os.path.dirname(fs_small_dir), 'timestamps.json')
                    with open(ts_path, 'w') as tf:
                        json.dump(filmstrip_timestamps, tf)

                filmstrip_next_time = now + filmstrip_interval
            except Exception:
                pass

        def encode_audio_frame(a_frame, ts=None):
            nonlocal resampler, audio_samples_written, audio_first_frame_time, _debug_audio_calls
            if not out_container or not out_audio:
                return
            try:
                if resampler is None:
                    resampler = av.AudioResampler(
                        format=out_audio.format.name,
                        layout=out_audio.layout.name,
                        rate=out_audio.rate
                    )

                t = ts if ts is not None else time.time()
                if audio_first_frame_time is None:
                    audio_first_frame_time = t
                    audio_samples_written = 0

                expected_elapsed = audio_samples_written / out_audio.rate
                actual_elapsed = t - audio_first_frame_time
                if actual_elapsed - expected_elapsed > 3.0:
                    gap_samples = int((actual_elapsed - expected_elapsed) * out_audio.rate)
                    audio_samples_written += gap_samples

                resampled_frames = resampler.resample(a_frame)
                for rf in resampled_frames:
                    if _debug_audio_calls < 20:
                        self.logger.info(
                            f"🔍 AUDIO-DEBUG #{_debug_audio_calls}: t={t:.4f} first_t={audio_first_frame_time:.4f} "
                            f"actual_elapsed={actual_elapsed:.4f} expected_elapsed={expected_elapsed:.4f} "
                            f"out_audio.rate={out_audio.rate} rf.rate={rf.rate} rf.samples={rf.samples} "
                            f"rf.time_base={rf.time_base} out_audio.time_base={out_audio.time_base} "
                            f"a_frame.rate={a_frame.rate} a_frame.samples={a_frame.samples} "
                            f"samples_written_before={audio_samples_written}"
                        )
                        _debug_audio_calls += 1
                    rf.pts = audio_samples_written
                    audio_samples_written += rf.samples
                    for packet in out_audio.encode(rf):
                        out_container.mux(packet)
            except Exception as e:
                self.logger.error(f"❌ Audio encoding error: {e}")

        def write_buffered_item(item_type, data, ts=None):
            if item_type == "video":
                encode_video_frame(data, ts)
            elif item_type == "audio":
                encode_audio_frame(data, ts)

        # NVDEC-Hardware-Decode vorbereiten (Punkt "GPU voll nutzen" — bisher
        # lag utilization.decoder konstant bei 0%). Das HWAccel-Objekt selbst
        # wird nur EINMAL pro Prozess konstruiert (defensiv gegen abweichende
        # PyAV-Versionen: schlägt Import/Konstruktion fehl, ist NVDEC auf
        # dieser Maschine grundsätzlich nicht verfügbar — dauerhaft aus).
        # Schlägt aber nur EIN Verbindungsversuch fehl (z.B. weil die Cam
        # gerade kurz weg ist), bleibt hw_device erhalten: bei jedem neuen
        # Reconnect wird NVDEC erneut probiert, sobald der Stream wieder da
        # ist — nur der jeweils fehlgeschlagene Versuch selbst fällt auf
        # Software-Decode zurück.
        hw_device = None
        try:
            from av.codec.hwaccel import HWAccel
            hw_device = HWAccel(device_type='cuda', device='0', allow_software_fallback=True)
            self.logger.info(f"🎮 [{self.name}] NVDEC-Hardware-Decode wird versucht.")
        except Exception as e:
            self.logger.info(f"ℹ️ [{self.name}] NVDEC nicht verfügbar ({e}) — nutze Software-Decoding (PyAV-Version prüfen für Hardware-Decode).")
            hw_device = None

        # Zählt NVDEC-Fehlversuche IN FOLGE (ohne zwischenzeitlichen Erfolg).
        # Kamera kurz weg -> ein paar Fehlversuche, dann klappt's wieder ->
        # Zähler wird zurückgesetzt, NVDEC bleibt aktiv. Erst wenn NVDEC
        # mehrfach hintereinander NIE erfolgreich verbindet, deutet das auf
        # ein grundsätzliches Problem hin (nicht auf eine flackernde Cam) —
        # dann erst dauerhaft abschalten, um nicht endlos sinnlos zu retryen.
        nvdec_fail_streak = 0
        NVDEC_FAIL_THRESHOLD = 5
        using_nvdec = False

        try:
            while not self._stop_event.is_set():
                if container is None:
                    self.logger.info(f"🔗 Attempting connection to RTMP: {self.url}")
                    open_options = {"rtmp_live": "live", "rw_timeout": "5000000"}
                    using_nvdec = False
                    try:
                        if hw_device is not None:
                            container = av.open(self.url, options=open_options, hwaccel=hw_device)
                            using_nvdec = True
                            nvdec_fail_streak = 0
                            self.logger.info(f"✅ [CONNECTED] '{self.name}' via NVDEC established stream at {self.url}")
                        else:
                            container = av.open(self.url, options=open_options)
                            self.logger.info(f"✅ [CONNECTED] '{self.name}' established stream at {self.url} (Software-Decode)")
                    except Exception as e:
                        if hw_device is not None:
                            nvdec_fail_streak += 1
                            if nvdec_fail_streak >= NVDEC_FAIL_THRESHOLD:
                                self.logger.warning(
                                    f"⚠️ [{self.name}] NVDEC ist {nvdec_fail_streak}x in Folge ohne jeden Erfolg "
                                    f"fehlgeschlagen ({e}) — deaktiviere Hardware-Decode dauerhaft für diesen Prozess."
                                )
                                hw_device = None
                            else:
                                self.logger.warning(
                                    f"⚠️ [{self.name}] NVDEC-Verbindung fehlgeschlagen ({e}, {nvdec_fail_streak}/{NVDEC_FAIL_THRESHOLD}) — "
                                    f"versuche diesen Versuch mit Software-Decode, NVDEC wird beim nächsten Reconnect erneut probiert."
                                )
                            try:
                                container = av.open(self.url, options=open_options)
                                self.logger.info(f"✅ [CONNECTED] '{self.name}' established stream at {self.url} (Software-Decode, NVDEC-Fallback)")
                            except Exception as e2:
                                self.logger.error(f"❌ [CONNECTION FAILED] '{self.name}': {e2}. Retrying in 5s...")
                                container = None
                                time.sleep(5)
                                continue
                        else:
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
                                now = time.time()

                                # Drosselung zuerst (vor der BGR-Konvertierung!), damit
                                # übersprungene Quell-Frames auch den Konvertierungs-
                                # Overhead sparen, nicht nur die Inferenz.
                                if frame_interval > 0 and (now - last_processed_time) < frame_interval:
                                    continue
                                last_processed_time = now

                                img_bgr = frame.to_ndarray(format='bgr24')

                                av_buffer.append(("video", img_bgr.copy(), now))
                                trim_buffer()

                                # Modell nachladen, falls es beim Start (oder nach einem
                                # vorherigen Fehlversuch) noch nicht verfügbar war —
                                # siehe Kommentar bei MODEL_RETRY_INTERVAL weiter oben.
                                if detector is None and time.time() - last_model_retry > MODEL_RETRY_INTERVAL:
                                    last_model_retry = time.time()
                                    self.logger.info(f"🔄 [{self.name}] Erneuter Versuch, das KI-Modell zu laden...")
                                    if MODEL_PATH and os.path.exists(MODEL_PATH):
                                        detector, half_enabled = _try_load_model()
                                        if detector is not None:
                                            self.logger.info(f"✅ [{self.name}] KI-Modell nachträglich geladen — Erkennung ist jetzt wieder aktiv.")

                                # Fix 6: "person_detected"/"Person" umbenannt, da
                                # DETECTION_CLASSES inzwischen beliebige Objektarten
                                # sein können (Tiere, Pakete, ...) — die alten Logs
                                # sagten irreführend "Person found", auch wenn z.B.
                                # eine Katze erkannt wurde.
                                target_detected = False
                                results = None
                                if detector:
                                    # HIER wird die CONFIDENCE_THRESHOLD direkt genutzt.
                                    # quantize=16 statt half=True (deprecated, siehe
                                    # Kommentar im Selbsttest oben) — sonst spammt die
                                    # Deprecation-Warnung bei JEDEM Frame ins Log.
                                    if half_enabled:
                                        results = detector(img_bgr, verbose=False, classes=DETECTION_CLASSES, conf=CONFIDENCE_THRESHOLD, device=device_target, quantize=16)
                                    else:
                                        results = detector(img_bgr, verbose=False, classes=DETECTION_CLASSES, conf=CONFIDENCE_THRESHOLD, device=device_target)
                                    target_detected = len(results[0].boxes) > 0

                                # Erst NACH der Detection: write_shared_frame bekommt die
                                # Ergebnisse mit, damit die Live-Vorschau (Grid + Lightbox)
                                # optional Erkennungs-Boxen zeigen kann — kostet keine
                                # zusätzliche Inferenz, nutzt nur das bereits berechnete Ergebnis.
                                write_shared_frame(img_bgr, results)

                                # Audio-Trigger klinkt sich hier NUR an das Ergebnis an —
                                # keine eigene State-Machine, keine eigene Aufnahme-Logik.
                                # is_triggered() liest nur ein Flag, das der Hintergrund-
                                # Thread setzt — kein blockierender Aufruf.
                                audio_triggered_now, audio_label, audio_score = (False, None, None)
                                if audio_trigger is not None:
                                    audio_triggered_now, audio_label, audio_score = audio_trigger.is_triggered()
                                    if audio_triggered_now and not target_detected:
                                        self.logger.warning(f"🔊 [{self.name}] Aufnahme durch Audio-Trigger ausgelöst: '{audio_label}'")
                                target_detected = target_detected or audio_triggered_now

                                if state == "IDLE":
                                    if target_detected:
                                        state = "RECORDING"
                                        _write_state(self.name, "RECORDING")
                                        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                        os.makedirs(ALERTS_DIR, exist_ok=True)
                                        video_file_path = os.path.join(ALERTS_DIR, f"{self.name}_EVENT_{ts_str}.mp4")
                                        self.logger.warning(f"🚨 [DETECTED] Target object found! Starting recording (YOLO {YOLO_VERSION}, NVENC + Audio).")

                                        # Zeit-Nullpunkt fürs PTS: ältester Pre-Roll-Frame,
                                        # damit dessen echte Zeitstempel korrekt einsortiert werden.
                                        recording_start_time = av_buffer[0][2] if av_buffer else time.time()
                                        last_pts = -1

                                        # Trigger-Screenshot mit eingezeichneter Erkennungs-Box
                                        # speichern (gleicher Basisname, .jpg) — wird von
                                        # web_ui.py automatisch als Vorschaubild im Dashboard
                                        # (Recent Recordings + Archiv) angezeigt.
                                        try:
                                            thumb_path = os.path.splitext(video_file_path)[0] + '.jpg'
                                            annotated = results[0].plot() if results else img_bgr
                                            cv2.imwrite(thumb_path, annotated)

                                            # Konfidenz + Klasse der stärksten Erkennung als kleines
                                            # Sidecar — fürs Badge auf dem Thumbnail im Dashboard.
                                            trigger_meta = {}
                                            if results and len(results[0].boxes) > 0:
                                                confs = results[0].boxes.conf.tolist()
                                                clss = results[0].boxes.cls.tolist()
                                                top_idx = confs.index(max(confs))
                                                trigger_meta['confidence'] = round(float(confs[top_idx]), 3)
                                                trigger_meta['class'] = str(results[0].names[int(clss[top_idx])])
                                            if audio_triggered_now and audio_label:
                                                trigger_meta['audio_trigger'] = audio_label
                                                if audio_score is not None:
                                                    trigger_meta['audio_confidence'] = round(float(audio_score), 3)
                                            if trigger_meta:
                                                meta_path = os.path.splitext(video_file_path)[0] + '.trigger.json'
                                                with open(meta_path, 'w') as mf:
                                                    json.dump(trigger_meta, mf)
                                        except Exception as e:
                                            self.logger.warning(f"⚠️ [{self.name}] Trigger-Screenshot konnte nicht gespeichert werden: {e}")

                                        filmstrip_count_target, filmstrip_interval = _load_filmstrip_settings()
                                        if filmstrip_count_target > 0:
                                            fs_name = os.path.splitext(os.path.basename(video_file_path))[0]
                                            fs_small_dir = os.path.join(ALERTS_DIR, '.thumbs', fs_name, 'small')
                                            fs_large_dir = os.path.join(ALERTS_DIR, '.thumbs', fs_name, 'large')
                                            os.makedirs(fs_small_dir, exist_ok=True)
                                            os.makedirs(fs_large_dir, exist_ok=True)
                                        else:
                                            fs_small_dir = fs_large_dir = None
                                        filmstrip_taken_total = 0
                                        filmstrip_timestamps = {}
                                        filmstrip_next_time = time.time()

                                        h, w = img_bgr.shape[:2]
                                        # ~2s Keyframe-Abstand: macht die Event-Clips beim
                                        # Scrubben im Lightbox-Player deutlich reaktionsfreudiger
                                        # (ohne das brauchen Player oft den letzten Keyframe,
                                        # der bei sehr langen GOP-Defaults weit zurückliegen kann).
                                        gop_size = str(max(1, TARGET_FPS * 2))
                                        try:
                                            out_container = av.open(video_file_path, mode='w')

                                            if nvenc_available:
                                                out_video = out_container.add_stream('h264_nvenc', rate=TARGET_FPS)
                                                # Bewusst konservative Optionen (breite Kompatibilität über
                                                # NVENC-Generationen von Turing bis Blackwell) — keine
                                                # generationsspezifischen Preset-Namen (p1-p7 vs. ältere
                                                # Namen wie 'hq'/'ll' unterscheiden sich je nach
                                                # ffmpeg/Treiber-Version), daher separat abgesichert.
                                                try:
                                                    out_video.options = {'rc': 'vbr', 'cq': '23', 'gpu': '0', 'g': gop_size}
                                                except Exception as opt_err:
                                                    self.logger.warning(f"⚠️ NVENC-Optionen konnten nicht gesetzt werden ({opt_err}), nutze Encoder-Defaults.")
                                            else:
                                                out_video = out_container.add_stream('libx264', rate=TARGET_FPS)
                                                try:
                                                    out_video.options = {'preset': 'veryfast', 'crf': '23', 'g': gop_size}
                                                except Exception:
                                                    pass

                                            out_video.width = w
                                            out_video.height = h
                                            out_video.pix_fmt = 'yuv420p'
                                            out_video.time_base = Fraction(1, TARGET_FPS)

                                            audio_in_stream = next((s for s in container.streams if s.type == 'audio'), None)
                                            if audio_in_stream:
                                                out_audio = out_container.add_stream('aac')
                                                out_audio.rate = audio_in_stream.rate if audio_in_stream.rate else 44100
                                                out_audio.layout = audio_in_stream.layout.name if audio_in_stream.layout else 'stereo'
                                                # DER Root-Cause-Bug hinter der wild falschen Spieldauer
                                                # (169MB-Datei meldete 6:56:20, Axel fand's per ffprobe):
                                                # ohne explizite Zeitbasis nimmt PyAV/der Muxer für den
                                                # Audio-Stream einen Standardwert (empirisch bestätigt:
                                                # 1/1000, nicht 1/rate) — jedes rf.pts wird zwar KORREKT
                                                # in Samples hochgezählt (encode_audio_frame), aber beim
                                                # Schreiben als "pts * falsche_zeitbasis" interpretiert.
                                                # Bei 1024 Samples/Frame und rate=48000 macht das aus
                                                # ~21ms echter Paketlänge fälschlich 1024/1000 = 1.024s —
                                                # exakt der Faktor ~48, der bei jeder Messung auftauchte.
                                                # Video hatte diesen Bug nie, weil dort schon immer
                                                # explizit `out_video.time_base = Fraction(1, TARGET_FPS)`
                                                # gesetzt wurde (siehe zwei Zeilen oben) — bei Audio fehlte
                                                # das exakte Gegenstück komplett. Meine vorherigen PTS-
                                                # Gap-Korrektur-Fixes waren zwar für sich genommen korrekt,
                                                # konnten das aber nie beheben, weil das Problem gar nicht
                                                # in der Gap-Logik lag.
                                                out_audio.time_base = Fraction(1, out_audio.rate)

                                            for item_type, data, item_ts in av_buffer:
                                                write_buffered_item(item_type, data, item_ts)

                                        except Exception as e:
                                            self.logger.error(f"❌ Failed to initialize video writer: {e}")
                                            close_writer()
                                            state = "IDLE"
                                            _write_state(self.name, "IDLE")

                                elif state == "RECORDING":
                                    if target_detected:
                                        encode_video_frame(img_bgr, now)
                                        capture_filmstrip(img_bgr, results)
                                    else:
                                        state = "POST_ROLL"
                                        _write_state(self.name, "POST_ROLL")
                                        post_roll_end_time = time.time() + POST_ROLL_SEC
                                        self.logger.info(f"🏠 [GONE] Target object left frame. Monitoring for {POST_ROLL_SEC}s extra.")
                                        encode_video_frame(img_bgr, now)
                                        capture_filmstrip(img_bgr, results)

                                elif state == "POST_ROLL":
                                    if target_detected:
                                        state = "RECORDING"
                                        _write_state(self.name, "RECORDING")
                                        vision_hit = bool(results and len(results[0].boxes) > 0)
                                        sources = []
                                        if vision_hit:
                                            sources.append("visual detection")
                                        if audio_triggered_now:
                                            sources.append(f"audio ('{audio_label}')" if audio_label else "audio")
                                        source_desc = " + ".join(sources) if sources else "detection"
                                        self.logger.info(f"🚨 [DETECTED] Target returned ({source_desc})! Resuming recording.")
                                        encode_video_frame(img_bgr, now)
                                        capture_filmstrip(img_bgr, results)
                                    else:
                                        encode_video_frame(img_bgr, now)
                                        capture_filmstrip(img_bgr, results)
                                        if time.time() > post_roll_end_time:
                                            self.logger.info(f"✅ Session ended for {self.name}. Closing file.")
                                            close_writer()
                                            state = "IDLE"
                                            _write_state(self.name, "IDLE")
                                            if _postprocessing_enabled():
                                                try:
                                                    vb = os.path.splitext(os.path.basename(video_file_path))[0]
                                                    subprocess.Popen(
                                                        [sys.executable, os.path.join(DIR, 'postprocess.py'), vb, ALERTS_DIR]
                                                        # stdout/stderr NICHT auf DEVNULL: Fehler (z.B. Ollama nicht
                                                        # erreichbar) landen so im selben Log wie die restliche Pipeline
                                                        # statt spurlos zu verschwinden.
                                                    )
                                                except Exception as e:
                                                    self.logger.warning(f"⚠️ [{self.name}] Konnte Nachbearbeitung nicht starten: {e}")

                        # AUDIO FRAME PROCESSING
                        elif packet.stream.type == 'audio':
                            for a_frame in packet.decode():
                                now = time.time()
                                av_buffer.append(("audio", a_frame, now))
                                trim_buffer()
                                if state in ["RECORDING", "POST_ROLL"]:
                                    encode_audio_frame(a_frame, now)

                                # Audio-Trigger füttern: NUR ein billiger Buffer-Append,
                                # die eigentliche (langsame) Klassifikation läuft komplett
                                # in AudioTrigger's eigenem Hintergrund-Thread — blockiert
                                # hier nichts.
                                if audio_trigger is not None:
                                    try:
                                        samples = a_frame.to_ndarray()
                                        if samples.ndim > 1:
                                            samples = samples.mean(axis=0)
                                        samples = samples.astype(np.float32)
                                        if np.issubdtype(samples.dtype, np.integer):
                                            samples = samples / 32768.0
                                        max_abs = np.abs(samples).max() if samples.size else 0
                                        if max_abs > 4.0:  # vermutlich noch Integer-PCM (z.B. int16-Range)
                                            samples = samples / 32768.0
                                        audio_trigger.feed(samples, a_frame.sample_rate)
                                    except Exception:
                                        pass

                except GracefulShutdown:
                    raise
                except Exception as e:
                    self.logger.error(f"⚠️ [STREAM LOST] '{self.name}': {e}. Retrying in 5s...")
                    if using_nvdec:
                        # Zählt als Fehlversuch für den NVDEC-Streak — die Cam war
                        # evtl. nur kurz weg, dann wird beim nächsten Reconnect (oben)
                        # NVDEC ganz normal wieder probiert. Erst nach mehreren
                        # Fehlversuchen IN FOLGE ohne jeden Erfolg schaltet der
                        # Verbindungs-Block hw_device dauerhaft ab.
                        nvdec_fail_streak += 1
                        using_nvdec = False
                    close_writer()
                    state = "IDLE"
                    _write_state(self.name, "IDLE")
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
            if audio_trigger is not None:
                audio_trigger.stop()
            if container:
                try:
                    container.close()
                except Exception:
                    pass
            self.logger.info("🛑 Agent process shutting down.")

    def stop_agent(self):
        self._stop_event.set()


def detect_gpu_profile(logger):
    """Liest die verbaute GPU einmalig VOR dem Start der Pipeline aus und
    bestimmt sichere Defaults — von RTX 2060 (Turing) bis RTX 5090 (Blackwell).
    Läuft im Master, das Ergebnis wird an jeden CameraAgent durchgereicht,
    statt dass jeder Worker-Prozess einzeln (und potenziell widersprüchlich)
    dieselbe Erkennung nochmal macht."""
    profile = {"cuda_available": False, "half_precision": False, "name": "CPU"}
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
            cuda_version = torch.version.cuda
            profile.update({
                "cuda_available": True,
                "name": name,
                "compute_capability": f"{major}.{minor}",
                "vram_gb": vram_gb,
                "cuda_version": cuda_version,
                # Ab Volta (7.0) sind Tensor Cores für schnelles FP16 vorhanden —
                # von Turing (RTX 2060, 7.5) bis Blackwell (RTX 5090, 12.0)
                # durchgängig gegeben. Darunter lohnt sich FP16 nicht.
                "half_precision": major >= 7,
            })
            logger.info(
                f"🎮 [MASTER] GPU erkannt: {name} | Compute Capability {major}.{minor} | "
                f"{vram_gb} GB VRAM | PyTorch-CUDA {cuda_version} | FP16-Inferenz: "
                f"{'aktiv' if profile['half_precision'] else 'inaktiv (Architektur zu alt)'}"
            )
        else:
            logger.warning("⚠️ [MASTER] Keine CUDA-GPU gefunden — Pipeline läuft komplett auf CPU (deutlich langsamer).")
    except Exception as e:
        logger.warning(f"⚠️ [MASTER] GPU-Erkennung fehlgeschlagen ({e}) — nehme sicheren CPU-Fallback an.")
    return profile


# ---------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------
if __name__ == "__main__":
    # KRITISCH: muss vor jeglicher Prozess-Erzeugung passieren, und vor jedem
    # CUDA-Zugriff im Master (detect_gpu_profile() gleich unten tut genau
    # das). Linux nutzt standardmäßig 'fork' für multiprocessing — forkt der
    # Master aber NACHDEM er selbst schon CUDA berührt hat (torch.cuda.
    # is_available() etc. in detect_gpu_profile), erben die Worker-Prozesse
    # einen bereits angefassten CUDA-Kontext, der sich nicht sauber
    # re-initialisieren lässt: "Cannot re-initialize CUDA in forked
    # subprocess. To use CUDA with multiprocessing, you must use the
    # 'spawn' start method" — exakt die von PyTorch selbst empfohlene
    # Lösung, hier umgesetzt. 'spawn' startet jeden Worker als komplett
    # frischen Python-Interpreter (kein geerbter Speicherzustand), auf
    # Kosten eines minimal langsameren Prozessstarts.
    multiprocessing.set_start_method('spawn', force=True)

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

    # Hardware VOR dem Start der Pipeline auslesen (RTX 2060 bis RTX 5090) —
    # bestimmt einmalig zentral, ob FP16-Inferenz probiert werden soll.
    gpu_profile = detect_gpu_profile(system_logger)

    # Vollständiges Startup-Log: welches Modell, welche Settings, welche
    # Kameras — alles, was die Pipeline für diesen Lauf tatsächlich nutzt.
    enabled_names = [s['name'] for s in STREAMS if s.get('enabled', False)]
    disabled_names = [s['name'] for s in STREAMS if not s.get('enabled', False)]
    system_logger.info("=" * 70)
    system_logger.info("🚀 [MASTER] IDguard Pipeline Startup")
    system_logger.info(f"   KI-Modell     : YOLO {YOLO_VERSION} | Pfad: {MODEL_PATH}")
    system_logger.info(f"   Detection     : Klassen={DETECTION_CLASSES} | Confidence={CONFIDENCE_THRESHOLD}")
    system_logger.info(f"   Aufnahme      : Ziel-FPS={TARGET_FPS} | Pre-Roll={PRE_ROLL_SEC}s | Post-Roll={POST_ROLL_SEC}s")
    system_logger.info(f"   Alerts-Pfad   : {ALERTS_DIR}")
    system_logger.info(f"   cuDNN         : {'per DISABLE_CUDNN erzwungen aus' if DISABLE_CUDNN else 'wird versucht (mit Selbsttest-Fallback)'}")
    system_logger.info(f"   GPU           : {gpu_profile['name']}" + (f" ({gpu_profile.get('compute_capability')}, {gpu_profile.get('vram_gb')} GB)" if gpu_profile['cuda_available'] else ""))
    system_logger.info(f"   Aktive Kameras ({len(enabled_names)}): {', '.join(enabled_names) if enabled_names else '—'}")
    system_logger.info(f"   Inaktive Kameras ({len(disabled_names)}): {', '.join(disabled_names) if disabled_names else '—'}")
    system_logger.info("=" * 70)

    # Fix 3: statt nur (proc) merken wir uns auch die Stream-Config, damit ein
    # abgestürzter Worker mit denselben Settings automatisch neu gestartet
    # werden kann, statt nur eine Warnung zu loggen.
    agents = []
    for stream in STREAMS:
        if stream.get("enabled", False):
            agent_proc = CameraAgent(stream, half_precision=gpu_profile["half_precision"])
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
                new_proc = CameraAgent(stream, half_precision=gpu_profile["half_precision"])
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
