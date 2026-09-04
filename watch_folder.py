#!/usr/bin/env python3
"""
watch_folder.py - Überwacht einen konfigurierbaren Import-Ordner auf neue
Videodateien und importiert sie automatisch, sobald sie vollständig
geschrieben sind (Modus 2 — "warten bis fertig", siehe Diskussion zu Modus 1
"wachsende Datei als Stream lesen", der bewusst NICHT hier implementiert ist:
funktioniert nur zuverlässig bei streambaren Containern wie MPEG-TS, nicht
bei MP4, wo der moov-Atom oft erst am Dateiende steht).

Läuft als eigener Prozess, vom Master (recorder_pipeline.py) genauso
gestartet wie ein CameraAgent, wenn WATCH_FOLDER_ENABLED aktiv ist — nutzt
dieselbe stop.sh/start_detached.sh-Lebenszyklus-Verwaltung automatisch mit.

Datei-Vollständigkeits-Erkennung: Linux hat kein zuverlässiges Betriebssystem-
Signal dafür. Üblicher, robuster Ansatz (auch von rsync/Samba-Übertragungen
so gehandhabt): Dateigröße wird periodisch geprüft, erst nach
WATCH_FOLDER_STABILITY_SEC Sekunden ohne Änderung gilt die Datei als fertig.
"""
import os
import sys
import time
import glob
import json
import shutil
import signal
import subprocess
import multiprocessing

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
try:
    from config import ALERTS_DIR, SETTINGS_F, DETECTION_CLASSES
except ImportError:
    ALERTS_DIR = "./alerts"
    SETTINGS_F = "pipeline_settings.json"
    DETECTION_CLASSES = [0]

import backfill_filmstrips

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts")
POLL_INTERVAL_SEC = 2.0


def _load_settings():
    try:
        with open(SETTINGS_F) as f:
            return json.load(f)
    except Exception:
        return {}


class GracefulShutdown(BaseException):
    pass


def _ensure_mp4(src_path, logger=print):
    """Falls die Quelldatei nicht schon .mp4 ist: Container per ffmpeg
    umverpacken (-c copy, KEIN Neu-Encoding — dieselbe Packet-Copy-Philosophie
    wie beim Rest des Systems), sonst würde alles Nachgelagerte (Dauer-
    Anzeige, Filmstrip-Erzeugung, Player-Kompatibilität im Dashboard) auf
    einer Datei mit falscher Endung aufsetzen. Bei bereits-.mp4 keine
    Kopie/Umwandlung nötig, Originaldatei wird direkt weiterverwendet."""
    if src_path.lower().endswith(".mp4"):
        return src_path
    tmp_out = os.path.splitext(src_path)[0] + "__remux.mp4"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-c", "copy", tmp_out],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0 or not os.path.exists(tmp_out) or os.path.getsize(tmp_out) == 0:
            logger(f"⚠️ [Watchfolder] Remux nach MP4 fehlgeschlagen für {src_path}: {result.stderr[-300:]}")
            return None
        return tmp_out
    except Exception as e:
        logger(f"⚠️ [Watchfolder] Remux-Fehler für {src_path}: {e}")
        return None


def _passes_detection_filter(video_path, run_detection, model, logger=print):
    """Optionaler YOLO-Vorfilter: nur behalten, wenn eine der konfigurierten
    DETECTION_CLASSES irgendwo im Video vorkommt. Ohne aktivierten Filter
    (Standard) wird jede importierte Datei bedingungslos behalten — der
    Watchfolder ist dann ein reiner "alles reinkopierte landet im System"-
    Import, kein Ereignis-Filter wie bei den Live-Kameras."""
    if not run_detection or model is None:
        return True
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        # 8 gleichmäßig verteilte Stichproben statt jedes Frames -- reicht für
        # eine grobe Ja/Nein-Entscheidung, ohne das ganze Video zu dekodieren.
        sample_count = 8
        found = False
        for i in range(sample_count):
            frame_idx = int(total * i / sample_count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            results = model(frame, verbose=False)
            for box in results[0].boxes:
                if int(box.cls[0]) in DETECTION_CLASSES:
                    found = True
                    break
            if found:
                break
        cap.release()
        return found
    except Exception as e:
        logger(f"⚠️ [Watchfolder] Erkennungs-Vorfilter fehlgeschlagen, importiere sicherheitshalber trotzdem: {e}")
        return True


def process_file(src_path, source_name, delete_source, run_detection, model, logger=print):
    """Ein einzelner, vollständig geschriebener Fund aus dem Import-Ordner
    wird hierdurch komplett durchgeschleust: Container-Absicherung, optionaler
    Erkennungs-Filter, Umbenennung in die Standard-Namenskonvention, Filmstrip,
    Trigger-Screenshot, und schließlich postprocess.py für die KI-Analyse."""
    logger(f"📥 [Watchfolder] Neue Datei gefunden: {src_path}")

    mp4_path = _ensure_mp4(src_path, logger)
    if not mp4_path:
        return False

    if not _passes_detection_filter(mp4_path, run_detection, model, logger):
        logger(f"⏭️ [Watchfolder] Kein relevantes Objekt gefunden, verworfen: {src_path}")
        if mp4_path != src_path:
            os.remove(mp4_path)
        if delete_source:
            os.remove(src_path)
        return True

    ts = time.strftime("%Y%m%d_%H%M%S")
    basename = f"{source_name}_EVENT_{ts}"
    dest_path = os.path.join(ALERTS_DIR, basename + ".mp4")
    # Kollisionsschutz -- zwei Importe in derselben Sekunde sind unwahrscheinlich,
    # aber ein Zahlenanhängsel kostet nichts und verhindert ein stilles Überschreiben.
    n = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(ALERTS_DIR, f"{basename}_{n}.mp4")
        n += 1
        basename = os.path.splitext(os.path.basename(dest_path))[0]

    was_remuxed = (mp4_path != src_path)
    if delete_source or was_remuxed:
        # Entweder soll die Quelle sowieso weg (delete_source), oder mp4_path
        # ist bereits ein neu erzeugtes Remux-Temp-Derivat -- in beiden Fällen
        # spricht nichts dagegen, die Datei zu verschieben statt zu kopieren.
        shutil.move(mp4_path, dest_path)
    else:
        # Kein Remux nötig UND die Originaldatei soll im Import-Ordner
        # erhalten bleiben -- kopieren, nicht verschieben. Vorher hier ein
        # echter Bug: mp4_path == src_path bei bereits-.mp4-Dateien führte zu
        # shutil.move() unabhängig von delete_source, die Originaldatei
        # verschwand also selbst dann aus dem Import-Ordner, wenn der Nutzer
        # explizit "Original behalten" eingestellt hatte.
        shutil.copy2(mp4_path, dest_path)
    if delete_source and was_remuxed and os.path.exists(src_path):
        os.remove(src_path)

    # Trigger-Screenshot: ein einzelner Frame, analog zum Live-Pfad, damit die
    # Kachel im Dashboard nicht leer bleibt.
    try:
        thumb_path = os.path.splitext(dest_path)[0] + ".jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "1", "-i", dest_path, "-frames:v", "1", thumb_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
        )
    except Exception:
        pass

    # Filmstrip -- dieselbe Funktion, die auch backfill_filmstrips.py nutzt.
    thumbs_root = os.path.join(ALERTS_DIR, ".thumbs")
    os.makedirs(thumbs_root, exist_ok=True)
    count = int(_load_settings().get("FILMSTRIP_COUNT", 8)) or 8
    backfill_filmstrips.backfill_filmstrip(dest_path, thumbs_root, count)

    # Kleine Trigger-Metadaten -- markiert klar als Import, nicht als
    # "erkanntes Objekt", damit das im Dashboard nicht mit einer echten
    # Live-Erkennung verwechselt wird.
    try:
        with open(os.path.splitext(dest_path)[0] + ".trigger.json", "w") as f:
            json.dump({"source": "watch_folder_import", "original_filename": os.path.basename(src_path)}, f)
    except Exception:
        pass

    logger(f"✅ [Watchfolder] Importiert als {os.path.basename(dest_path)}")

    # KI-Analyse anstoßen -- derselbe GPU-gesperrte Pfad wie bei normalen
    # Aufnahmen, damit sich Watchfolder-Importe und Live-Kameras nicht um die
    # GPU streiten.
    try:
        subprocess.Popen([sys.executable, os.path.join(DIR, "postprocess.py"), basename, ALERTS_DIR])
    except Exception as e:
        logger(f"⚠️ [Watchfolder] postprocess.py konnte nicht gestartet werden: {e}")

    return True


class WatchFolderAgent(multiprocessing.Process):
    """Eigener Prozess, analog zu CameraAgent -- vom Master gestartet, wenn
    WATCH_FOLDER_ENABLED in den Settings aktiv ist."""

    def __init__(self):
        super().__init__(daemon=False)
        self._stop_event = multiprocessing.Event()

    def stop_agent(self):
        self._stop_event.set()

    def run(self):
        def _handle_signal(signum, frame):
            self._stop_event.set()
            raise GracefulShutdown()
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        print("🚀 [Watchfolder] Prozess gestartet.")
        model = None
        seen = {}  # path -> (size, last_change_ts)

        try:
            while not self._stop_event.is_set():
                settings = _load_settings()
                folder = (settings.get("WATCH_FOLDER_PATH") or "").strip()
                source_name = (settings.get("WATCH_FOLDER_SOURCE_NAME") or "Import").strip() or "Import"
                stability_sec = float(settings.get("WATCH_FOLDER_STABILITY_SEC", 5) or 5)
                delete_source = bool(settings.get("WATCH_FOLDER_DELETE_SOURCE", False))
                run_detection = bool(settings.get("WATCH_FOLDER_RUN_DETECTION", False))

                if not folder or not os.path.isdir(folder):
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                if run_detection and model is None:
                    try:
                        from ultralytics import YOLO
                        from config import MODEL_FILENAME
                        model = YOLO(MODEL_FILENAME)
                    except Exception as e:
                        print(f"⚠️ [Watchfolder] YOLO-Modell für Vorfilter konnte nicht geladen werden, importiere ungefiltert: {e}")

                current_files = set()
                for ext in VIDEO_EXTENSIONS:
                    current_files.update(glob.glob(os.path.join(folder, f"*{ext}")))

                now = time.time()
                for path in current_files:
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        continue
                    if path not in seen:
                        seen[path] = (size, now)
                    else:
                        old_size, last_change = seen[path]
                        if size != old_size:
                            seen[path] = (size, now)
                        elif now - last_change >= stability_sec:
                            process_file(path, source_name, delete_source, run_detection, model)
                            seen.pop(path, None)

                # Verwaiste Einträge (Datei zwischenzeitlich verschwunden) aufräumen
                for path in list(seen.keys()):
                    if path not in current_files:
                        seen.pop(path, None)

                time.sleep(POLL_INTERVAL_SEC)
        except GracefulShutdown:
            print("🛑 [Watchfolder] Shutdown-Signal empfangen.")
        except Exception as e:
            print(f"💥 [Watchfolder] Prozess-Crash: {e}")
        finally:
            print("🛑 [Watchfolder] Prozess beendet.")


if __name__ == "__main__":
    agent = WatchFolderAgent()
    agent.run()
