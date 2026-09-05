"""
quick_record.py — sofortige Ad-hoc-Aufnahme, komplett unabhängig von der
laufenden Pipeline. Kein YOLO, keine Zustandsmaschine, kein Pre-/Post-Roll --
nur "verbinde mit der Kamera, nimm genau N Sekunden auf, speichere, fertig".

Gedacht für "nimm das schnell für eine Minute auf", wenn die Pipeline gar
nicht läuft oder die Kamera-Erkennung aus anderen Gründen nicht in Frage
kommt. Nutzt ffmpegs eigenes -t-Flag für die exakte Dauer, statt selbst
Frames zu zählen -- das ist ffmpegs ureigenste Aufgabe und zuverlässiger als
ein eigener Timer über PyAV.

Läuft als Job (analog zu mam_api.py) mit Status-Tracking, weil die
Aufnahme je nach Dauer eine Weile läuft und der aufrufende HTTP-Request
sofort antworten soll.
"""
import os
import sys
import json
import time
import uuid
import subprocess

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
try:
    from config import ALERTS_DIR, STREAMS_F
except ImportError:
    ALERTS_DIR = "./alerts"
    STREAMS_F = "streams.json"

JOBS_DIR = os.path.join(ALERTS_DIR, ".quick_record_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

MAX_DURATION_SEC = 300  # Absicherung gegen versehentliche Stunden-Aufnahmen über diesen Weg


def _camera_url(name):
    try:
        with open(STREAMS_F) as f:
            streams = json.load(f)
    except Exception:
        return None
    for s in streams:
        if s.get("name") == name:
            return s.get("url")
    return None


def _ffmpeg_input_args(url):
    """CLI-Äquivalent zu _build_open_options() in recorder_pipeline.py --
    dieselbe Protokoll-Logik, nur als Kommandozeilen-Flags statt PyAV-Options-Dict."""
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    if scheme == "rtsp":
        return ["-rtsp_transport", "tcp", "-i", url]
    return ["-i", url]


def _job_path(job_id):
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def _save_job(job):
    with open(_job_path(job["job_id"]), "w") as f:
        json.dump(job, f, indent=2)


def load_job(job_id):
    try:
        with open(_job_path(job_id)) as f:
            return json.load(f)
    except Exception:
        return None


def start_quick_record(camera_name, duration_sec, run_ai_analysis=True):
    """Startet die Aufnahme in einem Hintergrund-Prozess, gibt sofort eine
    job_id zurück. Gibt (job_id, error) zurück -- error ist gesetzt, wenn
    die Kamera nicht gefunden wurde oder die Dauer ungültig ist, dann wird
    gar nichts gestartet."""
    duration_sec = max(1, min(int(duration_sec), MAX_DURATION_SEC))
    url = _camera_url(camera_name)
    if not url:
        return None, f"Camera '{camera_name}' not found in streams.json."

    job_id = uuid.uuid4().hex
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(ALERTS_DIR, f"{camera_name}_QUICKREC_{ts_str}.mp4")
    job = {
        "job_id": job_id, "camera": camera_name, "duration_sec": duration_sec,
        "status": "recording", "output_path": None, "started_at": time.time(),
        "finished_at": None, "error": None,
    }
    _save_job(job)

    subprocess.Popen([
        sys.executable, os.path.abspath(__file__),
        "--worker", job_id, url, str(duration_sec), output_path, str(run_ai_analysis)
    ], cwd=DIR)
    return job_id, None


def _run_worker(job_id, url, duration_sec, output_path, run_ai_analysis):
    job = load_job(job_id)
    args = ["ffmpeg", "-y"] + _ffmpeg_input_args(url) + ["-t", str(duration_sec), "-c", "copy", output_path]
    try:
        # Timeout deutlich über der Aufnahmedauer -- ffmpeg selbst beendet
        # sich nach -t von allein, das hier ist nur ein Sicherheitsnetz
        # gegen ein hängendes ffmpeg (z.B. Quelle antwortet nicht mehr).
        result = subprocess.run(args, capture_output=True, text=True, timeout=duration_sec + 30)
        if result.returncode != 0 or not os.path.exists(output_path):
            # Packet-Copy kann fehlschlagen, wenn der Quell-Codec nicht ins
            # Zielcontainer-Format passt (selten bei mp4, aber möglich) --
            # Fallback auf echtes Encoding statt komplett aufzugeben.
            fallback_args = ["ffmpeg", "-y"] + _ffmpeg_input_args(url) + [
                "-t", str(duration_sec), "-c:v", "libx264", "-c:a", "aac", output_path
            ]
            result = subprocess.run(fallback_args, capture_output=True, text=True, timeout=duration_sec + 60)
        if result.returncode != 0 or not os.path.exists(output_path):
            job["status"] = "failed"
            job["error"] = result.stderr[-500:]
        else:
            job["status"] = "done"
            job["output_path"] = output_path

            # Haupt-Thumbnail (Dashboard-Kachel) -- derselbe Dateiname wie
            # der normale "Trigger-Screenshot" der Pipeline (<basename>.jpg
            # direkt neben dem Video), nur ohne Erkennungs-Box, da hier gar
            # keine YOLO-Erkennung stattfindet. Ein Frame aus der Mitte der
            # Aufnahme statt dem allerersten -- robuster gegen einen
            # eventuellen schwarzen/leeren allerersten Frame mancher Quellen.
            thumb_path = os.path.splitext(output_path)[0] + ".jpg"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", output_path, "-ss", str(duration_sec / 2),
                     "-frames:v", "1", "-q:v", "3", thumb_path],
                    capture_output=True, timeout=30
                )
            except Exception as e:
                print(f"⚠️ [QuickRecord] Haupt-Thumbnail konnte nicht erzeugt werden: {e}")

            # Filmstrip -- dieselbe Funktion wie beim Watchfolder-Import,
            # sonst bleibt die Aufnahme im Dashboard ohne Vorschaubild
            # (nur ein leeres Icon statt eines echten Frames).
            try:
                import backfill_filmstrips
                thumbs_root = os.path.join(ALERTS_DIR, ".thumbs")
                os.makedirs(thumbs_root, exist_ok=True)
                try:
                    with open(os.path.join(DIR, "pipeline_settings.json")) as f:
                        count = int(json.load(f).get("FILMSTRIP_COUNT", 8)) or 8
                except Exception:
                    count = 8
                backfill_filmstrips.backfill_filmstrip(output_path, thumbs_root, count)
            except Exception as e:
                print(f"⚠️ [QuickRecord] Filmstrip konnte nicht erzeugt werden: {e}")

            job["ai_analysis"] = "skipped"
            if run_ai_analysis in ("True", True):
                try:
                    import ai_analyze
                    basename = os.path.splitext(os.path.basename(output_path))[0]
                    ai_analyze.analyze(basename, ALERTS_DIR)
                    meta_path = os.path.join(ALERTS_DIR, f"{basename}.ai.json")
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            meta = json.load(f)
                        if meta.get("description"):
                            job["ai_analysis"] = "done"
                            job["description"] = meta.get("description")
                        else:
                            # analyze() lief durch, aber ohne Beschreibung --
                            # z.B. weil Ollama nicht erreichbar war und der
                            # Fallback-Text griff, oder der Ollama-Request
                            # selbst leer zurückkam.
                            job["ai_analysis"] = "no_description"
                    else:
                        job["ai_analysis"] = "no_metadata_written"
                except Exception as e:
                    job["ai_analysis"] = "failed"
                    job["ai_analysis_error"] = str(e)
                    print(f"⚠️ [QuickRecord] KI-Analyse fehlgeschlagen: {e}")
    except subprocess.TimeoutExpired:
        job["status"] = "failed"
        job["error"] = "ffmpeg timed out."
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
    job["finished_at"] = time.time()
    _save_job(job)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6])
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Quick ad-hoc recording, independent of the pipeline.")
        parser.add_argument("camera", help="Camera name, as configured in streams.json")
        parser.add_argument("duration", type=int, help="Duration in seconds")
        args = parser.parse_args()
        job_id, error = start_quick_record(args.camera, args.duration)
        if error:
            print(f"❌ {error}")
            sys.exit(1)
        print(f"Started job {job_id}, recording {args.duration}s from {args.camera}...")
