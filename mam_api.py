"""
mam_api.py — externe API zur Remote-Steuerung von IDguard PRO.

Erlaubt fremden Systemen, Video/Audio/Bilder von außen einzureichen,
die IDguard PRO komplett durch dieselbe Pipeline schickt wie eine eigene
Aufnahme (Codec-Absicherung, Filmstrip, KI-Analyse, Gesichtserkennung),
mit Job-spezifischen Parametern statt der globalen Settings, und liefert
das Ergebnis per Webhook-Callback UND per Status-Abfrage zurück.

Als eigenes Flask-Blueprint gebaut statt alles in web_ui.py zu packen --
klare Trennung, eigene Auth (API-Keys, nicht die Dashboard-Session), und
web_ui.py ist mit fast 2000 Zeilen ohnehin schon groß genug.

AUTH-MODELL: API-Keys werden gehasht gespeichert (wie Passwörter) -- der
Klartext-Key wird nur EINMAL beim Erzeugen angezeigt, danach ist er nicht
mehr abrufbar. Ein Leak von api_keys.json allein reicht nicht, um die
Keys selbst zu rekonstruieren.
"""
import os
import sys
import json
import time
import uuid
import hashlib
import secrets
import threading
import subprocess
from functools import wraps

from flask import Blueprint, request, jsonify, send_file, g

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
try:
    from config import ALERTS_DIR, SETTINGS_F
except ImportError:
    ALERTS_DIR = "./alerts"
    SETTINGS_F = "pipeline_settings.json"

API_KEYS_PATH = os.path.join(DIR, "api_keys.json")
JOBS_DIR = os.path.join(DIR, ".mam_jobs")
JOBS_UPLOAD_DIR = os.path.join(JOBS_DIR, "uploads")
JOBS_OUTPUT_DIR = os.path.join(JOBS_DIR, "output")

os.makedirs(JOBS_UPLOAD_DIR, exist_ok=True)
os.makedirs(JOBS_OUTPUT_DIR, exist_ok=True)

mam_bp = Blueprint("mam_api", __name__, url_prefix="/api/v1")

# Erlaubte Medientypen und ihre Dateiendungen -- alles andere wird beim
# Upload abgelehnt, statt der Pipeline etwas Unbekanntes unterzujubeln.
ALLOWED_VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts"}
ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# API-Key-Verwaltung
# ---------------------------------------------------------------------------

def _hash_key(raw_key):
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _load_api_keys():
    try:
        with open(API_KEYS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_api_keys(keys):
    with open(API_KEYS_PATH, "w") as f:
        json.dump(keys, f, indent=2)


def generate_api_key(label):
    """Erzeugt einen neuen API-Key, speichert nur dessen Hash. Gibt den
    Klartext-Key EINMALIG zurück -- das ist die einzige Gelegenheit, ihn zu
    sehen, danach ist er aus IDguard PRO selbst nicht mehr rekonstruierbar."""
    raw_key = "idg_" + secrets.token_urlsafe(32)
    keys = _load_api_keys()
    keys[_hash_key(raw_key)] = {
        "label": label or "Unnamed key",
        "created_at": time.time(),
        "last_used_at": None,
    }
    _save_api_keys(keys)
    return raw_key


def revoke_api_key(key_hash):
    keys = _load_api_keys()
    if key_hash in keys:
        del keys[key_hash]
        _save_api_keys(keys)
        return True
    return False


def list_api_keys():
    """Für die GUI: Label + Metadaten, NIE den Key selbst (der ist nach der
    Erzeugung ohnehin nur noch als Hash vorhanden)."""
    keys = _load_api_keys()
    return [
        {"key_hash": h, "label": v.get("label"), "created_at": v.get("created_at"),
         "last_used_at": v.get("last_used_at")}
        for h, v in keys.items()
    ]


def _touch_key_last_used(key_hash):
    keys = _load_api_keys()
    if key_hash in keys:
        keys[key_hash]["last_used_at"] = time.time()
        _save_api_keys(keys)


def requires_api_key(f):
    """Prüft Authorization: Bearer <key> ODER X-API-Key: <key> -- beide
    gängigen Konventionen unterstützt, damit möglichst viele externe
    Systeme ohne Anpassung funktionieren."""
    @wraps(f)
    def decorated(*args, **kwargs):
        raw_key = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            raw_key = auth_header[len("Bearer "):].strip()
        if not raw_key:
            raw_key = request.headers.get("X-API-Key", "").strip()
        if not raw_key:
            return jsonify({"error": "Missing API key (Authorization: Bearer <key> or X-API-Key header)."}), 401
        key_hash = _hash_key(raw_key)
        keys = _load_api_keys()
        if key_hash not in keys:
            return jsonify({"error": "Invalid API key."}), 401
        g.api_key_hash = key_hash
        g.api_key_label = keys[key_hash].get("label")
        _touch_key_last_used(key_hash)
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Job-Zustand
# ---------------------------------------------------------------------------

def _job_path(job_id):
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def _load_job(job_id):
    try:
        with open(_job_path(job_id)) as f:
            return json.load(f)
    except Exception:
        return None


def _save_job(job):
    with open(_job_path(job["job_id"]), "w") as f:
        json.dump(job, f, indent=2)


def _create_job(media_type, original_filename, params):
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "status": "queued",  # queued -> processing -> done | failed
        "media_type": media_type,
        "original_filename": original_filename,
        "params": params,
        "submitted_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "result": None,
        "callback_url": params.get("callback_url"),
        "callback_delivered": False,
        "callback_attempts": 0,
    }
    _save_job(job)
    return job


# ---------------------------------------------------------------------------
# Verarbeitung -- läuft in einem Hintergrund-Thread pro Job, damit der
# einreichende HTTP-Request sofort mit der job_id antworten kann, statt auf
# die komplette Analyse zu warten (die je nach Ollama-Last mehrere Minuten
# dauern kann).
# ---------------------------------------------------------------------------

def _process_video_job(job_id, upload_path, params):
    job = _load_job(job_id)
    job["status"] = "processing"
    job["started_at"] = time.time()
    _save_job(job)

    try:
        import watch_folder
        import ai_analyze
        import face_recognize
        import backfill_filmstrips

        # Dieselbe Codec-Absicherung wie beim Watchfolder-Import -- Job kann
        # von irgendeinem fremden System kommen, dessen Video-Codec nicht
        # zwingend browser-/pipeline-kompatibel ist.
        mp4_path = watch_folder._ensure_mp4(upload_path, logger=print)
        if not mp4_path:
            raise RuntimeError("Could not process the uploaded video (unsupported or corrupt file).")

        basename = f"mam_{job_id}"
        dest_path = os.path.join(JOBS_OUTPUT_DIR, basename + ".mp4")
        if mp4_path != dest_path:
            os.replace(mp4_path, dest_path)
        if upload_path != mp4_path and os.path.exists(upload_path):
            os.remove(upload_path)

        # Filmstrip -- dieselbe Funktion wie überall sonst im System.
        thumbs_root = os.path.join(JOBS_OUTPUT_DIR, ".thumbs")
        os.makedirs(thumbs_root, exist_ok=True)
        backfill_filmstrips.backfill_filmstrip(dest_path, thumbs_root, 12)

        # KI-Analyse -- mit Job-spezifischen Themen, falls mitgegeben,
        # sonst exakt dasselbe Verhalten wie bei einer normalen Aufnahme
        # (globale Settings-Themen).
        topics_override = params.get("topics") or None
        ai_analyze.analyze(basename, JOBS_OUTPUT_DIR, topics_override=topics_override)

        # Gesichtserkennung -- optional, falls im Job angefordert.
        if params.get("detect_faces", True):
            try:
                face_recognize.recognize(basename, JOBS_OUTPUT_DIR)
            except Exception as e:
                print(f"⚠️ [API] Gesichtserkennung für Job {job_id} fehlgeschlagen: {e}")

        meta_path = os.path.join(JOBS_OUTPUT_DIR, f"{basename}.ai.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        job["status"] = "done"
        job["finished_at"] = time.time()
        job["result"] = {
            "video_filename": basename + ".mp4",
            "description": meta.get("description"),
            "topics": meta.get("topics"),
            "transcript": meta.get("transcript"),
            "faces": meta.get("faces"),
        }
    except Exception as e:
        job["status"] = "failed"
        job["finished_at"] = time.time()
        job["error"] = str(e)
        print(f"❌ [API] Job {job_id} fehlgeschlagen: {e}")
    finally:
        _save_job(job)
        _deliver_callback(job)


def _deliver_callback(job, max_attempts=5):
    """Feuert den Webhook, falls konfiguriert -- mit Retry bei
    Fehlschlag (der Aufrufer könnte gerade neu starten o.ä.). Läuft
    NICHT blockierend für die Job-Verarbeitung selbst (wird nach deren
    Abschluss aufgerufen, aber in einem eigenen Thread, damit ein
    langsamer/unerreichbarer Callback-Empfänger nicht den Worker-Thread
    blockiert, der ja schon fertig ist -- reine Absicherung)."""
    callback_url = job.get("callback_url")
    if not callback_url:
        return

    def _attempt():
        import urllib.request
        import urllib.error
        payload = json.dumps({
            "job_id": job["job_id"],
            "status": job["status"],
            "result": job.get("result"),
            "error": job.get("error"),
        }).encode("utf-8")
        delay = 2
        for attempt in range(1, max_attempts + 1):
            try:
                req = urllib.request.Request(
                    callback_url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST"
                )
                urllib.request.urlopen(req, timeout=15)
                current = _load_job(job["job_id"])
                if current:
                    current["callback_delivered"] = True
                    current["callback_attempts"] = attempt
                    _save_job(current)
                return
            except Exception as e:
                print(f"⚠️ [API] Callback-Zustellung an {callback_url} fehlgeschlagen (Versuch {attempt}/{max_attempts}): {e}")
                time.sleep(delay)
                delay = min(delay * 2, 60)
        current = _load_job(job["job_id"])
        if current:
            current["callback_attempts"] = max_attempts
            current.setdefault("callback_delivered", False)
            _save_job(current)

    threading.Thread(target=_attempt, daemon=True).start()


# ---------------------------------------------------------------------------
# Routen
# ---------------------------------------------------------------------------

@mam_bp.route("/jobs", methods=["POST"])
@requires_api_key
def submit_job():
    """Nimmt ein Video per multipart-Upload entgegen (Feldname 'file'),
    plus optionale Formularfelder:
      topics            - kommagetrennte Liste, überschreibt für diesen
                           Job die globalen Themen-Settings
      detect_faces      - "true"/"false", Default an
      callback_url       - wird bei Fertigstellung per POST benachrichtigt
    Antwortet SOFORT mit der job_id -- die eigentliche Verarbeitung läuft
    im Hintergrund, kann je nach Ollama-Last mehrere Minuten dauern."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (expected multipart field 'file')."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext in ALLOWED_VIDEO_EXT:
        media_type = "video"
    elif ext in ALLOWED_AUDIO_EXT:
        return jsonify({"error": "Audio-only jobs are not yet supported — video only for now."}), 400
    elif ext in ALLOWED_IMAGE_EXT:
        return jsonify({"error": "Image-only jobs are not yet supported — video only for now."}), 400
    else:
        return jsonify({"error": f"Unsupported file type '{ext}'."}), 400

    topics_raw = request.form.get("topics", "").strip()
    params = {
        "topics": [t.strip() for t in topics_raw.split(",") if t.strip()] if topics_raw else None,
        "detect_faces": request.form.get("detect_faces", "true").lower() != "false",
        "callback_url": request.form.get("callback_url", "").strip() or None,
    }

    job = _create_job(media_type, file.filename, params)
    upload_path = os.path.join(JOBS_UPLOAD_DIR, f"{job['job_id']}{ext}")
    file.save(upload_path)

    threading.Thread(target=_process_video_job, args=(job["job_id"], upload_path, params), daemon=True).start()

    return jsonify({"job_id": job["job_id"], "status": "queued"}), 202


@mam_bp.route("/jobs/<job_id>", methods=["GET"])
@requires_api_key
def get_job_status(job_id):
    """Passiver Status-Abruf -- funktioniert unabhängig vom Callback, für
    Aufrufer, die lieber pollen als einen Webhook-Endpunkt zu betreiben."""
    job = _load_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(job)


@mam_bp.route("/jobs/<job_id>/video", methods=["GET"])
@requires_api_key
def get_job_video(job_id):
    """Liefert das verarbeitete Video aus. Optionale Query-Parameter
    ?start=SEKUNDEN&end=SEKUNDEN schneiden nur diesen Ausschnitt heraus
    (per ffmpeg, Stream-Copy wo möglich -- kein Neu-Encoding, außer der
    Schnittpunkt liegt nicht exakt auf einem Keyframe, dann entscheidet
    ffmpeg selbst, ob es re-encodieren muss für einen exakten Schnitt)."""
    job = _load_job(job_id)
    if job is None or job.get("status") != "done":
        return jsonify({"error": "Job not found or not finished yet."}), 404
    video_path = os.path.join(JOBS_OUTPUT_DIR, job["result"]["video_filename"])
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file missing on disk."}), 404

    start = request.args.get("start")
    end = request.args.get("end")
    if start is None and end is None:
        return send_file(video_path, mimetype="video/mp4", conditional=True)

    try:
        start_f = float(start) if start is not None else 0.0
        segment_path = os.path.join(JOBS_OUTPUT_DIR, f"{job_id}_segment_{start}_{end}.mp4")
        if not os.path.exists(segment_path):
            args = ["ffmpeg", "-y", "-ss", str(start_f), "-i", video_path]
            if end is not None:
                duration = float(end) - start_f
                if duration <= 0:
                    return jsonify({"error": "'end' must be greater than 'start'."}), 400
                args += ["-t", str(duration)]
            args += ["-c", "copy", segment_path]
            result = subprocess.run(args, capture_output=True, text=True, timeout=60)
            if result.returncode != 0 or not os.path.exists(segment_path):
                return jsonify({"error": f"Could not extract segment: {result.stderr[-300:]}"}), 500
        return send_file(segment_path, mimetype="video/mp4", conditional=True)
    except ValueError:
        return jsonify({"error": "'start'/'end' must be numbers (seconds)."}), 400


@mam_bp.route("/jobs/<job_id>/metadata", methods=["GET"])
@requires_api_key
def get_job_metadata(job_id):
    job = _load_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404
    if job.get("status") != "done":
        return jsonify({"job_id": job_id, "status": job["status"], "error": job.get("error")})
    return jsonify({"job_id": job_id, "status": "done", **job["result"]})
