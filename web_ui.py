from flask import Flask, render_template, request, Response, abort, send_file
import os
import sys
import glob
import csv
import io
import subprocess
import shutil
import json
import secrets
import threading
import time
import psutil
import urllib.request
from collections import deque
from datetime import datetime

# Stellt sicher, dass das Arbeitsverzeichnis und der Import-Pfad passen
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from config import STREAMS, STREAMS_F, ALERTS_DIR, PROJECT_ROOT, SETTINGS_F, YOLO_VERSION, MODEL_SIZE, MODEL_FILENAME
from auth import requires_auth
from helpers import (
    LATEST_FRAMES, start_thumbnail_thread, is_pipeline_running,
    load_overrides, load_settings, save_overrides, format_size
)
try:
    import search_index
except ImportError:
    search_index = None  # Optionales Feature — Dashboard läuft unverändert ohne Suche
try:
    import faces_db
except ImportError:
    faces_db = None  # Optionales Feature — Dashboard läuft unverändert ohne Gesichtserkennung

app = Flask(__name__)

# Archiv-Unterordner für aufbewahrte Aufnahmen (getrennt von den aktiven Alerts)
ARCHIVE_DIR = os.path.join(ALERTS_DIR, 'archive')
os.makedirs(ARCHIVE_DIR, exist_ok=True)

def _cleanup_old_recordings():
    """Löscht unarchivierte Aufnahmen älter als RETENTION_DAYS (0 = aus).
    Nur ALERTS_DIR, nie ARCHIVE_DIR — Archivieren bedeutet bewusst 'behalten'."""
    while True:
        try:
            days = load_settings().get('RETENTION_DAYS', 0)
            if days and days > 0:
                cutoff = time.time() - days * 86400
                for f in glob.glob(os.path.join(ALERTS_DIR, '*.mp4')):
                    try:
                        if os.path.getmtime(f) < cutoff:
                            os.remove(f)
                            _remove_matching_thumbnail(f)
                    except OSError:
                        pass
        except Exception:
            pass
        time.sleep(3600)

threading.Thread(target=_cleanup_old_recordings, daemon=True).start()

# Harte Obergrenze für geladene Event-Listen, damit glob()/sort() bei Monaten
# an Aufnahmen nicht bei jedem Request/Poll unnötig groß wird.
MAX_EVENTS = 200

# CSRF-Token: pro Prozessstart neu generiert, in jede Seite eingebettet und bei
# jeder zustandsändernden POST-Route geprüft. Schützt vor klassischem CSRF von
# einer fremden Seite aus (die den Token nicht kennt), auch wenn der Browser
# gecachte Basic-Auth-Header automatisch mitschickt.
CSRF_TOKEN = secrets.token_hex(32)

def _verify_csrf():
    token = request.form.get('csrf_token', '')
    if not secrets.compare_digest(token, CSRF_TOKEN):
        abort(403)

AVAILABLE_CLASSES = {
    0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 4: "Airplane", 5: "Bus",
    6: "Train", 7: "Truck", 8: "Boat", 9: "Traffic light", 10: "Fire hydrant", 11: "Stop sign",
    12: "Parking meter", 13: "Bench", 14: "Bird", 15: "Cat", 16: "Dog", 17: "Horse",
    18: "Sheep", 19: "Cow", 20: "Elephant", 21: "Bear", 22: "Zebra", 23: "Giraffe",
    24: "Backpack", 25: "Umbrella", 26: "Handbag", 27: "Tie", 28: "Suitcase",
    29: "Frisbee", 30: "Skis", 31: "Snowboard", 32: "Sports ball", 33: "Kite",
    34: "Baseballschläger", 35: "Baseball glove", 36: "Skateboard", 37: "Surfboard",
    38: "Tennis racquet", 39: "Bottle", 40: "Wine glass", 41: "Cup", 42: "Fork",
    43: "Knife", 44: "Spoon", 45: "Bowl", 46: "Banana", 47: "Apple", 48: "Sandwich",
    49: "Orange", 50: "Broccoli", 51: "Carrot", 52: "Hotdog", 53: "Pizza", 54: "Donut",
    55: "Cake", 56: "Chair", 57: "Couch", 58: "Potted plant", 59: "Bed", 60: "Dining table",
    61: "Toilet", 62: "Television", 63: "Laptop", 64: "Mouse", 65: "Remote control",
    66: "Keyboard", 67: "Mobile phone", 68: "Microwave", 69: "Oven", 70: "Toaster",
    71: "Sink", 72: "Refrigerator", 73: "Book", 74: "Clock", 75: "Vase", 76: "Scissors",
    77: "Teddy bear", 78: "Hair dryer", 79: "Toothbrush"
}

# Worker-Thread starten
start_thumbnail_thread()

# --- Pipeline-Neustart im Hintergrund (Punkt 1: blockiert das UI nicht mehr) ---
pipeline_restart_status = {"restarting": False}
_restart_lock = threading.Lock()

def _restart_pipeline_background():
    with _restart_lock:
        pipeline_restart_status["restarting"] = True
    try:
        subprocess.run(
            ['/bin/bash', os.path.join(PROJECT_ROOT, 'stop.sh')],
            cwd=PROJECT_ROOT
        )
        subprocess.Popen(
            ['/bin/bash', os.path.join(PROJECT_ROOT, 'start_detached.sh')],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Kurze Wartezeit, damit is_pipeline_running() den neuen Prozess
        # sicher erkennt, bevor das Restarting-Flag zurückgesetzt wird.
        time.sleep(2)
    finally:
        with _restart_lock:
            pipeline_restart_status["restarting"] = False

def _read_recording_states():
    """Liest die von recorder_pipeline.py geschriebenen State-Dateien (.status/<name>.json)."""
    states = {}
    status_dir = os.path.join(ALERTS_DIR, '.status')
    for f in glob.glob(os.path.join(status_dir, '*.json')):
        try:
            with open(f) as fh:
                states[os.path.splitext(os.path.basename(f))[0]] = json.load(fh).get('state', 'IDLE')
        except Exception:
            pass
    return states

_event_cache = {}  # directory -> (expires_at, events)
EVENT_CACHE_TTL = 4  # Sekunden — knapp über dem 3s-Poll-Intervall, spart die teure

def build_event_list(directory, limit=MAX_EVENTS):
    now = time.time()
    cached = _event_cache.get(directory)
    if cached and cached[0] > now:
        return cached[1]
    events = _build_event_list(directory, limit)
    _event_cache[directory] = (now + EVENT_CACHE_TTL, events)
    return events

def _event_from_file(f):
    try:
        mtime = os.path.getmtime(f)
        size = os.path.getsize(f)
        # recorder_pipeline.py legt beim Trigger einen Screenshot mit
        # gleichem Basisnamen ab (<name>.mp4 -> <name>.jpg)
        thumb_path = os.path.splitext(f)[0] + '.jpg'
        fs_base_dir = os.path.join(os.path.dirname(f), '.thumbs', os.path.splitext(os.path.basename(f))[0])
        fs_dir = os.path.join(fs_base_dir, 'small')
        fs_count = len(glob.glob(os.path.join(fs_dir, '*.jpg'))) if os.path.isdir(fs_dir) else 0
        # Durch Reservoir Sampling (siehe recorder_pipeline.py) entspricht die
        # Dateinummer NICHT mehr zwangsläufig der zeitlichen Reihenfolge —
        # timestamps.json verrät die echte Chronologie. Fehlt sie (alte
        # Aufnahmen von vor diesem Fix), einfach numerisch sortieren.
        fs_order = list(range(fs_count))
        ts_path = os.path.join(fs_base_dir, 'timestamps.json')
        if fs_count and os.path.exists(ts_path):
            try:
                with open(ts_path) as tsf:
                    ts_map = json.load(tsf)
                fs_order = sorted(range(fs_count), key=lambda i: ts_map.get(str(i), i))
            except Exception:
                pass
        ai_desc = None
        top_topic, top_topic_conf = None, None
        detected_topics = []
        transcript = None
        ai_path = os.path.splitext(f)[0] + '.ai.json'
        if os.path.exists(ai_path):
            try:
                with open(ai_path) as af:
                    ai_meta = json.load(af)
                ai_desc = ai_meta.get('description')
                top_topic = ai_meta.get('top_topic')
                top_topic_conf = ai_meta.get('top_topic_confidence')
                detected_topics = ai_meta.get('detected_topics') or []
                transcript = ai_meta.get('transcript')
            except Exception:
                pass
        ai_pending = os.path.exists(os.path.splitext(f)[0] + '.ai.pending')
        trigger_conf, trigger_cls = None, None
        audio_trigger_label, audio_trigger_conf = None, None
        trigger_path = os.path.splitext(f)[0] + '.trigger.json'
        if os.path.exists(trigger_path):
            try:
                with open(trigger_path) as tf:
                    tmeta = json.load(tf)
                trigger_conf = tmeta.get('confidence')
                trigger_cls = tmeta.get('class')
                audio_trigger_label = tmeta.get('audio_trigger')
                audio_trigger_conf = tmeta.get('audio_confidence')
            except Exception:
                pass
        faces_summary = {'people': [], 'unnamed_count': 0}
        if faces_db is not None:
            try:
                faces_summary = faces_db.get_faces_summary_for_video(os.path.basename(f))
            except Exception:
                pass
        return {
            'filename': os.path.basename(f),
            'datetime': datetime.fromtimestamp(mtime).strftime('%d.%m.%Y %H:%M'),
            'size': format_size(size),
            'has_thumbnail': os.path.exists(thumb_path),
            'filmstrip_count': fs_count,
            'filmstrip_order': fs_order,
            'ai_description': ai_desc,
            'ai_pending': ai_pending,
            'top_topic': top_topic,
            'top_topic_confidence': top_topic_conf,
            'detected_topics': detected_topics,
            'transcript': transcript,
            'trigger_confidence': trigger_conf,
            'trigger_class': trigger_cls,
            'audio_trigger_label': audio_trigger_label,
            'audio_trigger_confidence': audio_trigger_conf,
            'people_in_video': faces_summary['people'],
            'unrecognized_face_count': faces_summary['unnamed_count']
        }
    except OSError:
        return None

def _build_event_list(directory, limit=MAX_EVENTS):
    """Baut die Event-Liste (Dateiname, Datum, Größe) für ein gegebenes Verzeichnis."""
    files = sorted(glob.glob(os.path.join(directory, '*.mp4')), key=os.path.getmtime, reverse=True)[:limit]
    return [e for e in (_event_from_file(f) for f in files) if e]

def _build_full_event_list(directory):
    """Wie _build_event_list, aber ohne MAX_EVENTS-Obergrenze — fürs
    Export gedacht, wo wirklich der komplette Bestand gebraucht wird,
    nicht nur die für die Dashboard-Ansicht ohnehin gedeckelten neuesten."""
    files = sorted(glob.glob(os.path.join(directory, '*.mp4')), key=os.path.getmtime, reverse=True)
    return [e for e in (_event_from_file(f) for f in files) if e]

def _camera_name_from_filename(filename):
    """Kameraname aus dem Dateinamen extrahieren: <Kamera>_EVENT_<Zeitstempel>.mp4
    (siehe recorder_pipeline.py, video_file_path). Exakter Split statt
    Prefix-Vergleich, damit z.B. 'Bed' nicht fälschlich 'Bedroom' matcht."""
    return filename.split('_EVENT_')[0] if '_EVENT_' in filename else filename

@app.route('/api/filter_events')
@requires_auth
def api_filter_events():
    """Durchsucht den KOMPLETTEN Bestand (nicht nur die im Dashboard geladenen/
    paginierten Events) nach Kamera/Person/Thema — im Unterschied zu einem rein
    clientseitigen Filter über die schon geladenen Events, der bei Kameras/
    Personen/Themen aus älterer, noch nicht nachgeladener Historie sonst
    unvollständige Ergebnisse liefern würde."""
    camera = request.args.get('camera', '').strip()
    person = request.args.get('person', '').strip()
    topic = request.args.get('topic', '').strip()

    if not camera and not person and not topic:
        return json.dumps({'recent': [], 'archived': []})

    def matches(e):
        if camera and _camera_name_from_filename(e['filename']) != camera:
            return False
        if person and not any(p.get('name') == person for p in (e.get('people_in_video') or [])):
            return False
        if topic and not any(t.get('topic') == topic for t in (e.get('detected_topics') or [])):
            return False
        return True

    recent = [e for e in _build_full_event_list(ALERTS_DIR) if matches(e)]
    archived = [e for e in _build_full_event_list(ARCHIVE_DIR) if matches(e)]
    return json.dumps({'recent': recent, 'archived': archived})

@app.route('/api/export_metadata')
@requires_auth
def api_export_metadata():
    fmt = request.args.get('format', 'csv')
    only_filenames = request.args.get('filenames')
    filter_set = set(only_filenames.split(',')) if only_filenames else None

    recent = _build_full_event_list(ALERTS_DIR)
    for e in recent:
        e['archived'] = False
    archived = _build_full_event_list(ARCHIVE_DIR)
    for e in archived:
        e['archived'] = True
    events = recent + archived
    if filter_set is not None:
        events = [e for e in events if e['filename'] in filter_set]
    # Neueste zuerst, über beide Quellen hinweg einheitlich sortiert
    events.sort(key=lambda e: e['datetime'], reverse=True)

    if fmt == 'json':
        resp = Response(json.dumps(events, indent=2, ensure_ascii=False), mimetype='application/json')
        resp.headers['Content-Disposition'] = 'attachment; filename=idguard_export.json'
        return resp

    output = io.StringIO()
    fieldnames = ['filename', 'archived', 'datetime', 'size', 'trigger_class', 'trigger_confidence',
                  'audio_trigger_label', 'audio_trigger_confidence', 'detected_topics', 'people',
                  'unrecognized_face_count', 'ai_description', 'transcript']
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for e in events:
        row = dict(e)
        row['detected_topics'] = '; '.join(
            f"{t.get('topic')} ({t.get('score')}%)" if t.get('score') is not None else str(t.get('topic'))
            for t in (e.get('detected_topics') or [])
        )
        row['people'] = ', '.join(p.get('name', '') for p in (e.get('people_in_video') or []))
        writer.writerow(row)
    resp = Response(output.getvalue(), mimetype='text/csv')
    resp.headers['Content-Disposition'] = 'attachment; filename=idguard_export.csv'
    return resp

def _get_disk_status():
    try:
        total, used, free = shutil.disk_usage(ALERTS_DIR)
        return {
            'total': round(total / (1024 ** 3), 1),
            'used': round(used / (1024 ** 3), 1),
            'free': round(free / (1024 ** 3), 1),
            'percent': round(used / total * 100, 1) if total else 0
        }
    except Exception:
        return {'total': 0, 'used': 0, 'free': 0, 'percent': 0}

_ollama_check_cache = {'ts': 0, 'status': 'disabled'}
OLLAMA_CHECK_TTL = 20  # Sekunden — kein API-Ping bei jedem 3s-Dashboard-Poll

def _check_ollama_status():
    now = time.time()
    if now - _ollama_check_cache['ts'] < OLLAMA_CHECK_TTL:
        return _ollama_check_cache['status']
    settings = load_settings()
    if not settings.get('AI_ANALYSIS_ENABLED'):
        status = 'disabled'
    else:
        url = settings.get('OLLAMA_URL', 'http://localhost:11434').rstrip('/') + '/api/tags'
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                status = 'ok' if resp.status == 200 else 'error'
        except Exception:
            status = 'error'
    _ollama_check_cache['ts'] = now
    _ollama_check_cache['status'] = status
    return status

def get_detailed_system_status():
    """Ermittelt Modell, VRAM, RAM, CPU, GPU sowie Pipeline-/Event-Status für Dashboard + /api/status"""
    settings = load_settings()
    active_version = settings.get('YOLO_VERSION', YOLO_VERSION)
    active_size = settings.get('MODEL_SIZE', MODEL_SIZE)

    if active_version == "v26":
        active_filename = f"yolo26{active_size}.pt"
    elif active_version == "v12":
        active_filename = f"yolo12{active_size}.pt"
    else:
        active_filename = f"yolov10{active_size}.pt"

    formatted_model_name = f"YOLO {active_version} ({active_size})"

    # Gesamtes System (CPU & RAM via psutil)
    cpu_percent = psutil.cpu_percent(interval=None)
    virtual_mem = psutil.virtual_memory()
    ram_total_gb = round(virtual_mem.total / (1024 ** 3), 1)
    ram_used_gb = round(virtual_mem.used / (1024 ** 3), 1)
    ram_percent = virtual_mem.percent

    # GPU / VRAM über nvidia-smi
    gpu_name = "NVIDIA GeForce RTX 5090"
    vram_used = 0.0
    vram_total = 32.6  # Standardwert in GB
    vram_percent = 0.0
    gpu_temp = 35.0
    encoder_util = 0.0
    decoder_util = 0.0

    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.encoder,utilization.decoder", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd, encoding='utf-8').strip().split('\n')[0]
        parts = [p.strip() for p in output.split(',')]
        if len(parts) >= 6:
            gpu_name = parts[0]
            vram_used = round(float(parts[1]) / 1024.0, 1)   # Umrechnung MB -> GB
            vram_total = round(float(parts[2]) / 1024.0, 1)  # Umrechnung MB -> GB
            vram_percent = round((vram_used / vram_total) * 100, 1) if vram_total > 0 else 0.0
            gpu_temp = float(parts[3])
            encoder_util = float(parts[4])
            decoder_util = float(parts[5])
    except Exception:
        pass

    # CPU-Temperatur (falls unter Linux verfügbar)
    cpu_temp = 42.0
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if 'coretemp' in name.lower() or 'cpu' in name.lower():
                    for entry in entries:
                        if entry.current:
                            cpu_temp = entry.current
                            break
    except Exception:
        pass

    # Worker-Prozesse ermitteln (inklusive des robusten CPU-Zeit-Filters)
    enabled_streams = [s["name"] for s in STREAMS if s.get("enabled", False)]
    worker_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'create_time']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline:
                cmd_str = " ".join(cmdline)
                if 'recorder_pipeline.py' in cmd_str and 'forkserver' in cmd_str:
                    if proc.cpu_times().user + proc.cpu_times().system > 0.5:
                        worker_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    worker_procs.sort(key=lambda p: p.create_time())
    processes_data = []

    for idx, proc in enumerate(worker_procs):
        try:
            if idx < len(enabled_streams):
                stream_name = enabled_streams[idx]
            else:
                stream_name = f"Worker #{idx + 1}"

            cpu = proc.cpu_percent(interval=None)
            mem = round(proc.memory_info().rss / (1024 ** 2), 1)

            processes_data.append({
                'name': stream_name,
                'pid': proc.pid,
                'status': 'LÄUFT',
                'cpu': round(cpu, 1),
                'ram': mem
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    with _restart_lock:
        restarting = pipeline_restart_status["restarting"]

    # Exakte Datenstruktur, die das Frontend (Dashboard JS) erwartet
    return {
        'cpu': {
            'percent': cpu_percent,
            'temp': cpu_temp
        },
        'ram': {
            'percent': ram_percent,
            'used': ram_used_gb,
            'total': ram_total_gb
        },
        'vram': {
            'percent': vram_percent,
            'used': vram_used,
            'total': vram_total
        },
        'gpu': {
            'temp': gpu_temp,
            'status': 'Normal' if gpu_temp < 80 else 'Warning',
            'encoder_util': encoder_util,
            'decoder_util': decoder_util
        },
        'model_version': active_version,
        'model_size': active_size,
        'model_filename': active_filename,
        'yolo_version': active_version,
        'active_model': formatted_model_name,
        'gpu_name': gpu_name,
        'processes': processes_data,
        'active_count': len(processes_data),
        'pipeline_active': is_pipeline_running(),
        'restarting': restarting,
        'recent_events': build_event_list(ALERTS_DIR),
        'archived_events': build_event_list(ARCHIVE_DIR),
        'recording_states': _read_recording_states(),
        'disk': _get_disk_status(),
        'ollama_status': _check_ollama_status(),
    }

def _load_streams_display():
    """Frische Kamera-Liste fürs Rendern — nicht die beim web_ui.py-Start
    fixierte STREAMS-Konstante, damit eine gerade gespeicherte Kamera sofort
    in der GUI auftaucht. Live-Vorschau (helpers.py-Thread) und tatsächliche
    Aufnahme (recorder_pipeline.py) brauchen trotzdem ihren jeweiligen
    Prozess-Neustart, um eine neue Kamera wirklich zu bedienen — das kann
    aus einem laufenden Web-Request heraus nicht sauber selbst ausgelöst
    werden (würde die eigene Antwort mit abwürgen)."""
    try:
        if os.path.exists(STREAMS_F):
            with open(STREAMS_F) as f:
                loaded = json.load(f)
            if isinstance(loaded, list) and loaded:
                streams_copy = [dict(s) for s in loaded]
                overrides = load_overrides()
                for s in streams_copy:
                    if s.get("name") in overrides:
                        s["enabled"] = (overrides[s["name"]] == "ON")
                return streams_copy
    except Exception:
        pass
    return STREAMS

@app.route('/')
@requires_auth
def dashboard():
    overrides = load_overrides()
    settings = load_settings()
    system_status = get_detailed_system_status()

    streams_full = _load_streams_display()
    streams = [s["name"] for s in streams_full]

    # Konfigurierbare Vorschau-Rate (Grid-Thumbnails + Live-View-Lightbox),
    # per Slider in den Settings zwischen 0.5 und 5 fps einstellbar.
    thumbnail_fps = settings.get('THUMBNAIL_FPS', 1) or 1
    thumbnail_interval_ms = int(round(1000 / thumbnail_fps))

    return render_template(
        'dashboard.html',
        streams=streams,
        streams_full=streams_full,
        overrides=overrides,
        settings=settings,
        available_classes=AVAILABLE_CLASSES,
        recent_events=system_status['recent_events'],
        archived_events=system_status['archived_events'],
        pipeline_active=system_status['pipeline_active'],
        pipeline_restarting=system_status['restarting'],
        system_status=system_status,
        csrf_token=CSRF_TOKEN,
        thumbnail_interval_ms=thumbnail_interval_ms
    )

def _trigger_analysis(base_dir, filename):
    settings = load_settings()
    ai_enabled = settings.get('AI_ANALYSIS_ENABLED')
    transcription_enabled = settings.get('TRANSCRIPTION_ENABLED')
    if not ai_enabled and not transcription_enabled:
        return False, "Neither AI video analysis nor transcription is enabled (Settings)."
    basename = os.path.splitext(filename)[0]
    if ai_enabled:
        fs_dir = os.path.join(base_dir, '.thumbs', basename, 'large')
        if (not os.path.isdir(fs_dir) or not glob.glob(os.path.join(fs_dir, '*.jpg'))) and not transcription_enabled:
            return False, "No filmstrip frames available for this video."
        # Fehlende Filmstrip-Frames sind kein harter Fehler, solange Transkription
        # aktiv ist — postprocess.py lässt die Vision-Analyse dann intern einfach
        # leer laufen (ai_analyze.py prüft das selbst) und transkribiert trotzdem.
    try:
        subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, 'postprocess.py'), basename, base_dir])
        _event_cache.clear()
        return True, None
    except Exception as e:
        return False, str(e)

@app.route('/analyze/<filename>', methods=['POST'])
@requires_auth
def analyze_video(filename: str):
    _verify_csrf()
    ok, err = _trigger_analysis(ALERTS_DIR, filename)
    return json.dumps({'ok': ok, 'error': err})

@app.route('/analyze/archive/<filename>', methods=['POST'])
@requires_auth
def analyze_archived_video(filename: str):
    _verify_csrf()
    ok, err = _trigger_analysis(ARCHIVE_DIR, filename)
    return json.dumps({'ok': ok, 'error': err})

def _export_folder_name(filename, topic=None):
    """Event_<Kamera>_<Zeitstempel>[ Topic_<Thema>], Dateisystem-sicher
    (auch für den Fall, dass das Ziel eine Windows-SMB-Freigabe ist)."""
    base = os.path.splitext(filename)[0]
    if '_EVENT_' in base:
        camera, _, timestamp = base.partition('_EVENT_')
    else:
        camera, timestamp = base, ''
    name = f"Event_{camera}_{timestamp}" if timestamp else f"Event_{camera}"
    if topic:
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()
        if safe_topic:
            name += f" Topic_{safe_topic}"
    return "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name)

def _run_export(src_dir, filename, dest_root):
    """Kopiert Video + alle Sidecar-Metadaten + Filmstrip-Ordner eines Events
    in einen eigenen, benannten Unterordner unter dest_root.

    dest_root kann ein lokaler Pfad ODER ein rsync-Remote-Ziel sein
    (user@host:/pfad) — für Remote-Ziele wird bereits eingerichteter
    passwortloser SSH-Zugriff (Public-Key-Auth) vorausgesetzt; das kann diese
    Funktion nicht für euch einrichten, rsync würde sonst nach einem
    Passwort fragen und (da hier kein Terminal angehängt ist) hängen bleiben
    bis der Timeout greift."""
    base = os.path.splitext(filename)[0]
    video_path = os.path.join(src_dir, filename)
    if not os.path.exists(video_path):
        return False, "Video not found."

    topic = None
    ai_path = os.path.join(src_dir, f"{base}.ai.json")
    if os.path.exists(ai_path):
        try:
            with open(ai_path) as f:
                topic = json.load(f).get('top_topic')
        except Exception:
            pass

    folder_name = _export_folder_name(filename, topic)
    is_remote = ('@' in dest_root and ':' in dest_root) or dest_root.startswith('rsync://')

    candidates = [
        video_path,
        os.path.join(src_dir, f"{base}.jpg"),
        os.path.join(src_dir, f"{base}.ai.json"),
        os.path.join(src_dir, f"{base}.trigger.json"),
        os.path.join(src_dir, f"{filename}.xmp"),
    ]
    files_to_copy = [p for p in candidates if os.path.exists(p)]
    thumbs_dir = os.path.join(src_dir, '.thumbs', base)

    if is_remote:
        remote_target = dest_root.rstrip('/') + '/' + folder_name + '/'
        try:
            args = ['rsync', '-a'] + files_to_copy
            if os.path.isdir(thumbs_dir):
                args.append(thumbs_dir)
            args.append(remote_target)
            result = subprocess.run(args, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return False, f"rsync failed: {result.stderr.strip()[:300]}"
            return True, folder_name
        except FileNotFoundError:
            return False, "rsync is not installed on this system."
        except subprocess.TimeoutExpired:
            return False, "rsync timed out (5 min) — is the remote destination reachable?"
        except Exception as e:
            return False, str(e)
    else:
        try:
            dest_dir = os.path.join(dest_root, folder_name)
            os.makedirs(dest_dir, exist_ok=True)
            for p in files_to_copy:
                shutil.copy2(p, dest_dir)
            if os.path.isdir(thumbs_dir):
                shutil.copytree(thumbs_dir, os.path.join(dest_dir, 'thumbs'), dirs_exist_ok=True)
            return True, folder_name
        except Exception as e:
            return False, str(e)

def _export_route_handler(src_dir, filename):
    settings = load_settings()
    export_dir = (settings.get('EXPORT_DIR') or '').strip()
    if not export_dir:
        return json.dumps({'ok': False, 'error': 'No export folder configured (Settings).'})
    ok, result = _run_export(src_dir, filename, export_dir)
    if ok:
        return json.dumps({'ok': True, 'folder': result})
    return json.dumps({'ok': False, 'error': result})

@app.route('/export/<filename>', methods=['POST'])
@requires_auth
def export_video(filename: str):
    _verify_csrf()
    return _export_route_handler(ALERTS_DIR, filename)

@app.route('/export/archive/<filename>', methods=['POST'])
@requires_auth
def export_archived_video(filename: str):
    _verify_csrf()
    return _export_route_handler(ARCHIVE_DIR, filename)

@app.route('/api/events/<kind>')
@requires_auth
def api_events_page(kind):
    """Für 'Ältere laden' im Dashboard — umgeht den MAX_EVENTS-Deckel gezielt,
    ohne den normalen 3s-Poll teurer zu machen."""
    directory = ALERTS_DIR if kind == 'recent' else ARCHIVE_DIR if kind == 'archived' else None
    if directory is None:
        return "", 404
    try:
        offset = max(0, int(request.args.get('offset', 0)))
    except (TypeError, ValueError):
        offset = 0
    page_size = 50
    files = sorted(glob.glob(os.path.join(directory, '*.mp4')), key=os.path.getmtime, reverse=True)
    page_files = files[offset:offset + page_size]
    events = [e for e in (_event_from_file(f) for f in page_files) if e]
    return json.dumps({'events': events, 'has_more': offset + page_size < len(files)})

@app.route('/api/log')
@requires_auth
def api_log():
    log_path = os.path.join(PROJECT_ROOT, 'logs', 'pipeline_runtime.log')
    try:
        n = min(max(int(request.args.get('lines', 50)), 1), 500)
    except (TypeError, ValueError):
        n = 50
    if not os.path.exists(log_path):
        return json.dumps({'lines': [], 'error': 'Log-Datei nicht gefunden: ' + log_path})
    try:
        with open(log_path, 'r', errors='replace') as f:
            lines = list(deque(f, maxlen=n))
        return json.dumps({'lines': lines})
    except Exception as e:
        return json.dumps({'lines': [], 'error': str(e)})

@app.route('/api/search')
@requires_auth
def api_search():
    if search_index is None:
        return json.dumps({'results': [], 'error': 'Search index module not available.'})
    query = request.args.get('q', '').strip()
    if not query:
        return json.dumps({'results': []})
    try:
        matches = search_index.search(query)
    except Exception as e:
        return json.dumps({'results': [], 'error': str(e)})

    results = []
    for filename, base_dir, description, score in matches:
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            continue  # Datei zwischenzeitlich gelöscht, Index noch nicht nachgezogen
        ev = _event_from_file(full_path)
        if ev:
            ev['archived'] = (os.path.abspath(base_dir) == os.path.abspath(ARCHIVE_DIR))
            results.append(ev)
    return json.dumps({'results': results})

@app.route('/api/people_data')
@requires_auth
def api_people_data():
    if faces_db is None:
        return json.dumps({'people': [], 'clusters': {}, 'error': 'Face recognition module not available.'})
    return json.dumps({
        'people': faces_db.list_people(),
        'clusters': faces_db.list_clusters()
    })

@app.route('/api/person_faces/<int:person_id>')
@requires_auth
def api_person_faces(person_id):
    if faces_db is None:
        return json.dumps({'faces': [], 'error': 'Face recognition module not available.'})
    return json.dumps({'faces': faces_db.get_faces_for_person(person_id)})

@app.route('/face_crop/<int:face_id>')
@requires_auth
def face_crop(face_id):
    if faces_db is None:
        abort(404)
    face = faces_db.get_face(face_id)
    if not face:
        abort(404)
    base_dir, crop_path = face['base_dir'], face['crop_path']
    full_path = os.path.abspath(os.path.join(base_dir, crop_path))
    # Sicherheitscheck: der aufgelöste Pfad muss tatsächlich innerhalb ALERTS_DIR
    # oder ARCHIVE_DIR liegen (verhindert Path-Traversal über einen manipulierten
    # base_dir/crop_path-Datensatz)
    if not (full_path.startswith(os.path.abspath(ALERTS_DIR)) or full_path.startswith(os.path.abspath(ARCHIVE_DIR))):
        abort(403)
    if not os.path.exists(full_path):
        abort(404)
    return send_file(full_path)

@app.route('/api/create_person', methods=['POST'])
@requires_auth
def api_create_person():
    _verify_csrf()
    if faces_db is None:
        return json.dumps({'ok': False, 'error': 'Face recognition module not available.'})
    name = request.form.get('name', '').strip()
    face_ids = [int(x) for x in request.form.getlist('face_ids') if x.isdigit()]
    if not name or not face_ids:
        return json.dumps({'ok': False, 'error': 'Name and at least one face are required.'})
    person_id = faces_db.create_person(name, face_ids)
    return json.dumps({'ok': person_id is not None, 'person_id': person_id})

@app.route('/api/assign_to_person', methods=['POST'])
@requires_auth
def api_assign_to_person():
    _verify_csrf()
    if faces_db is None:
        return json.dumps({'ok': False, 'error': 'Face recognition module not available.'})
    try:
        person_id = int(request.form.get('person_id'))
    except (TypeError, ValueError):
        return json.dumps({'ok': False, 'error': 'Invalid person_id.'})
    face_ids = [int(x) for x in request.form.getlist('face_ids') if x.isdigit()]
    faces_db.assign_faces_to_person(person_id, face_ids)
    return json.dumps({'ok': True})

@app.route('/api/unassign_face', methods=['POST'])
@requires_auth
def api_unassign_face():
    _verify_csrf()
    if faces_db is None:
        return json.dumps({'ok': False, 'error': 'Face recognition module not available.'})
    try:
        face_id = int(request.form.get('face_id'))
    except (TypeError, ValueError):
        return json.dumps({'ok': False, 'error': 'Invalid face_id.'})
    faces_db.unassign_face(face_id)
    return json.dumps({'ok': True})

@app.route('/api/unassign_faces_bulk', methods=['POST'])
@requires_auth
def api_unassign_faces_bulk():
    _verify_csrf()
    if faces_db is None:
        return json.dumps({'ok': False, 'error': 'Face recognition module not available.'})
    try:
        face_ids = [int(fid) for fid in request.form.getlist('face_ids')]
    except (TypeError, ValueError):
        return json.dumps({'ok': False, 'error': 'Invalid face_ids.'})
    if not face_ids:
        return json.dumps({'ok': False, 'error': 'No face_ids provided.'})
    faces_db.unassign_faces(face_ids)
    return json.dumps({'ok': True, 'count': len(face_ids)})

@app.route('/api/reject_face', methods=['POST'])
@requires_auth
def api_reject_face():
    _verify_csrf()
    if faces_db is None:
        return json.dumps({'ok': False, 'error': 'Face recognition module not available.'})
    try:
        face_id = int(request.form.get('face_id'))
    except (TypeError, ValueError):
        return json.dumps({'ok': False, 'error': 'Invalid face_id.'})
    faces_db.reject_face(face_id)
    return json.dumps({'ok': True})

@app.route('/api/reject_faces_bulk', methods=['POST'])
@requires_auth
def api_reject_faces_bulk():
    _verify_csrf()
    if faces_db is None:
        return json.dumps({'ok': False, 'error': 'Face recognition module not available.'})
    try:
        face_ids = [int(fid) for fid in request.form.getlist('face_ids')]
    except (TypeError, ValueError):
        return json.dumps({'ok': False, 'error': 'Invalid face_ids.'})
    if not face_ids:
        return json.dumps({'ok': False, 'error': 'No face_ids provided.'})
    faces_db.reject_faces(face_ids)
    return json.dumps({'ok': True, 'count': len(face_ids)})

@app.route('/api/rename_person', methods=['POST'])
@requires_auth
def api_rename_person():
    _verify_csrf()
    if faces_db is None:
        return json.dumps({'ok': False, 'error': 'Face recognition module not available.'})
    try:
        person_id = int(request.form.get('person_id'))
    except (TypeError, ValueError):
        return json.dumps({'ok': False, 'error': 'Invalid person_id.'})
    new_name = request.form.get('name', '').strip()
    if not new_name:
        return json.dumps({'ok': False, 'error': 'Name cannot be empty.'})
    faces_db.rename_person(person_id, new_name)
    return json.dumps({'ok': True})

@app.route('/api/delete_person', methods=['POST'])
@requires_auth
def api_delete_person():
    _verify_csrf()
    if faces_db is None:
        return json.dumps({'ok': False, 'error': 'Face recognition module not available.'})
    try:
        person_id = int(request.form.get('person_id'))
    except (TypeError, ValueError):
        return json.dumps({'ok': False, 'error': 'Invalid person_id.'})
    faces_db.delete_person(person_id)
    return json.dumps({'ok': True})

@app.route('/api/recluster_faces', methods=['POST'])
@requires_auth
def api_recluster_faces():
    _verify_csrf()
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, 'cluster_faces.py')],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return json.dumps({'ok': False, 'error': result.stderr.strip()[:300]})
        return json.dumps({'ok': True, 'output': result.stdout.strip()[-500:]})
    except subprocess.TimeoutExpired:
        return json.dumps({'ok': False, 'error': 'Clustering timed out (60s).'})
    except Exception as e:
        return json.dumps({'ok': False, 'error': str(e)})

@app.route('/api/cleanup_orphaned_faces', methods=['POST'])
@requires_auth
def api_cleanup_orphaned_faces():
    _verify_csrf()
    if faces_db is None:
        return json.dumps({'ok': False, 'error': 'Face recognition module not available.'})
    removed = faces_db.remove_orphaned_faces()
    return json.dumps({'ok': True, 'removed': removed})

@app.route('/health')
def health():
    """Bewusst OHNE @requires_auth: für externe Watchdogs/Monitoring gedacht.
    Liefert nur 'läuft', keine sensiblen Daten."""
    return {"status": "ok"}, 200

@app.route('/api/status')
@requires_auth
def api_status():
    return json.dumps(get_detailed_system_status())

@app.route('/thumbnail/<stream_name>')
@requires_auth
def get_thumbnail(stream_name):
    if stream_name in LATEST_FRAMES:
        return Response(LATEST_FRAMES[stream_name], mimetype='image/jpeg')
    return Response("", status=204)

@app.route('/stream/<stream_name>')
@requires_auth
def video_stream(stream_name):
    """Echter MJPEG-Push-Stream für die Live-View-Lightbox, statt Client-seitigem
    Bild-Polling alle 250ms."""
    def generate():
        boundary = b'--frame'
        while True:
            frame = LATEST_FRAMES.get(stream_name)
            if frame:
                yield (boundary + b'\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)  # ~10 fps Server-Push
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
@requires_auth
def start_pipeline():
    _verify_csrf()
    subprocess.Popen(
        ['/bin/bash', os.path.join(PROJECT_ROOT, 'start_detached.sh')],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return json.dumps({'ok': True})

@app.route('/stop', methods=['POST'])
@requires_auth
def stop_pipeline():
    _verify_csrf()
    subprocess.run(
        ['/bin/bash', os.path.join(PROJECT_ROOT, 'stop.sh')],
        cwd=PROJECT_ROOT
    )
    return json.dumps({'ok': True})

@app.route('/toggle/<name>', methods=['POST'])
@requires_auth
def toggle_stream(name):
    _verify_csrf()
    overrides = load_overrides()
    overrides[name] = 'ON' if overrides.get(name, 'OFF') == 'OFF' else 'OFF'
    save_overrides(overrides)
    return json.dumps({'ok': True, 'state': overrides[name]})

def _clamp(value, lo, hi):
    return max(lo, min(hi, value))

@app.route('/save_settings', methods=['POST'])
@requires_auth
def save_pipeline_settings():
    _verify_csrf()

    # Server-seitige Validierung/Clamping: verhindert unsinnige/negative Werte,
    # auch falls das Formular umgangen oder manipuliert wird.
    try:
        target_fps = int(request.form.get('TARGET_FPS', 30))
    except (TypeError, ValueError):
        target_fps = 30
    try:
        confidence = float(request.form.get('CONFIDENCE_THRESHOLD', 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    try:
        pre_roll = int(request.form.get('PRE_ROLL_SEC', 10))
    except (TypeError, ValueError):
        pre_roll = 10
    try:
        post_roll = int(request.form.get('POST_ROLL_SEC', 30))
    except (TypeError, ValueError):
        post_roll = 30
    try:
        thumbnail_fps = float(request.form.get('THUMBNAIL_FPS', 1))
    except (TypeError, ValueError):
        thumbnail_fps = 1.0
    try:
        retention_days = int(request.form.get('RETENTION_DAYS', 0))
    except (TypeError, ValueError):
        retention_days = 0
    try:
        filmstrip_count = int(request.form.get('FILMSTRIP_COUNT', 0))
    except (TypeError, ValueError):
        filmstrip_count = 0
    try:
        filmstrip_interval = float(request.form.get('FILMSTRIP_INTERVAL_SEC', 2.0))
    except (TypeError, ValueError):
        filmstrip_interval = 2.0
    try:
        ai_max_frames = int(request.form.get('AI_ANALYZE_MAX_FRAMES', 12))
    except (TypeError, ValueError):
        ai_max_frames = 12
    try:
        audio_threshold = float(request.form.get('AUDIO_TRIGGER_THRESHOLD', 0.3))
    except (TypeError, ValueError):
        audio_threshold = 0.3
    try:
        audio_interval = float(request.form.get('AUDIO_TRIGGER_INTERVAL_SEC', 2.0))
    except (TypeError, ValueError):
        audio_interval = 2.0
    audio_categories = [
        line.strip() for line in request.form.get('AUDIO_TRIGGER_CATEGORIES', '').splitlines()
        if line.strip()
    ][:20]  # Sicherheitsdecke — 20 Kategorien sind mehr als genug, jede kostet einen CLAP-Vergleich pro Durchlauf
    try:
        topics_threshold = float(request.form.get('AI_TOPICS_THRESHOLD', 50))
    except (TypeError, ValueError):
        topics_threshold = 50
    try:
        face_min_confidence = float(request.form.get('FACE_MIN_CONFIDENCE', 0.5))
    except (TypeError, ValueError):
        face_min_confidence = 0.5
    ai_topics = [
        line.strip() for line in request.form.get('AI_TOPICS', '').splitlines()
        if line.strip()
    ][:15]  # Sicherheitsdecke — jedes Thema kostet einen Ollama-Vergleich pro Analyse

    old_settings = load_settings()

    settings = {
        "YOLO_VERSION": request.form.get('YOLO_VERSION', 'v26'),
        "MODEL_SIZE": request.form.get('MODEL_SIZE', 'x'),
        "TARGET_FPS": _clamp(target_fps, 1, 60),
        "CONFIDENCE_THRESHOLD": round(_clamp(confidence, 0.05, 1.0), 2),
        "PRE_ROLL_SEC": _clamp(pre_roll, 0, 120),
        "POST_ROLL_SEC": _clamp(post_roll, 0, 300),
        "DETECTION_CLASSES": [int(x) for x in request.form.getlist('DETECTION_CLASSES')] or [0],
        "THUMBNAIL_FPS": round(_clamp(thumbnail_fps, 0.5, 5.0), 1),
        "THEME": request.form.get('THEME', 'dark') if request.form.get('THEME') in ('dark', 'light') else 'dark',
        "RETENTION_DAYS": _clamp(retention_days, 0, 365),
        "FILMSTRIP_COUNT": _clamp(filmstrip_count, 0, 2000),
        "FILMSTRIP_INTERVAL_SEC": round(_clamp(filmstrip_interval, 0.5, 30.0), 1),
        "AI_ANALYSIS_ENABLED": request.form.get('AI_ANALYSIS_ENABLED') == 'on',
        "OLLAMA_URL": request.form.get('OLLAMA_URL', 'http://localhost:11434').strip() or 'http://localhost:11434',
        "OLLAMA_VISION_MODEL": request.form.get('OLLAMA_VISION_MODEL', 'llava:latest').strip() or 'llava:latest',
        "AI_ANALYZE_MAX_FRAMES": _clamp(ai_max_frames, 1, 64),
        "SHOW_DETECTION_BOXES": request.form.get('SHOW_DETECTION_BOXES') == 'on',
        "AUDIO_TRIGGER_ENABLED": request.form.get('AUDIO_TRIGGER_ENABLED') == 'on',
        "AUDIO_TRIGGER_CATEGORIES": audio_categories,
        "AUDIO_TRIGGER_THRESHOLD": round(_clamp(audio_threshold, 0.05, 0.95), 2),
        "AUDIO_TRIGGER_INTERVAL_SEC": round(_clamp(audio_interval, 0.5, 30.0), 1),
        "AI_TOPICS_ENABLED": request.form.get('AI_TOPICS_ENABLED') == 'on',
        "AI_TOPICS": ai_topics,
        "AI_TOPICS_THRESHOLD": round(_clamp(topics_threshold, 0, 100), 0),
        "EXPORT_DIR": request.form.get('EXPORT_DIR', '').strip(),
        "TRANSCRIPTION_ENABLED": request.form.get('TRANSCRIPTION_ENABLED') == 'on',
        "WHISPER_MODEL_SIZE": request.form.get('WHISPER_MODEL_SIZE', 'small') if request.form.get('WHISPER_MODEL_SIZE') in ('tiny', 'base', 'small', 'medium', 'large-v3') else 'small',
        "TRANSCRIPTION_LANGUAGE": request.form.get('TRANSCRIPTION_LANGUAGE', '').strip(),
        "FACE_RECOGNITION_ENABLED": request.form.get('FACE_RECOGNITION_ENABLED') == 'on',
        "FACE_MODEL_PACK": request.form.get('FACE_MODEL_PACK', 'buffalo_s') if request.form.get('FACE_MODEL_PACK') in ('buffalo_s', 'buffalo_m', 'buffalo_l', 'antelopev2') else 'buffalo_s',
        "FACE_MIN_CONFIDENCE": round(_clamp(face_min_confidence, 0.1, 0.95), 2)
    }

    with open(SETTINGS_F, 'w') as f:
        json.dump(settings, f, indent=4)

    # Nur neu starten, wenn sich tatsächlich pipeline-relevante Werte geändert
    # haben — THUMBNAIL_FPS ist eine reine Anzeige-Einstellung fürs Dashboard
    # und würde sonst unnötig einen Neustart auslösen.
    PIPELINE_RELEVANT_KEYS = (
        "YOLO_VERSION", "MODEL_SIZE", "TARGET_FPS", "CONFIDENCE_THRESHOLD",
        "PRE_ROLL_SEC", "POST_ROLL_SEC", "DETECTION_CLASSES"
    )
    pipeline_relevant_changed = any(
        old_settings.get(k) != settings.get(k) for k in PIPELINE_RELEVANT_KEYS
    )

    restarted = False
    if pipeline_relevant_changed and is_pipeline_running():
        threading.Thread(target=_restart_pipeline_background, daemon=True).start()
        restarted = True

    return json.dumps({'ok': True, 'restarted': restarted, 'theme': settings['THEME']})

@app.route('/save_streams', methods=['POST'])
@requires_auth
def save_streams():
    _verify_csrf()
    names = request.form.getlist('stream_name')
    urls = request.form.getlist('stream_url')
    # Ein Hidden-Feld pro Zeile (per Checkbox-onchange auf '0'/'1' gesetzt),
    # NICHT positions-/index-basiert — bleibt so auch nach dynamischem
    # Hinzufügen/Entfernen von Zeilen im JS korrekt korreliert.
    enabled_flags = request.form.getlist('stream_enabled_flag')
    audio_enabled_flags = request.form.getlist('stream_audio_enabled_flag')

    new_streams = []
    seen_names = set()
    error = None
    for name, url, flag, audio_flag in zip(names, urls, enabled_flags, audio_enabled_flags):
        name = name.strip()
        url = url.strip()
        if not name or not url:
            continue  # leere Zeile (z.B. gerade erst per "+ Kamera" hinzugefügt, noch nicht ausgefüllt) überspringen
        if name in seen_names:
            error = f"Doppelter Kamera-Name: '{name}' — Namen müssen eindeutig sein."
            break
        seen_names.add(name)
        new_streams.append({
            "name": name,
            "url": url,
            "enabled": flag == '1',
            "audio_enabled": audio_flag == '1',
            "type": "VIDEO"
        })

    if error:
        return json.dumps({'ok': False, 'error': error})
    if not new_streams:
        return json.dumps({'ok': False, 'error': 'At least one camera with a name and URL is required.'})

    try:
        with open(STREAMS_F, 'w') as f:
            json.dump(new_streams, f, indent=2)
    except Exception as e:
        return json.dumps({'ok': False, 'error': f'Could not save: {e}'})

    # Kamera-Liste ist immer pipeline-relevant (neue/entfernte CameraAgent-Prozesse) —
    # anders als die meisten Settings in save_settings() gibt es hier keinen
    # "live ohne Neustart"-Fall.
    restarted = False
    if is_pipeline_running():
        threading.Thread(target=_restart_pipeline_background, daemon=True).start()
        restarted = True

    overrides = load_overrides()
    display_streams = [
        {'name': s['name'], 'override_on': overrides.get(s['name'], 'ON') == 'ON'}
        for s in new_streams
    ]
    return json.dumps({'ok': True, 'restarted': restarted, 'streams': display_streams})

def _remove_matching_thumbnail(video_path):
    """Löscht Trigger-Screenshot, Filmstrip-Ordner und AI-Metadaten/-Sidecar zu einem Video."""
    base = os.path.splitext(video_path)[0]
    for extra in (base + '.jpg', base + '.ai.json', base + '.trigger.json', video_path + '.xmp'):
        if os.path.exists(extra):
            try:
                os.remove(extra)
            except Exception as e:
                print(f"Fehler beim Löschen von {extra}: {e}")
    fs_dir = os.path.join(os.path.dirname(video_path), '.thumbs', os.path.basename(base))
    if os.path.isdir(fs_dir):
        try:
            shutil.rmtree(fs_dir)
        except Exception as e:
            print(f"Fehler beim Löschen des Filmstrips: {e}")

@app.route('/delete/<filename>', methods=['POST'])
@requires_auth
def delete_video(filename: str):
    _verify_csrf()
    file_path = os.path.abspath(os.path.join(ALERTS_DIR, filename))
    if file_path.startswith(os.path.abspath(ALERTS_DIR)) and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            _event_cache.clear()
            return json.dumps({'ok': False, 'error': str(e)})
        _remove_matching_thumbnail(file_path)
        if search_index is not None:
            search_index.remove_event(filename)
        if faces_db is not None:
            faces_db.remove_faces_for_video(filename)
    _event_cache.clear()
    return json.dumps({'ok': True})

@app.route('/archive/<filename>', methods=['POST'])
@requires_auth
def archive_video(filename: str):
    _verify_csrf()
    src_path = os.path.abspath(os.path.join(ALERTS_DIR, filename))
    # Sicherheitscheck: Datei muss direkt (nicht rekursiv) im ALERTS_DIR liegen
    if not (src_path.startswith(os.path.abspath(ALERTS_DIR)) and os.path.isfile(src_path)
            and os.path.dirname(src_path) == os.path.abspath(ALERTS_DIR)):
        return json.dumps({'ok': False, 'error': 'Video not found.'})
    try:
        shutil.move(src_path, os.path.join(ARCHIVE_DIR, os.path.basename(src_path)))
    except Exception as e:
        _event_cache.clear()
        return json.dumps({'ok': False, 'error': str(e)})
    # Passenden Trigger-Screenshot mitnehmen, damit er im Archiv weiterhin angezeigt wird
    thumb_src = os.path.splitext(src_path)[0] + '.jpg'
    if os.path.exists(thumb_src):
        try:
            shutil.move(thumb_src, os.path.join(ARCHIVE_DIR, os.path.basename(thumb_src)))
        except Exception as e:
            print(f"Fehler beim Archivieren des Thumbnails: {e}")
    # AI-Metadaten (JSON fürs Dashboard + XMP-Sidecar für Immich) mitnehmen
    for extra in (os.path.splitext(src_path)[0] + '.ai.json', os.path.splitext(src_path)[0] + '.trigger.json', src_path + '.xmp'):
        if os.path.exists(extra):
            try:
                shutil.move(extra, os.path.join(ARCHIVE_DIR, os.path.basename(extra)))
            except Exception as e:
                print(f"Fehler beim Archivieren von {extra}: {e}")
    # Filmstrip-Ordner mitnehmen
    fs_src = os.path.join(ALERTS_DIR, '.thumbs', os.path.splitext(os.path.basename(src_path))[0])
    if os.path.isdir(fs_src):
        try:
            shutil.move(fs_src, os.path.join(ARCHIVE_DIR, '.thumbs', os.path.basename(fs_src)))
        except Exception as e:
            print(f"Fehler beim Archivieren des Filmstrips: {e}")
    if search_index is not None:
        search_index.update_location(filename, ARCHIVE_DIR)
    if faces_db is not None:
        faces_db.update_base_dir(filename, ARCHIVE_DIR)
    _event_cache.clear()
    return json.dumps({'ok': True})

@app.route('/delete_archived/<filename>', methods=['POST'])
@requires_auth
def delete_archived_video(filename: str):
    _verify_csrf()
    file_path = os.path.abspath(os.path.join(ARCHIVE_DIR, filename))
    if file_path.startswith(os.path.abspath(ARCHIVE_DIR)) and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            _event_cache.clear()
            return json.dumps({'ok': False, 'error': str(e)})
        _remove_matching_thumbnail(file_path)
        if search_index is not None:
            search_index.remove_event(filename)
        if faces_db is not None:
            faces_db.remove_faces_for_video(filename)
    _event_cache.clear()
    return json.dumps({'ok': True})

@app.route('/thumb/<filename>')
@requires_auth
def serve_thumbnail_image(filename):
    file_path = os.path.abspath(os.path.join(ALERTS_DIR, filename))
    if not file_path.startswith(os.path.abspath(ALERTS_DIR)) or not os.path.exists(file_path):
        return "", 404
    return Response(open(file_path, 'rb').read(), mimetype='image/jpeg')

@app.route('/thumb/archive/<filename>')
@requires_auth
def serve_archived_thumbnail_image(filename):
    file_path = os.path.abspath(os.path.join(ARCHIVE_DIR, filename))
    if not file_path.startswith(os.path.abspath(ARCHIVE_DIR)) or not os.path.exists(file_path):
        return "", 404
    return Response(open(file_path, 'rb').read(), mimetype='image/jpeg')

def _serve_filmstrip(base_dir, basename, index):
    file_path = os.path.abspath(os.path.join(base_dir, '.thumbs', basename, 'small', f'{index:04d}.jpg'))
    if not file_path.startswith(os.path.abspath(os.path.join(base_dir, '.thumbs'))) or not os.path.exists(file_path):
        return "", 404
    return Response(open(file_path, 'rb').read(), mimetype='image/jpeg')

@app.route('/filmstrip/<basename>/<int:index>')
@requires_auth
def serve_filmstrip(basename, index):
    return _serve_filmstrip(ALERTS_DIR, basename, index)

@app.route('/filmstrip/archive/<basename>/<int:index>')
@requires_auth
def serve_archived_filmstrip(basename, index):
    return _serve_filmstrip(ARCHIVE_DIR, basename, index)



def _transcode_stream(input_file):
    """Transcodiert eine Datei on-the-fly per ffmpeg und streamt sie als MP4-Response."""
    def generate():
        process = subprocess.Popen(
            ['ffmpeg', '-i', input_file, '-c:v', 'libx264', '-preset', 'ultrafast',
             '-tune', 'zerolatency', '-crf', '28', '-c:a', 'mp3', '-f', 'mp4',
             '-movflags', 'frag_keyframe+empty_moov', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=-1
        )
        try:
            while True:
                chunk = process.stdout.read(8192)
                if not chunk:
                    break
                yield chunk
        except Exception:
            process.kill()
        finally:
            process.stdout.close()

    return Response(generate(), mimetype='video/mp4')

@app.route('/video/<filename>')
@requires_auth
def serve_annot_video(filename):
    input_file = os.path.join(ALERTS_DIR, filename)
    if not os.path.exists(input_file):
        return f"File not found: {filename}", 404
    return _transcode_stream(input_file)

@app.route('/video/archive/<filename>')
@requires_auth
def serve_archived_video(filename):
    input_file = os.path.join(ARCHIVE_DIR, filename)
    if not os.path.exists(input_file):
        return f"File not found: {filename}", 404
    return _transcode_stream(input_file)

if __name__ == '__main__':
    # threaded=True: sonst blockiert ein laufender Pipeline-Neustart (stop.sh +
    # start_detached.sh) die komplette Web-UI inkl. /api/status-Polling, da der
    # Flask-Dev-Server standardmäßig single-threaded ist.
    #
    # Hinweis für Dauerbetrieb: der eingebaute Dev-Server ist nicht für
    # Produktivbetrieb gedacht. Für fenrir empfiehlt sich gunicorn/waitress
    # dahinter plus ein Reverse Proxy (Caddy/nginx) mit TLS davor, siehe Chat.
    app.run(host='0.0.0.0', port=19473, threaded=True)
