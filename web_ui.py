from flask import Flask, render_template, request, redirect, url_for, Response, abort
import os
import sys
import glob
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

from config import STREAMS, ALERTS_DIR, PROJECT_ROOT, SETTINGS_F, YOLO_VERSION, MODEL_SIZE, MODEL_FILENAME
from auth import requires_auth
from helpers import (
    LATEST_FRAMES, start_thumbnail_thread, is_pipeline_running,
    load_overrides, load_settings, save_overrides, format_size
)

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
        fs_dir = os.path.join(os.path.dirname(f), '.thumbs', os.path.splitext(os.path.basename(f))[0], 'small')
        fs_count = len(glob.glob(os.path.join(fs_dir, '*.jpg'))) if os.path.isdir(fs_dir) else 0
        ai_desc = None
        ai_path = os.path.splitext(f)[0] + '.ai.json'
        if os.path.exists(ai_path):
            try:
                with open(ai_path) as af:
                    ai_desc = json.load(af).get('description')
            except Exception:
                pass
        ai_pending = os.path.exists(os.path.splitext(f)[0] + '.ai.pending')
        trigger_conf, trigger_cls = None, None
        trigger_path = os.path.splitext(f)[0] + '.trigger.json'
        if os.path.exists(trigger_path):
            try:
                with open(trigger_path) as tf:
                    tmeta = json.load(tf)
                trigger_conf = tmeta.get('confidence')
                trigger_cls = tmeta.get('class')
            except Exception:
                pass
        return {
            'filename': os.path.basename(f),
            'datetime': datetime.fromtimestamp(mtime).strftime('%d.%m.%Y %H:%M'),
            'size': format_size(size),
            'has_thumbnail': os.path.exists(thumb_path),
            'filmstrip_count': fs_count,
            'ai_description': ai_desc,
            'ai_pending': ai_pending,
            'trigger_confidence': trigger_conf,
            'trigger_class': trigger_cls
        }
    except OSError:
        return None

def _build_event_list(directory, limit=MAX_EVENTS):
    """Baut die Event-Liste (Dateiname, Datum, Größe) für ein gegebenes Verzeichnis."""
    files = sorted(glob.glob(os.path.join(directory, '*.mp4')), key=os.path.getmtime, reverse=True)[:limit]
    return [e for e in (_event_from_file(f) for f in files) if e]

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

@app.route('/')
@requires_auth
def dashboard():
    overrides = load_overrides()
    settings = load_settings()
    system_status = get_detailed_system_status()

    streams = [s["name"] for s in STREAMS]

    # Konfigurierbare Vorschau-Rate (Grid-Thumbnails + Live-View-Lightbox),
    # per Slider in den Settings zwischen 0.5 und 5 fps einstellbar.
    thumbnail_fps = settings.get('THUMBNAIL_FPS', 1) or 1
    thumbnail_interval_ms = int(round(1000 / thumbnail_fps))

    return render_template(
        'dashboard.html',
        streams=streams,
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
    if not settings.get('AI_ANALYSIS_ENABLED'):
        return False, "KI-Videoanalyse ist nicht aktiviert (Settings)."
    basename = os.path.splitext(filename)[0]
    fs_dir = os.path.join(base_dir, '.thumbs', basename, 'large')
    if not os.path.isdir(fs_dir) or not glob.glob(os.path.join(fs_dir, '*.jpg')):
        return False, "Keine Filmstrip-Frames für dieses Video vorhanden."
    try:
        subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, 'ai_analyze.py'), basename, base_dir])
        _event_cache.clear()
        return True, None
    except Exception as e:
        return False, str(e)

@app.route('/analyze/<filename>', methods=['POST'])
@requires_auth
def analyze_video(filename: str):
    _verify_csrf()
    ok, err = _trigger_analysis(ALERTS_DIR, filename)
    return redirect(url_for('dashboard') if ok else url_for('dashboard', analyze_error=err))

@app.route('/analyze/archive/<filename>', methods=['POST'])
@requires_auth
def analyze_archived_video(filename: str):
    _verify_csrf()
    ok, err = _trigger_analysis(ARCHIVE_DIR, filename)
    return redirect(url_for('dashboard') if ok else url_for('dashboard', analyze_error=err))

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
    return redirect(url_for('dashboard'))

@app.route('/stop', methods=['POST'])
@requires_auth
def stop_pipeline():
    _verify_csrf()
    subprocess.run(
        ['/bin/bash', os.path.join(PROJECT_ROOT, 'stop.sh')],
        cwd=PROJECT_ROOT
    )
    return redirect(url_for('dashboard'))

@app.route('/toggle/<name>', methods=['POST'])
@requires_auth
def toggle_stream(name):
    _verify_csrf()
    overrides = load_overrides()
    overrides[name] = 'ON' if overrides.get(name, 'OFF') == 'OFF' else 'OFF'
    save_overrides(overrides)
    return redirect(url_for('dashboard'))

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
        "AUDIO_TRIGGER_INTERVAL_SEC": round(_clamp(audio_interval, 0.5, 30.0), 1)
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

    return redirect(url_for('dashboard', saved=1, restarted=int(restarted)))

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
            print(f"Fehler beim Löschen: {e}")
        _remove_matching_thumbnail(file_path)
    _event_cache.clear()
    return redirect(url_for('dashboard'))

@app.route('/archive/<filename>', methods=['POST'])
@requires_auth
def archive_video(filename: str):
    _verify_csrf()
    src_path = os.path.abspath(os.path.join(ALERTS_DIR, filename))
    # Sicherheitscheck: Datei muss direkt (nicht rekursiv) im ALERTS_DIR liegen
    if src_path.startswith(os.path.abspath(ALERTS_DIR)) and os.path.isfile(src_path) \
            and os.path.dirname(src_path) == os.path.abspath(ALERTS_DIR):
        try:
            shutil.move(src_path, os.path.join(ARCHIVE_DIR, os.path.basename(src_path)))
        except Exception as e:
            print(f"Fehler beim Archivieren: {e}")
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
    _event_cache.clear()
    return redirect(url_for('dashboard'))

@app.route('/delete_archived/<filename>', methods=['POST'])
@requires_auth
def delete_archived_video(filename: str):
    _verify_csrf()
    file_path = os.path.abspath(os.path.join(ARCHIVE_DIR, filename))
    if file_path.startswith(os.path.abspath(ARCHIVE_DIR)) and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Fehler beim Löschen: {e}")
        _remove_matching_thumbnail(file_path)
    _event_cache.clear()
    return redirect(url_for('dashboard'))

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
