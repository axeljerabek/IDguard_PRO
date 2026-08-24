from flask import Flask, render_template, request, redirect, url_for, Response
import os
import sys
import glob
import subprocess
import json
import psutil
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

def get_detailed_system_status():
    """Ermittelt Modell, VRAM, RAM, CPU und GPU passend für das Dashboard-JS"""
    active_version = YOLO_VERSION
    active_size = MODEL_SIZE
    active_filename = MODEL_FILENAME

    if os.path.exists(SETTINGS_F):
        try:
            with open(SETTINGS_F, 'r') as f:
                sett = json.load(f)
                active_version = sett.get('YOLO_VERSION', active_version)
                active_size = sett.get('MODEL_SIZE', active_size)
                if active_version == "v26":
                    active_filename = f"yolo26{active_size}.pt"
                elif active_version == "v12":
                    active_filename = f"yolo12{active_size}.pt"
                else:
                    active_filename = f"yolov10{active_size}.pt"
        except Exception:
            pass

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

    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd, encoding='utf-8').strip().split('\n')[0]
        parts = [p.strip() for p in output.split(',')]
        if len(parts) >= 4:
            gpu_name = parts[0]
            vram_used = round(float(parts[1]) / 1024.0, 1)   # Umrechnung MB -> GB
            vram_total = round(float(parts[2]) / 1024.0, 1)  # Umrechnung MB -> GB
            vram_percent = round((vram_used / vram_total) * 100, 1) if vram_total > 0 else 0.0
            gpu_temp = float(parts[3])
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

    # Worker-Prozesse ermitteln
    enabled_streams = [s["name"] for s in STREAMS if s.get("enabled", False)]
    worker_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'create_time']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline:
                cmd_str = " ".join(cmdline)
                if 'recorder_pipeline.py' in cmd_str and 'forkserver' in cmd_str:
                    worker_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    worker_procs.sort(key=lambda p: p.create_time())
    processes_data = []

    for idx, proc in enumerate(worker_procs):
        try:
            stream_name = enabled_streams[idx] if idx < len(enabled_streams) else f"Worker #{idx + 1}"
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
            'status': 'Normal' if gpu_temp < 80 else 'Warning'
        },
        'model_version': active_version,
        'model_size': active_size,
        'model_filename': active_filename,
        'gpu_name': gpu_name,
        'processes': processes_data,
        'active_count': len(processes_data)
    }

@app.route('/')
@requires_auth
def dashboard():
    overrides = load_overrides()
    settings = load_settings()
    pipeline_active = is_pipeline_running()
    system_status = get_detailed_system_status()

    alerts = sorted(glob.glob(os.path.join(ALERTS_DIR, '*.mp4')), key=os.path.getmtime, reverse=True)
    recent_events = []
    for alert in alerts:
        try:
            mtime = os.path.getmtime(alert)
            size = os.path.getsize(alert)
            dt_str = datetime.fromtimestamp(mtime).strftime('%d.%m.%Y %H:%M')
            recent_events.append({
                'filename': os.path.basename(alert),
                'datetime': dt_str,
                'size': format_size(size)
            })
        except OSError:
            pass

    streams = [s["name"] for s in STREAMS]

    return render_template(
        'dashboard.html',
        streams=streams,
        overrides=overrides,
        settings=settings,
        available_classes=AVAILABLE_CLASSES,
        recent_events=recent_events,
        pipeline_active=pipeline_active,
        system_status=system_status
    )

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

@app.route('/start', methods=['POST'])
@requires_auth
def start_pipeline():
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
    subprocess.run(
        ['/bin/bash', os.path.join(PROJECT_ROOT, 'stop.sh')],
        cwd=PROJECT_ROOT
    )
    return redirect(url_for('dashboard'))

@app.route('/toggle/<name>', methods=['POST'])
@requires_auth
def toggle_stream(name):
    overrides = load_overrides()
    overrides[name] = 'ON' if overrides.get(name, 'OFF') == 'OFF' else 'OFF'
    save_overrides(overrides)
    return redirect(url_for('dashboard'))

@app.route('/save_settings', methods=['POST'])
@requires_auth
def save_pipeline_settings():
    settings = {
        "YOLO_VERSION": request.form.get('YOLO_VERSION', 'v26'),
        "MODEL_SIZE": request.form.get('MODEL_SIZE', 'x'),
        "TARGET_FPS": int(request.form.get('TARGET_FPS', 30)),
        "CONFIDENCE_THRESHOLD": float(request.form.get('CONFIDENCE_THRESHOLD', 0.5)),
        "PRE_ROLL_SEC": int(request.form.get('PRE_ROLL_SEC', 10)),
        "POST_ROLL_SEC": int(request.form.get('POST_ROLL_SEC', 30)),
        "DETECTION_CLASSES": [int(x) for x in request.form.getlist('DETECTION_CLASSES')] or [0]
    }

    with open(SETTINGS_F, 'w') as f:
        json.dump(settings, f, indent=4)

    return redirect(url_for('dashboard'))

@app.route('/delete/<filename>', methods=['POST'])
@requires_auth
def delete_video(filename: str):
    file_path = os.path.abspath(os.path.join(ALERTS_DIR, filename))
    if file_path.startswith(os.path.abspath(ALERTS_DIR)) and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Fehler beim Löschen: {e}")
    return redirect(url_for('dashboard'))

@app.route('/video/<filename>')
@requires_auth
def serve_annot_video(filename):
    input_file = os.path.join(ALERTS_DIR, filename)
    if not os.path.exists(input_file):
        return f"File not found: {filename}", 404

    def generate():
        process = subprocess.Popen(
            ['ffmpeg', '-i', input_file, '-c:v', 'libx264', '-preset', 'ultrafast',
             '-tune', 'zerolatency', '-crf', '28', '-c:a', 'mp3', '-f', 'mp4',
             '-movflags', 'frag_keyframe+empty_moov', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=-1
        )
        try:
            while true:
                chunk = process.stdout.read(8192)
                if not chunk:
                    break
                yield chunk
        except Exception:
            process.kill()
        finally:
            process.stdout.close()

    return Response(generate(), mimetype='video/mp4')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=19473)
