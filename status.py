import os
import sys
import psutil
import json
import subprocess

# Pfad-Auflösung
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)

try:
    from config import STREAMS, SETTINGS_F, YOLO_VERSION, MODEL_SIZE, MODEL_FILENAME
except ImportError:
    STREAMS = []
    SETTINGS_F = "pipeline_settings.json"
    YOLO_VERSION = "v26"
    MODEL_SIZE = "x"
    MODEL_FILENAME = "yolo26x.pt"

def get_system_status():
    print("=" * 72)
    print("  IDENTITY-GUARD PRO - SYSTEM STATUS MONITOR (RTX 5090 EDITION)")
    print("=" * 72)

    # 1. Aktive Einstellungen auslesen
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

    print(f"🤖 Aktives KI-Modell : YOLO {active_version} (Größe: {active_size})")
    print(f"📂 Modell-Datei      : {active_filename}")
    print("-" * 72)

    # 2. GPU / VRAM Status über nvidia-smi
    gpu_success = False
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd, encoding='utf-8').strip().split('\n')[0]
        parts = [p.strip() for p in output.split(',')]
        if len(parts) >= 3:
            gpu_name = parts[0]
            vram_used = float(parts[1])
            vram_total = float(parts[2])
            print(f"🎮 GPU (NVIDIA)      : {gpu_name}")
            print(f"💾 VRAM Belegt       : {vram_used:.1f} MB / {vram_total:.1f} MB ({(vram_used/vram_total)*100:.1f}%)")
            gpu_success = True
    except Exception:
        pass

    if not gpu_success:
        print("🎮 GPU Status        : ⚠️ Konnte nvidia-smi nicht abfragen")

    print("-" * 72)

    # 3. Prozesse & Ressourcen filtern (Nur aktive Worker mit CPU/RAM-Last ungleich Leerlauf)
    active_pipelines = 0
    total_cpu_percent = 0.0
    total_rss_mem = 0.0

    print(f"{'Stream-Name':<15} | {'PID':<6} | {'Status':<10} | {'CPU (%)':<8} | {'RAM (MB)':<10}")
    print("-" * 72)

    enabled_streams = [s["name"] for s in STREAMS if s.get("enabled", False)]
    
    worker_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'create_time']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline:
                cmd_str = " ".join(cmdline)
                if 'recorder_pipeline.py' in cmd_str and 'forkserver' in cmd_str:
                    # Filter den inaktiven Leerlauf-Worker heraus (der 0 CPU-Zeit hat)
                    # Wir prüfen hier die CPU-Zeit direkt über psutil
                    if proc.cpu_times().user + proc.cpu_times().system > 0.5:
                        worker_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Nach Startzeit sortieren
    worker_procs.sort(key=lambda p: p.create_time())

    for idx, proc in enumerate(worker_procs):
        try:
            if idx < len(enabled_streams):
                stream_name = enabled_streams[idx]
            else:
                stream_name = f"Worker #{idx + 1}"
            
            status_label = "LÄUFT"

            cpu = proc.cpu_percent(interval=0.1)
            mem = proc.memory_info().rss / (1024 ** 2)
            
            total_cpu_percent += cpu
            total_rss_mem += mem
            active_pipelines += 1

            print(f"{stream_name:<15} | {proc.pid:<6} | {status_label:<10} | {cpu:<8.1f} | {mem:<10.1f}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if active_pipelines == 0:
        print("ℹ️ Keine aktiven Kamera-Pipeline-Prozesse gefunden.")
    else:
        print("-" * 72)
        print(f"📊 Gesamt aktiv      : {active_pipelines} Pipeline(s)")
        print(f"🔥 Gesamte CPU-Last  : {total_cpu_percent:.1f}%")
        print(f"🧠 Gesamter RAM-Verbrauch: {total_rss_mem:.1f} MB")

    print("=" * 72)

if __name__ == "__main__":
    get_system_status()
