import subprocess
import time
import os
from datetime import datetime, timedelta

# Dynamically use the discovered VENV Python
VENV_PYTHON="/workspace/IDguard_PRO/.venv/bin/python"
PIPELINE_SCRIPT = "/workspace/IDguard_PRO_COMPLETE/recorder_pipeline.py"
ALERTS_DIR = "/workspace/IDguard_PRO_COMPLETE/alerts"

def run_capture():
    if not VENV_PYTHON or not os.path.exists(VENV_PYTHON):
        print("ERROR: Could not find a working VENV Python!")
        return

    print(f"🚀 Starting Pipeline Capture using {VENV_PYTHON}...")
    # We use the current directory as workdir to ensure config files are found
    process = subprocess.Popen([VENV_PYTHON, PIPELINE_SCRIPT], 
                               cwd="/workspace/IDguard_PRO_COMPLETE",
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT, 
                               text=True)

    # Let it run long enough to capture a few seconds of video
    time.sleep(15)

    print("🛑 Stopping Pipeline...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    
    print("✅ Capture routine finished.")

    # 3. Find files created in the last 2 minutes (to avoid old test artifacts)
    now = datetime.now()
    two_mins_ago = now - timedelta(minutes=2)
    found_files = []

    if os.path.exists(ALERTS_DIR):
        for f in os.listdir(ALERTS_DIR):
            f_path = os.path.join(ALERTS_DIR, f)
            # We check for files modified very recently
            mtime = os.path.getmtime(f_path)
            if mtime > two_mins_ago.timestamp():
                if f.endswith('.mp4'):
                    found_files.append(f_path)

    # 4. Output findings for the Assistant to deliver
    if not found_files:
        print("❌ No NEW video files (.mp4) were detected in the alerts directory.")
    else:
        print(f"✨ SUCCESS! Found {len(found_files)} new videos:")
        for path in sorted(found_files):
            print(f"DELIVER:{path}")

if __name__ == "__main__":
    run_capture()
