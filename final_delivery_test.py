import subprocess
import time
import os
from datetime import datetime, timedelta

VENV_PYTHON="/workspace/IDguard_PRO/.venv/bin/python"
PIPELINE_SCRIPT="/workspace/ID_PATH_REPLACEMENT_PLACEHOLDER/recorder_pipeline.py" # We will fix this path below
ALERTS_DIR="/workspace/IDguard_PRO_COMPLETE/alerts"

def run_final_test():
    # Correct the pipeline path to use the absolute working directory script
    actual_pipeline = "/workspace/IDguard_PRO_COMPLETE/recorder_pipeline.py"
    
    if not os.path.exists(actual_pipeline):
        print("❌ ERROR: Pipeline script NOT found!")
        return

    print(f"🚀 Starting FINAL, long-duration capture test (45s)...")
    process = subprocess.Popen([VENV_PYTHON, actual_pipeline], 
                               cwd="/workspace/IDguard_PRO_COMPLETE",
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT, 
                               text=True)

    # We run for a longer time to ensure detection and writing happens
    start_time = time.time()
    timeout = 45 
    found_video = None

    print("⏳ Monitoring streams for detections/recordings...")
    
    try:
        while time.time() - start_time < timeout:
            line = process.stdout.readline()
            if line:
                stripped_line = line.strip()
                print(f"  [PIPELINE] {stripped_line}")
                # Look for the magic string that indicates a file was created or logged
                if "/alerts/" in stripped_line and ".mp4" in stripped_line:
                    # Extract path from log if possible, or just look at filesystem
                    pass 
            else:
                time.sleep(0.5)
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
    finally:
        process.terminate()
        print("\n🛑 Capture period finished.")

    # Step 3: The real verification - scrape the filesystem for RECENT mp4s
    print("\n🔍 Scanning /alerts/ folder for NEW videos...")
    now = datetime.now()
    cutoff = now - timedelta(minutes=5)
    new_videos = []

    if os.path.exists(ALERTS_DIR):
        for f in os.listdir(ALERTS_DIR):
            f_path = os.path.join(ALERTS_DIR, f)
            if f.endswith('.mp4') and os.path.getmtime(f_path) > cutoff.timestamp():
                new_videos.append(f_path)

    if not new_videos:
        print("❌ ERROR: No NEW .mp4 files found in alerts folder after 45s!")
    else:
        print(f"✨ SUCCESS! Found {len(new_videos)} new videos:")
        for path in sorted(new_videos):
            print(f"DELIVER:{path}")

if __name__ == '__main__':
    run_final_test()
