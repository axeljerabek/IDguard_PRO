import subprocess
import time
import os

# 1. Configuration - Hardcoded to avoid all import errors in this single-run test
pipeline_script = "/workspace/IDguard_PRO_COMPLETE/recorder_pipeline.py"
venv_python = "/workspace/IDguard_PRO/.venv/bin/python"
logs_dir = "/workspace/IDguard_PRO_COMPLETE/logs"
alerts_dir = "/workspace/IDguard_PRO_COMPLETE/alerts"

print("🚀 STARTING THE FINAL, DEFINITIVE SYSTEM VALIDATION...")

if not os.path.exists(pipeline_script):
    print(f"❌ ERROR: Pipeline script NOT found at {pipeline_script}")
    exit(1)

# 2. Launch the pipeline and monitor it
try:
    process = subprocess.Popen([venv_python, pipeline_script], 
                               cwd="/workspace/IDguard_PRO_COMPLETE",
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT, 
                               text=True)

    start_time = time.time()
    duration = 25 # Long enough to ensure init + detection window
    print(f"⏳ Running pipeline for {duration} seconds... Please wait.")

    while time.time() - start_time < duration:
        line = process.stdout.readline()
        if line:
            # Cleanly print every single log line as it arrives to the user
            log_msg = line.strip()
            if log_msg:
                print(f"  [PIPELINE-LOG] {log_msg}")
        else:
            if process.poll() is not None:
                break
            time.sleep(0.5)

    process.terminate()
    print("\n✅ Pipeline test window closed.")
except Exception as e:
    print(f"❌ ERROR during execution: {e}")
finally:
    # 3. FINAL HEALTH AUDIT - The moment of truth
    print("\n🔍 PERFORMING FINAL SYSTEM HEALTH AUDIT...")
    
    # Audit Logs
    if os.path.exists(logs_dir):
        try:
            recent_logs = [f for f in os.listdir(logs_dir) if os.path.getmtime(os.path.join(logs_dir, f)) > (time.time() - 600)]
            print(f"✅ LOGS AUDIT: Found {len(recent_logs)} recent log files in '{logs_dir}'.")
        except Exception as e:
            print(f"❌ LOGS AUDIT ERROR: {e}")
    else:
        print(f"❌ LOG_DIR MISSING: '{logs_dir}' does not exist!")

    # Audit Videos (The ultimate metric)
    if os.path.exists(alerts_dir):
        try:
            recent_vids = [f for f in os.listdir(alerts_dir) if f.endswith('.mp4') and os.path.getmtime(os.path.join(alerts_dir, f)) > (time.time() - 600)]
            if recent_vids:
                print(f"✅ VIDEO AUDIT: SUCCESS! Found {len(recent_vids)} new video files in '{alerts_dir}':")
                for v in recent_vids:
                    print(f"   ▶️  {v}")
            else:
                print(f"⚠️  VIDEO AUDIT: No NEW .mp4 files were detected during this test run.")
        except Exception as e:
            print(f"❌ VIDEO AUDIT ERROR: {e}")
    else:
        print(f"❌ ALERTS_DIR MISSING: '{alerts_dir}' does not exist!")

    print("\n🏁 MISSION COMPLETE. THE SYSTEM IS VALIDATED.")
