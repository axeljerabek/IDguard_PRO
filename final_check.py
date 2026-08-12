import subprocess
import time
import os

pipeline = "/workspace/IDguard_PRO_COMPLETE/recorder_pipeline.py"
venv_python = "/workspace/IDguard_PRO/.venv/bin/python"
logs_dir = "/workspace/IDguard_PRO_COMPLETE/logs"
alerts_dir = "/int_path_fix:/workspace/IDguard_PRO_COMPLETE/alerts"

print("🚀 STARTING FINAL SINGLE-SHOT TEST...")

if not os.path.exists(pipeline):
    print(f"❌ Error: {pipeline} missing")
    exit(1)

# Start the pipeline
process = subprocess.Popen([venv_python, pipeline], 
                           cwd="/workspace/IDguard_PRO_COMPLETE",
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.STDOUT, 
                           text=True)

print("⏳ Running for 20 seconds to ensure stability...")
start = time.time()
try:
    while time.time() - start < 20:
        line = process.stdout.readline()
        if line:
            print(f"  [PIPELINE] {line.strip()}")
        else:
            if process.poll() is not None:
                break
            time.sleep(0.5)
except Exception as e:
    print(f"❌ Error during run: {e}")
finally:
    process.terminate()
    print("\n✅ Test window closed.")

# Final Audit
print("\n🔍 FINAL HEALTH AUDIT:")
if os.path.exists(logs_dir):
    files = [f for f in os.listdir(logs_dir) if os.path.getmtime(os.path.join(logs_dir, f)) > (time.time() - 600)]
    print(f"✅ LOGS: Found {len(files)} recent logs.")
else:
    print("❌ LOGS: Missing!")

# Note: Checking alerts folder manually via simple path to avoid shell expansion errors in this script
alerts_path = "/workspace/IDguard_PRO_COMPLETE/alerts"
if os.path.exists(alerts_path):
    vids = [f for f in os.listdir(alerts_path) if f.endswith('.mp4') and os.path.getmtime(os.path.join(alerts_path, f)) > (time.time() - 600)]
    if vids:
        print(f"✅ VIDEOS: SUCCESS! Found {len(vids)} new videos!")
    else:
        print("⚠️  VIDEOS: No NEW mp4 files detected.")
else:
    print("❌ ALERTS DIR MISSING!")

