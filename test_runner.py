import subprocess
import time
import os

print("🚀 Starting Pipeline Test...")
process = subprocess.Popen(['python3', '/workspace/IDguard_PRO_COMPLETE/recorder_pipeline.py'], 
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

start_time = time.time()
timeout = 30 # Run for 30 seconds to see if it processes frames
output_log = []

try:
    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if line:
            print(f"  [PIPELINE] {line.strip()}")
            output_log.append(line.strip())
        else:
            time.sleep(0.1)
except Exception as e:
    print(f"Error during test: {e}")
finally:
    process.terminate()
    print("\n🏁 Test duration reached.")

# Step 3: Verification - Check logs and DB
print("\n--- VERIFICATION PHASE ---")
log_files = os.listdir('/workspace/IDguard_PRO_COMPLETE/logs')
if log_files:
    latest_log = sorted(log_files, key=lambda x: os.path.getmtime(os.path.join('/workspace/IDguard_PRO_COMPLETE/logs', x)))[-1]
    print(f"Latest Log File: {latest_log}")
    with open(f'/workspace/IDguard_PRO_COMPLETE/logs/{latest_log}', 'r') as f:
        last_lines = f.readlines()[-5:]
        print("Last 5 lines of log:")
        for l in last_lines: print(f"  {l.strip()}")

import sqlite3
db_path = '/workspace/IDguard_PRO_COMPLETE/database/alerts.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM alerts")
    count = cursor.fetchone()[0]
    print(f"\nTotal Alerts in DB: {count}")
    if count > 0:
        cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 1")
        print(f"Latest Alert Entry: {cursor.fetchone()}")
    conn.close()
else:
    print("\n❌ Database not found!")

