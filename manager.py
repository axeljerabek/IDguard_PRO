import os
import subprocess
import sys
import time

# Use the project root from our construction zone
PROJECT_ROOT = "/opt/data/IDguard_PRO_FINAL"
PIPELINE_SCRIPT = os.path.join(PROJECT_ROOT, "recorder_pipeline.py")
PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv/bin/python3")

# Process tracking file to allow 'stop' and 'status' functionality
PID_FILE = os.path.join(PROJECT_ROOT, "pipeline.pid")

def get_running_pid():
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return None
    return None

def is_running(pid):
    if pid is None:
        return False
    try:
        # Signal 0 checks if the process exists and we have permission to send signals
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def start_pipeline():
    pid = get_running_pid()
    if is_running(pid):
        print(f"⚠️  Pipeline is already running (PID: {pid}).")
        return

    print("🚀 Starting Identity-Guard PRO Pipeline...")
    # We use the venv python to ensure all dependencies are loaded correctly
    # Running in background
    process = subprocess.Popen(
        [PYTHON_EXE, PIPELINE_SCRIPT],
        stdout=None, # Let it print to terminal/logs handled by pipeline internal logging
        stderr=None,
        preexec_fn=os.setsid # Create a new session to allow killing the process group later
    )
    
    with open(PID_FILE, 'w') as f:
        f.write(str(process.pid))
    
    print(f"✅ Pipeline started successfully (PID: {process.pid}).")

def stop_pipeline():
    pid = get_running_pid()
    if not is_running(pid):
        print("⚠️  No running pipeline found to stop.")
        return

    print(f"🛑 Stopping Pipeline (PID: {pid})...")
    try:
        # Kill the entire process group
        pgid = os.getpgid(pid)
        os.killpg(pgid, 15) # SIGTERM for graceful shutdown
        
        # Wait for it to die
        for _ in range(10):
            if not is_running(pid):
                break
            time.sleep(1)
        
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
            
        print("✅ Pipeline stopped safely.")
    except Exception as e:
        print(f"❌ Error stopping pipeline: {e}")

def status_pipeline():
    pid = get_running_pid()
    if is_running(pid):
        print(f"🟢 PIPELINE: RUNNING (PID: {pid})")
    else:
        print("🔴 PIPELINE: STOPPED")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manager.py [start|stop|status]")
        sys.exit(1)

    action = sys.argv[1].lower()

    if action == "start":
        # Ensure venv exists or handle error
        if not os.path.exists(PYTHON_EXE):
            print(f"❌ Error: Python environment NOT found at {PYTHON_EXE}. Please run setup first.")
            sys.exit(1)
        start_pipeline()
    elif action == "stop":
        stop_pipeline()
    elif action == "status":
        status_pipeline()
    else:
        print(f"❌ Unknown command: {action}")
        sys.exit(1)
