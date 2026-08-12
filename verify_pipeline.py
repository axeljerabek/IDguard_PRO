import os

pipeline_path = "/workspace///IDguard_PRO_COMPLETE/recorder_pipeline.py" # wait, double slash error in my head... 
# Let us use the absolute path properly.
pipeline_path = "/workspace/IDguard_PRO_COMPLETE/recorder_pipeline.py"

if not os.path.exists(pipeline_path):
    print("Error: File NOT found at", pipeline_path)
    exit(1)

with open(pipeline_path, "r") as f:
    content = f.read()

print("🔍 Checking recorder_pipeline.py integrity...")

# 1. Check for the critical "frame = None" to prevent UnboundLocalError
if "frame = None" in content:
    print("✅ [PASS] Variable safety (frame=None) is present.")
else:
    print("❌ [FAIL] 'frame = None' initialization is MISSING!")

# 2. Check for adaptive switching logic
if "self.use_synthetic = False" in content and "cv2.VideoCapture(self.url)" in content:
    print("✅ [PASS] Adaptive mode-switching (Self-Healing) is present.")
else:
    print("❌ [FAIL] Adaptive/Self-healing logic is MISSING!")

# 3. Check for the retry mechanism in LIVE_STREAM branch
if "self.cap = cv2.VideoCapture(self.url)" in content and "continue" in content:
    print("✅ [PASS] Live stream retry loop (break/continue) is present.")
else:
    print("❌ [FAIL] Stream recovery logic is MISSING!")

# 4. Check for the trailing 'if __name__ == \"__main__\":' to ensure it remains executable
if "if __name__ == \"__main__\":" in content:
    print("✅ [PASS] Main entry point structure is intact.")
else:
    print("❌ [FAIL] Python main block was corrupted!")

