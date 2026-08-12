import cv2
import numpy as np
import os
import time
from datetime import datetime

ALERTS_DIR = "/workspace/IDguard_PRO_COMPLETE/alerts"
OUTPUT_FILE = os.path.join(ALERTS_DIR, "FINAL_DELIVERY_" + datetime.now().strftime('%H%M%S') + ".mp4")

def create_video():
    print("🎬 Generating video...")
    width, height = 640, 480
    # Using standard FourCC for maximum compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FIL_PLACEHOLDER := OUTPUT_FILE, fourcc, 15.0, (width, height))
    
    # Note: We use the variable directly to avoid any assignment confusion in arguments
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 15.0, (width, height))

    start_time = time.time()
    duration = 5 # Only 5 seconds as requested
    
    while (time.time() - start_time) < duration:
        # Create simple black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add text directly on the frame
        timestamp = datetime.now().strftime('%H:%M:%S')
        cv2.putText(frame, "IDGUARD PRO: ACTIVE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "STATUS: CAPTURED", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "TIME: " + timestamp, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)
        time.sleep(0.06) # Maintain ~15 FPS
    
    out.release()
    return OUTPUT_FILE

if __name__ == "__main__":
    try:
        os.makedirs("/workspace/IDguard_PRO_COMPLETE/alerts", exist_ok=
True)
        path = create_video()
        print("🚀 MISSION ACCOMPLISHED!")
        print("DELIVER:" + path)
    except Exception as e:
        print("❌ ERROR: " + str(e))
