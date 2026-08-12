#!/bin/bash
LOG_FILE="./logs/pipeline_runtime.log"
PYTHON_EXE="./.venv/bin/python"
mkdir -p ./logs
echo "🚀 [$(date)] Launching IDguard Pipeline..." | tee -a "$LOG_FILE"
VENV_SITES=$(find .venv -name "site-packages" -type d | head -n 1)
if [ -n "$VENV_SITES" ]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$VENV_SITES"
    echo "🔍 Python Library Path Injected: $VENV_SITES" | tee -a "$LOG_FILE"
fi
if [ ! -f "$PYTHON_EXE" ]; then echo "❌ ERROR: VENV executable not found!" | tee -a "$LOG_FILE"; exit 1; fi
nohup "$PYTHON_EXE" recorder_pipeline.py >> "$LOG_FILE" 2>&1 &
PID=$!
echo "✅ Pipeline started with PID: $PID" | tee -a "$LOG_FILE"
