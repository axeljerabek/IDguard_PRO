#!/bin/bash
# start_detached.sh - Startet die IDguard-Pipeline losgelöst vom aufrufenden Prozess.

# Robust gegen abweichendes Arbeitsverzeichnis (z.B. falls das Skript mal
# nicht mit cwd=PROJECT_ROOT aufgerufen wird): immer relativ zum eigenen
# Skript-Ordner arbeiten, statt uns auf den Aufrufer zu verlassen.
cd "$(dirname "$0")" || exit 1

LOG_FILE="./logs/pipeline_runtime.log"
PYTHON_EXE="./.venv/bin/python"
mkdir -p ./logs

# Verhindert doppelte Instanzen (z.B. Doppelklick auf Start, bevor die UI den
# Status aktualisiert hat) — zwei parallele Pipelines würden sich bei
# denselben Kamera-Streams und Ausgabedateien in die Quere kommen.
if pgrep -f "recorder_pipeline.py" > /dev/null; then
    echo "⚠️  [$(date)] Pipeline läuft bereits — Start übersprungen." | tee -a "$LOG_FILE"
    exit 0
fi

echo "🚀 [$(date)] Launching IDguard Pipeline..." | tee -a "$LOG_FILE"

VENV_SITES=$(find .venv -name "site-packages" -type d | head -n 1)
if [ -n "$VENV_SITES" ]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$VENV_SITES"
    echo "🔍 Python Library Path Injected: $VENV_SITES" | tee -a "$LOG_FILE"
fi

if [ ! -f "$PYTHON_EXE" ]; then
    echo "❌ ERROR: VENV executable not found!" | tee -a "$LOG_FILE"
    exit 1
fi

nohup "$PYTHON_EXE" recorder_pipeline.py >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > ./pipeline.pid
echo "✅ Pipeline started with PID: $PID" | tee -a "$LOG_FILE"
