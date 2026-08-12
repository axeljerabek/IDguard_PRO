#!/bin/bash

# 1. Pfade definieren (absolut sicher)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_ROOT/.venv/bin/python3"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/pipeline_runtime.log"

# 2. Vorbereitung (Ordner erstellen)
mkdir -p "$LOG_DIR"

echo "🚀 Starting IDguard PRO in BACKGROUND..."
echo "📍 Project Root: $PROJECT_ROOT"

# 3. Den Prozess mit nohup im Hintergrund starten
# 'nohup' sorgt dafür, dass er auch weiterläuft, wenn du das Terminal schließt.
# '> $LOG_FILE 2>&1' leitet alle Ausgaben (Logs & Fehler) in die Datei um.
# '&' schickt ihn in den Hintergrund.

cd "$PROJECT_ROOT"
nohup "$VENV" manager.py start > "$LOG_FILE" 2>&1 &

# 4. PID und Info ausgeben
PID=$!

echo "✅ Pipeline running with PID: $PID"
echo "📄 You can watch logs live with:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🛡️  To stop it, run:"
echo "   kill $PID"
