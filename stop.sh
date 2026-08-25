#!/bin/bash
# stop.sh - Stoppt die IDguard-Pipeline sauber.
#
# recorder_pipeline.py fängt SIGTERM jetzt selbst ab (siehe dortige
# Fixes) und schließt laufende Aufnahmen sauber (Flush + close), bevor der
# Prozess beendet wird. Das braucht kurz Zeit — deshalb hier aktiv pollen
# statt eines festen "sleep 2", und erst nach einer Grace Period hart
# nachhelfen (SIGKILL), falls ein Prozess doch mal hängt.

PATTERN="recorder_pipeline.py"
GRACE_SECONDS=15

echo "Stopping IDguard Pipeline..."

if ! pgrep -f "$PATTERN" > /dev/null; then
    echo "Pipeline was not running."
    exit 0
fi

# SIGTERM an Master + alle Worker-Prozesse (matchen alle auf denselben
# Skriptnamen in der Prozess-Cmdline)
pkill -TERM -f "$PATTERN"

waited=0
while pgrep -f "$PATTERN" > /dev/null; do
    if [ "$waited" -ge "$GRACE_SECONDS" ]; then
        echo "⚠️  Pipeline reagiert nach ${GRACE_SECONDS}s nicht auf SIGTERM — erzwinge SIGKILL."
        pkill -KILL -f "$PATTERN"
        sleep 1
        break
    fi
    sleep 1
    waited=$((waited + 1))
done

if pgrep -f "$PATTERN" > /dev/null; then
    echo "❌ Konnte nicht alle Pipeline-Prozesse beenden. Bitte manuell prüfen: pgrep -fa \"$PATTERN\""
    exit 1
fi

echo "✅ Pipeline Stopped (nach ${waited}s)."
