#!/bin/bash

echo "🛑 Searching for Web-UI process on port 19473..."

# lsof -t gibt nur die PIDs zurück. Wir speichern sie in einem Array/String.
PIDs=$(lsof -t -i:19473)

if [ -z "$PIDs" ]; then
    echo "❌ No Web-UI server is running on port 19473."
else
    # Schleife für den Fall, dass mehrere Prozesse den Port belegen
    for PID in $PIDs; do
        kill "$PID" 2>/dev/null
        echo "✅ Process $PID stopped successfully."
    done
fi
