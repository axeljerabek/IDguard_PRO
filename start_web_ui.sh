#!/bin/bash
# ---------------------------------------------------------
# IDGUARD PRO - Web UI Launcher
# Purpose: Starts the Flask Dashboard in the background
# ---------------------------------------------------------

PROJECT_ROOT=$(pwd)
VENV_PATH="$PROJECT_ROOT/.venv"
WEB_UI_SCRIPT="$PROJECT_ROOT/web_ui.py"

echo "🚀 [WebUI] Initializing Dashboard Launcher..."

# 1. Check if Virtual Environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# 2. Activate VENV
source "$VENV_PATH/bin/activate"

# 3. Ensure Flask is installed (Non-interactive)
if ! python -c "import flask" &> /dev/null; then
    echo "📦 [WebUI] Flask not detected. Installing requirements..."
    pip install --quiet flask
fi

# 4. Start the Server in the background
echo "📡 [WebUI] Launching Web Dashboard on port 19473..."
nohup python "$WEB_UI_SCRIPT" > "$PROJECT_ROOT/logs/web_ui.log" 2>&1 &

# 5. Final Confirmation
sleep 2
if ps aux | grep -v grep | grep "python.*web_ui.py" > /dev/null; then
    echo "✅ [WebUI] DASHBOARD IS LIVE!"
    echo "👉 Access it at: http://localhost:19473"
    echo "📄 Logs available at: $PROJECT_ROOT/logs/web_ui.log"
else
    echo "❌ [WebUI] FAILED to start. Check logs in $PROJECT_ROOT/logs/web_ui.log"
    exit 1
fi
