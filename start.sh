#!/bin/bash
# IDguard PRO - Portable Startup Script

# Determine the directory where THIS script resides. 
# This makes the script work from ANY folder or path.
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
VENV_PATH="$SCRIPT_DIR/.venv"

echo "🚀 Starting IDguard PRO Pipeline..."
echo "📍 Project Root: $SCRIPT_DIR"

# Check if the virtual environment exists relative to this script's location
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated from: $VENV_PATH"
else
    echo "❌ Error: Virtual environment NOT found at $VENV_PATH"
    echo "💡 Please ensure the '.venv' folder is present in your project directory."
    exit 1
fi

# Move into the correct working directory before running anything.
cd "$SCRIPT_DIR" || exit 1

# Run the pipeline via manager.py (which we previously fixed to use relative paths)
if [ -f "manager.py" ]; then
    echo "Running: python3 manager.py start"
    python3 manager.py start
else
    echo "❌ Error: manager.py not found in $SCRIPT_DIR"
    exit 1
fi
