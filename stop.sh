#!/bin/bash
# stop.sh - Stops the pipeline by killing all processes related to IDguard streams.

echo "Stopping IDguard Pipeline..."

# Kill all python processes that are part of the recorder_pipeline or manager
pkill -f "recorder_pipeline.py"
pkill -f "manager.py"

# Check if any process is still running and kill it
sleep 2
if pgrep -f "recorder_annot_pipeline.py" > /dev/null; then
    echo "Cleaning up leftover recorder processes..."
    pkill -f "recorder_annot_pipeline.py"
fi

echo "Pipeline Stopped."
