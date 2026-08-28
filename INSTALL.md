# Installation Guide: IDguard PRO

This document describes the process of installing `IDguard PRO` on a new Linux system (optimized for NVIDIA GPU setups).

## 1. Prerequisites

Before proceeding with the Python installation, your system must provide the necessary hardware foundations:

*   **NVIDIA Driver:** Ensure a modern NVIDIA driver is installed. For current-generation GPUs (RTX 40xx/50xx, e.g. Blackwell/RTX 5090) you need a driver new enough to support **CUDA 12.8+** — check the "CUDA Version" shown at the top right of `nvidia-smi`.
    *   Verify with: `nvidia-smi`
*   **CUDA Toolkit:** The CUDA toolkit should be available and compatible with your driver (recommended: `12.8` or newer).
*   **ffmpeg:** Required as a system binary — not just the Python bindings — for on-the-fly video transcoding when you play back a recording in the dashboard (the recording pipeline itself uses PyAV directly and does not need this, but the web UI's playback route does).
*   **System Packages:** You need Python 3, the venv module, and ffmpeg.
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip python3-venv git ffmpeg -y
    ```

## 2. Clone the Repository

Clone the repository onto your target machine:
```bash
git clone https://github.com/axeljerabek/IDguard_PRO
cd IDguard_PRO
```

## 3. Setup Virtual Environment (Python venv)

To keep your system clean, we use an isolated virtual environment (`.venv`). This prevents conflicts with other Python packages on your machine.

1.  **Create the venv:**
    ```bash 
    python3 -m venv .venv
    ```
2.  **Activate the environment:**
    ```bash
    source .venv/bin/activate
    ```
    *(After activation, you should see `(.venv)` prepended to your terminal prompt.)*

## 4. Install Dependencies

This installs the AI models (Ultralytics/YOLO) and the web components. Thanks to `requirements.txt`, most of this is automated — **but PyTorch needs one extra, explicit step**, since a plain `pip install torch` from PyPI does not reliably give you a build with full CUDA support for current-generation GPUs.

```bash
# First, upgrade pip itself
pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support EXPLICITLY (needed for Blackwell/RTX 50-series;
# also works fine on older cards down to Turing/RTX 20-series — one wheel covers both).
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then install the rest of the project dependencies
pip install -r requirements.txt
```

The YOLO model file (`.pt`) is downloaded automatically on first start, based on the `YOLO_VERSION`/`MODEL_SIZE` selected in `config.py` (or later, live, in the dashboard Settings).

### Optional: Ollama (AI scene descriptions)

If you want the optional "describe what happened in this recording" feature, you'll additionally need a running [Ollama](https://ollama.com) instance (commonly run in its own Docker container) with a vision-capable model pulled, e.g.:
```bash
docker exec -it <container-name> ollama pull llava:latest
```
This is entirely optional — IDguard PRO records and detects normally with no Ollama installed at all. Everything related to this (enable/disable, endpoint URL, which model) is configured later, live, in the dashboard under Settings → KI-Videoanalyse.

## 5. Configuration (Crucial!)

Before starting, you must configure the paths and settings in the environment or via `config.py`:

*   Verify that camera paths (`RTSP` or local `/dev/videoX`) are correct.
*   Ensure the project directory permissions allow writing to `alerts/` and `logs/`.
*   Copy the config.py.example file to config.py and edit it (stream locations etc):
    ```bash
    cp config.py.example config.py
    ```

`config.py` mainly needs your camera list (`STREAMS`) and, if you want non-default starting values, `YOLO_VERSION`/`MODEL_SIZE`. Almost everything else (FPS, thresholds, detection classes, thumbnails, retention, theme, AI analysis, ...) can be changed afterwards, live, from the dashboard's Settings page without touching this file again — see `manual.md` for the full reference of what lives where.

## 6. Starting the System

IDguard PRO is two separate processes: the **recording pipeline** (`recorder_pipeline.py`, one worker per camera) and the **web dashboard** (`web_ui.py`). The dashboard's Start/Stop button controls the pipeline process; the dashboard itself needs to be started separately.

**Start the web dashboard:**
```bash
python3 web_ui.py
```
The dashboard will be accessible at `http://0.0.0.0:19473`. From there, use the Start button in the pipeline control bar to launch the recording pipeline — or start it directly:
```bash
./start_detached.sh
```

**Running in Background (Optional):**
Use the provided shell scripts (`start_detached.sh` / `stop.sh` for the pipeline, your own wrapper such as `start_web_ui.sh` for the dashboard) to manage both processes cleanly and run them in the background. For unattended/production setups, consider wrapping both in systemd services, and optionally pairing `watchdog.sh` with a cron job to auto-restart the dashboard if it ever becomes unresponsive (see `manual.md`).

---

*⚠️ **Note:** If you are using a very new GPU (e.g., RTX 5090), double-check that PyTorch was actually installed with CUDA support (step 4 above) rather than a CPU-only build. To verify, run:*
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```
