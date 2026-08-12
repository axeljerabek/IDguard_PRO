# Installation Guide: IDguard PRO

This document describes the process of installing `IDguard PRO` on a new Linux system (optimized for NVIDIA GPU setups).

## 1. Prerequisites

Before proceeding with the Python installation, your system must provide the necessary hardware foundations:

*   **NVIDIA Driver:** Ensure a modern NVIDIA driver is installed (preferably version `535` or newer for modern architectures like RTX 40xx/50xx).
    *   Verify with: `nvidia-smi`
*   **CUDA Toolkit:** The CUDA toolkit should be available and compatible with your driver (recommended: `12.x` or `13.x`).
*   **System Packages:** You need Python 3 and the venv module.
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip python3-venv git -y
    ```

## 2. Clone the Repository

Clone the repository onto your target machine:
```bash
git clone https://github.com/axeljerabek/IDguard_PRO
cd IDguard_PRO_COMPLETE
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

This installs the AI models (Ultralytics/YOLO) and the web components. Thanks to `requirements.txt`, this is automated:

```bash
# First, upgrade pip itself
pip install --upgrade pip

# Install all project dependencies
pip install -r requirements.txt
```

Download the yolov10 model:

wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt

## 5. Configuration (Crucial!)

Before starting, you must configure the paths and settings in the environment or via `config.py`:
*   Verify that camera paths (`RTSP` or local `/dev/videoX`) are correct.
*   Ensure the project directory permissions allow writing to `alerts/` and `logs/`.

## 6. Starting the System

**Start Web Interface:**
```bash
python3 web_ui.py
```
The dashboard will be accessible at `http://0.0.0.0:19473`.

**Running in Background (Optional):**
Use the provided shell scripts (e.g., `start_web_ui.sh`) to manage services cleanly and run them in the background.

---

*⚠️ **Note:** If you are using a very new GPU (e.g., RTX 5090), ensure that PyTorch was installed with the corresponding CUDA support. To verify, run:*
`python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"`
