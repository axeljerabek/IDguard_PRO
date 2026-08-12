# IDguard PRO

IDguard PRO is a high-performance, local AI surveillance system designed for real-time person detection and event-driven video recording. It leverages the advanced YOLOv10 architecture to provide intelligent monitoring with minimal latency and zero reliance on cloud services.

## Overview

The core mission of IDguard PRO is to act as an intelligent edge-computing sentinel. Unlike traditional motion-detection systems that trigger on any pixel change, IDguard PRO uses deep learning to identify specific objects (primarily humans) and only initiates recording when a high-confidence detection occurs. This significantly reduces storage requirements and eliminates false positives caused by wind, shadows, or animals.

## Key Features

*   **Advanced AI Detection:** Powered by YOLOv10 for state-of-the-art object detection accuracy and speed.
*   **Event-Driven Recording:** Automatic MP4 recording triggered by person detection, including configurable pre-roll and post-roll buffers to capture the arrival and departure of subjects.
*   **Web Dashboard:** A lightweight Flask-based web interface for real-time stream monitoring, system configuration, and viewing recent event alerts.
*   **Multi-Threaded Architecture:** Each camera stream operates in its own independent thread, ensuring high stability and preventing a single connection failure from affecting the entire system.
*   **Edge Computing Focus:** Designed to run entirely on local hardware, ensuring maximum privacy and data sovereignty.
*   **Flexible Configuration:** Easily manage stream overrides, detection thresholds, and recording parameters via JSON configuration files.

## Hardware Requirements

To achieve optimal performance with high-resolution streams and real-time inference, the following hardware is recommended:

*   **GPU:** NVIDIA GPU compatible with CUDA 12/13 (e.g., RTX 30-series, 40-series, or 50-series).
*   **OS:** Linux-based distribution (Ubuntu recommended).
*   **Memory:** Minimum 8GB RAM (higher recommended for multiple simultaneous streams).
*   **Storage:** Sufficient space for high-bitrate MP4 event recordings.

## Tech Stack

*   **Language:** Python 3.x
*   **Inference Engine:** PyTorch with CUDA support
*   **Computer Vision:** Ultralytics YOLOv10, OpenCV
*   **Web Framework:** Flask
*   **Process Management:** Threading and Subprocess modules

## Installation

Detailed installation steps, including virtual environment setup and dependency management, are provided in the accompanying [INSTALL.md](./INSTALL.md) file.

## Project Structure

*   `web_ui.py`: The main entry point for the Flask web dashboard.
*   `recorder_pipeline.py`: The core detection and recording logic.
*   `manager.py`: Handles stream orchestration and thread management.
*   `config.py`: System-wide configuration and parameter settings.
*   `alerts/`: Directory containing recorded event MP4 files (automatically generated).
*   `logs/`: Application and system logs for debugging and auditing.

  ## Tested Hardware & Configurations

The system has been thoroughly tested and runs rock-solid across both compact edge/SFF builds and high-end workstations:

| System | Specs | Buffer Settings | Status |
| :--- | :--- | :--- | :--- |
| **Intel NUC 11 Enthusiast** | 32 GB RAM \| NVIDIA RTX 2060 (6 GB VRAM) | Pre-roll: 5s / Post-roll: 10s | Stable |
| **High-End Workstation** | Intel Core Ultra 9 285K \| 64 GB RAM \| NVIDIA RTX 5090 (32 GB VRAM) | Pre-roll: 10s / Post-roll: 30s | Stable |

---

 ## Acknowledgements & Citation

This project utilizes [YOLOv10](https://github.com/THU-MIG/yolov10) and is powered by the [Ultralytics](https://github.com/ultralytics/ultralytics) framework for real-time object detection.

If you use this repository, please consider citing the original YOLOv10 paper:

```bibtex
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}

## Disclaimer

This software is intended for educational and private security purposes. Users are responsible for ensuring that their use of surveillance technology complies with all local, regional, and international laws regarding privacy and data protection.
