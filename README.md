# IDguard PRO

IDguard PRO watches your cameras, decides what's actually worth recording, saves the event, and — if you want — tells you afterwards in plain words what happened in it. It does that by chaining three things together instead of expecting one AI to handle it all: a YOLO model (v10, v12, or v26 — your choice) watches every frame live and spots the moment something worth recording is happening; the recording itself then saves the event with a short buffer before and after; and, optionally, a vision-language model (via Ollama) looks at the finished clip and writes a short description of what it saw. Each model only does the one job it's good at, and the handoff between them is automatic. Everything runs on your own hardware, with GPU acceleration end to end — decoding, detection, and encoding — no cloud involved.

Some of this code was written with AI (Google Gemini, Claude, Claude Code, and local models). But the architecture — deciding how the pieces should fit together, and the work of actually making it all run reliably — was still a person's job.

## Overview

The core mission of IDguard PRO is to act as an intelligent edge-computing sentinel. Unlike traditional motion-detection systems that trigger on any pixel change, IDguard PRO uses deep learning to identify specific objects — people, vehicles, animals, packages, or any other COCO class — and only initiates recording when a high-confidence detection occurs. This significantly reduces storage requirements and eliminates false positives caused by wind, shadows, or animals (unless you want those too — detection classes are fully configurable).

## Key Features

### Detection & Recording
* **Switchable AI Backends:** YOLOv10, YOLOv12, and YOLO26, selectable per deployment (see model comparison below), with model size from Nano to Extra Large.
* **Event-Driven Recording:** Automatic MP4 recording triggered by detection, with configurable pre-roll and post-roll buffers to capture the arrival and departure of subjects.
* **Live Detection-Box Overlays:** Optionally see exactly what the model sees, in real time — bounding boxes (color-coded per class) drawn directly into the camera grid previews and the live-view lightbox, reusing the inference the pipeline already runs (no extra GPU cost). Toggle in Settings → Anzeige.
* **Trigger Screenshots:** Every recording gets an automatic screenshot with the detection box drawn in, taken at the trigger frame, plus a small confidence/class badge (e.g. "person 87%") shown right on the thumbnail in the dashboard.
* **Configurable Filmstrip Thumbnails:** A configurable number of small + large preview frames captured after pre-roll for each event. The small frames (with detection boxes, for humans) power a hover-to-scrub filmstrip preview right in the dashboard; the large frames (kept raw/unannotated, for cleaner AI input) are sized for feeding into a vision AI.
* **Resilient by Design:** Cameras that crash or disconnect auto-restart with the same config. If the AI model fails to load (e.g. a transient GPU/driver hiccup), the pipeline keeps retrying in the background instead of silently going blind.
* **GPU-Aware Startup:** Detects the installed GPU (Turing through Blackwell) and automatically determines whether FP16 and cuDNN are safe to use, with a staged self-test and fallback — same codebase runs unmodified from an RTX 2060 up to an RTX 5090.
* **Full Hardware Pipeline:** NVDEC-accelerated decoding and NVENC-accelerated encoding where available, with automatic, self-healing fallback to software decode/encode per-stream if hardware acceleration isn't supported or a camera drops out.
* **Accurate Timing:** Frame timestamps are wall-clock based rather than a fixed frame counter, so a brief network stall shows up as a natural pause in the recording instead of the rest of the clip playing back too fast.
* **Race-Safe Model Download:** The YOLO checkpoint auto-downloads on first run via an atomic, per-process temp-file + rename, with a timeout and a minimum-size sanity check — safe even though multiple processes (master, every camera worker, the dashboard) import the same config module concurrently.

### Optional AI Scene Description (Ollama)
* After a recording finishes, IDguard PRO can optionally hand the large filmstrip frames to a locally hosted **Ollama** vision model and have it describe what happened in the clip in plain language.
* **Model picker with tested presets** in Settings, plus a free-text "Eigenes Modell…" option for anything else pulled into Ollama. Note: Ollama has no native video-file input — every model, regardless of which one you pick, receives the same image-sequence (the filmstrip frames), never the raw video.
* A **live Ollama-connectivity badge** in Settings shows at a glance whether the configured endpoint is reachable, so a misconfigured URL or a down container is visible without needing to open a terminal.
* The result is written both as a small JSON file (shown directly in the dashboard, next to Recent Recordings and Archive) and as an XMP sidecar file for compatibility with photo/video managers like **Immich**.
* A manual **re-analyze button** per recording (and a visible "Analysiere…" state while it's running) — useful for re-running with a different model, or for older recordings from before AI analysis was enabled.
* Fully optional and off by default — no Ollama instance required unless you turn it on.

### Web Dashboard
* **CCTV-style layout:** a slim pipeline control bar up top, live camera previews and Recent Recordings front and center, with Settings, Hardware/System Status, Log, and Archive tucked into collapsible sections out of the way.
* **Live previews:** per-camera thumbnails and a full live view, with a configurable refresh rate (0.5–5 fps slider) — disabled or unreachable cameras simply show nothing instead of flickering broken-image icons.
* **REC indicators everywhere:** a live badge on any camera thumbnail currently recording, plus the browser tab title itself switches to "🔴 REC · IDguard PRO" while any camera is active — visible even from a background tab.
* **"Zuletzt aktiv" per camera:** each camera in the live-preview grid shows a relative timestamp of its most recent recorded event.
* **Card-style recording thumbnails:** large, full-width preview images in Recent Recordings and Archive — enough to actually recognize what happened at a glance, not a tiny icon.
* **In-dashboard log viewer:** the last 100 lines of the pipeline log, collapsible, auto-refreshing only while open.
* **Archive workflow:** archive recordings you want to keep permanently, separate from the auto-cleanup pool; deleting or archiving carries thumbnails, filmstrips, confidence metadata, and AI metadata along automatically.
* **Auto-retention:** optionally delete un-archived recordings older than N days; archived recordings are never touched by auto-cleanup.
* **Day-grouped recording lists** with clear separators, and a "load older" control once a list passes 200 entries.
* **Hardware status at a glance:** CPU, RAM, VRAM, GPU temp, disk space, and NVENC/NVDEC utilization, all live-polled.
* **Dark and Day themes**, switchable in Settings.
* **CSRF-protected, non-blocking:** settings changes and pipeline restarts run in the background — the UI stays responsive during a restart instead of freezing.
* **Resource-conscious:** the dashboard's live previews share frames with the recording pipeline's own decode instead of opening a second, redundant connection per camera whenever the pipeline is running.
* **`/health` endpoint + watchdog script** for external monitoring — pair with a cron job to auto-restart the dashboard if it ever goes unresponsive.

### Utilities
* **`backfill_thumbnails.py`:** a standalone script that grabs a frame (via ffmpeg) for older recordings from before the thumbnail feature existed — no detection boxes possible for these (the original inference data is long gone), but at least a visual reference instead of a blank entry. Doesn't touch the live pipeline.

## YOLO Model Comparison

IDguard PRO lets you switch the detection backend per deployment. Here's how they differ:

| Model | Best for | Architecture | Trade-offs |
| :--- | :--- | :--- | :--- |
| **YOLOv10** | Lean, predictable real-time detection | NMS-free end-to-end detection (pioneered the approach), dual-label assignment | Lower peak accuracy than v12/26, but very consistent latency — no NMS post-processing step to add jitter |
| **YOLOv12** | Maximum accuracy, GPU headroom to spare | Attention-centric design (Area Attention + R-ELAN) instead of pure CNN | Higher VRAM use and slower CPU throughput than v10/26; still relies on NMS + DFL; Ultralytics itself recommends v11/v26 over v12 for most production workloads |
| **YOLO26** | Best all-round default, especially on constrained hardware | Natively end-to-end (NMS-free, like v10) *and* removes Distribution Focal Loss (DFL) entirely — a simplification neither v10 nor v12 has | Up to ~43% faster CPU inference than the previous Ultralytics generation, deployment-first design; newest of the three, so less battle-tested in the wild |

**Rule of thumb:** YOLO26 is the sensible default for most setups. Reach for YOLOv10 if you want the simplest, most predictable latency profile. Reach for YOLOv12 only if you have GPU headroom to burn and accuracy matters more than efficiency.

## Hardware Requirements

* **GPU:** NVIDIA GPU with CUDA 12.8+ support recommended (RTX 20-series through 50-series). The pipeline auto-detects capability and degrades gracefully (FP16/cuDNN on/off) rather than requiring a specific generation — see [Tested Hardware](#tested-hardware--configurations) below.
* **OS:** Linux-based distribution (Ubuntu recommended).
* **Memory:** Minimum 8GB RAM (higher recommended for multiple simultaneous streams).
* **Storage:** Sufficient space for MP4 event recordings, filmstrip thumbnails, and (if archiving) long-term keepers. Auto-retention can cap unarchived storage growth automatically.
* **Optional:** A locally hosted [Ollama](https://ollama.com) instance with a vision-capable model, if you want AI scene descriptions. `llava` is the most broadly reliable choice (classic CLIP-based architecture); newer vision models vary in Ollama compatibility by build — the Settings page shows a live reachability check to help catch a broken model/endpoint quickly. See Settings → KI-Videoanalyse for the pull command and VRAM budget.

## Tech Stack

* **Language:** Python 3.x
* **Inference Engine:** PyTorch with CUDA support
* **Computer Vision:** Ultralytics (YOLOv10 / YOLOv12 / YOLO26), OpenCV, PyAV
* **Video I/O:** PyAV/ffmpeg with NVENC/NVDEC hardware acceleration and automatic software fallback
* **Web Framework:** Flask
* **Optional AI Analysis:** [Ollama](https://ollama.com) (any vision-capable model — `llava` recommended as a reliable default, with a model picker for others)
* **Process Management:** Threading, multiprocessing, and subprocess modules

## Installation

Detailed installation steps, including virtual environment setup and dependency management, are provided in the accompanying [INSTALL.md](./INSTALL.md) file.

## Project Structure

* `web_ui.py`: Flask web dashboard — routes, settings, event/thumbnail/filmstrip serving, log/health endpoints, Ollama connectivity check.
* `recorder_pipeline.py`: Core detection and recording logic — one process per camera, GPU-aware startup, filmstrip capture, shared live-preview frames, optional detection-box overlays.
* `ai_analyze.py`: Optional post-recording AI scene analysis via Ollama; writes dashboard metadata + Immich XMP sidecar.
* `backfill_thumbnails.py`: Standalone utility to generate thumbnails for older recordings that predate the thumbnail feature.
* `helpers.py`: Shared utilities for the dashboard — settings/override I/O, live-preview frame handling (reuses the pipeline's own decode when it's running).
* `config.py`: System-wide configuration, race-safe model auto-download, and defensive settings validation.
* `templates/dashboard.html`, `static/style.css`, `static/style-light.css`, `static/favicon.svg`: The dashboard UI, in dark and day themes.
* `start_detached.sh`, `stop.sh`: Pipeline lifecycle scripts with duplicate-instance and graceful-shutdown handling.
* `watchdog.sh`: Optional cron-friendly health check + auto-restart for the web dashboard.
* `alerts/`: Recorded event MP4s, trigger screenshots + confidence metadata, and filmstrip/AI metadata (auto-generated). Includes an `archive/` subfolder for permanently kept recordings.
* `logs/`: Application and system logs for debugging and auditing — also viewable directly in the dashboard.

## Tested Hardware & Configurations

The system has been thoroughly tested and runs rock-solid across both compact edge/SFF builds and high-end workstations — the GPU-aware startup (FP16/cuDNN auto-detection with fallback) means the same codebase runs unmodified on both ends of this range:

| System | Specs | Buffer Settings | Status |
| :--- | :--- | :--- | :--- |
| **Intel NUC 11 Enthusiast** | 32 GB RAM \| NVIDIA RTX 2060 (6 GB VRAM) | 4 streams 30 FPS FullHD / Pre-roll: 5s / Post-roll: 10s | Stable |
| **High-End Workstation** | Intel Core Ultra 9 285K \| 64 GB RAM \| NVIDIA RTX 5090 (32 GB VRAM) | 8 streams 30 FPS FullHD / Pre-roll: 10s / Post-roll: 30s | Stable |

---

## Acknowledgements & Citation

This project utilizes [YOLOv10](https://github.com/THU-MIG/yolov10), [YOLOv12](https://github.com/sunsmarterjie/yolov12), and YOLO26, and is powered by the [Ultralytics](https://github.com/ultralytics/ultralytics) framework for real-time object detection. Optional scene analysis is powered by [Ollama](https://ollama.com).

If you use this repository, please consider citing the original YOLOv10 paper:

```bibtex
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```

## Disclaimer

This software is intended for educational and private security purposes. Users are responsible for ensuring that their use of surveillance technology — including the optional AI scene-description feature — complies with all local, regional, and international laws regarding privacy and data protection.
