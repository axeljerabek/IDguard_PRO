# vigil

<img src="vigil-logo.svg" alt="IDguard PRO logo" width="800">

**A self-hosted camera system that watches, understands, and tells you what happened — in plain language, searchable, and now controllable by an AI agent.**

Instead of one AI trying to do everything, vigil chains together several small, specialized models: YOLO spots the moment something's worth recording, an audio model can trigger on sound alone, a vision-language model describes the clip afterward, Whisper transcribes speech, a face model recognizes who's there, and a text-embedding model makes it all searchable by meaning. Everything runs on your own GPU — no cloud, nothing leaves your network.

![Architecture Overview](architecture_overview.png)

[→ Full detailed architecture](ARCHITECTURE.html)

---

## What it can do

**Capture**

- RTMP, RTSP, MJPEG, USB/V4L2 webcams, or a watched import folder — any mix, per camera
- Packet-copy recording where possible (~1000× cheaper than re-encoding), automatic real-encoding fallback for cameras that need it
- Automatic HEVC/codec fixing so everything actually plays back in a browser

**Understand**

- YOLO object detection (v10/v12/26, your choice) as the trigger — not blind pixel-diffing
- CLAP audio triggers on sound alone, matched against categories *you* type
- Ollama vision model writes a plain-language description of every clip, aware of what you're actually watching for
- User-defined topic classification (`break-in`, `mail carrier`, anything) with confidence scores
- Whisper transcribes speech; InsightFace recognizes and groups people
- Isolation Forest anomaly detection flags recordings that don't match a camera's usual pattern — no need to know in advance what you're looking for

**Find & review**

- Semantic search across descriptions, topics, transcripts, and people — by meaning, not just keywords
- Daily/weekly AI-written summaries of what happened
- Star ratings, personal notes, and full Immich-compatible XMP export

**Connect & control**

- 🆕 **External API** — submit video for processing, get a webhook when it's done, pull a video segment or the full enriched metadata back
- 🆕 **Agent Control** — let an AI agent (Hermes, OpenClaw, or anything else) toggle cameras, tune settings, start/stop recording, and search, through a permission system with a master switch and per-capability toggles. **Off by default.**
- Home Assistant / MQTT integration with auto-discovery
- A dashboard built for this, not bolted on — live previews, filmstrip scrubbing, People management, hardware stats, dark/day themes

Every AI feature above is **off by default** and independently toggleable. Run it as a pure detection-triggered recorder, or turn on everything.

---

## Why not Motion or Frigate?

**[Motion/MotionEye](https://motion-project.github.io/)** triggers on pixel differences — reliable, decades-old, but it can't tell a cloud shadow from an intruder. Vigil replaced that with an actual object detector as the trigger from day one.

**[Frigate](https://frigate.video/)** is the closer, more mature comparison — it already does AI detection, semantic search, scene descriptions, transcription, and face recognition. Bigger community, wider hardware support (Coral, Hailo, Apple Silicon), a paid fine-tuning service. If you want a mature, broadly-compatible NVR, **Frigate is very likely the better choice today.**

What's different here is shape, not raw capability:

| | vigil | Typical NVR software |
| :--- | :--- | :--- |
| Audio trigger | Independent trigger source, free-text categories | Usually enrichment on already-recorded video |
| AI models | Wired together — detection classes shape the description prompt, one combined search signal | Often bolted on side by side |
| Agent control | Built-in, permissioned API for AI agents to operate the system | Not a thing yet, as far as we know |
| Design point | One strong GPU, single box | Often spread across cheap accelerators |
| Codebase | Small enough to read start to finish | Often years of accumulated features |

If your priority is a battle-tested, widely-supported project — pick Frigate. If you want the smaller, single-GPU version of the same idea, with agent-operability built in from the start — that's this.

---

## Documentation

| | |
| :--- | :--- |
| [INSTALL.md](./INSTALL.md) | Setup, virtual environment, dependencies |
| [DOCKER.md](./DOCKER.md) | Containerized install |
| [REMOTE_API.md](./REMOTE_API.md) | External API — job submission, webhooks, video/segment delivery |
| [AGENT_CONFIG.md](./AGENT_CONFIG.md) | Agent Control — permissions, endpoints, rollout |
| [HOME_ASSISTANT.md](./HOME_ASSISTANT.md) | MQTT setup, entities, example automations |

---

<details>
<summary><strong>Full feature details (click to expand)</strong></summary>

### Camera Input
* **RTMP, RTSP, MJPEG, and local USB/V4L2 webcams** (`/dev/video0` etc.) — just point a camera's URL field at any of these, no separate configuration needed. RTSP connections use TCP transport by default (more robust against packet loss than the ffmpeg default of UDP).
* **Automatic recording-path selection per camera:** cameras whose source codec plays back reliably in a browser (H.264, VP9, AV1) are recorded via packet copy — the already-compressed stream is written straight to disk, no re-encoding, ~1000× cheaper per frame. Cameras that don't (MJPEG, most raw USB webcam feeds) are recorded via real encoding instead (NVENC-accelerated where available, software fallback otherwise). Decided automatically per camera at connection time.
* **Watchfolder Import (optional):** point vigil at a folder instead of a camera, and it treats every video file dropped in there as a finished recording — waits for the file to stop growing, then imports it: renamed, filmstrip generated, run through the same AI post-processing as a live camera event.
* **Automatic playback-codec fix on import:** if an imported file's codec won't play reliably in a browser (HEVC being the common offender), it's transcoded to H.264 automatically — GPU-accelerated where available. Already-compatible video is only re-wrapped, never needlessly re-encoded.

### Detection & Recording
* **Switchable AI Backends:** YOLOv10, YOLOv12, and YOLO26, Nano through Extra Large.
* **Event-Driven Recording** with configurable pre-roll/post-roll buffers.
* **"Why did this trigger" readout** on every recording — object class and confidence, or the matched audio category — right in the dashboard.
* **Live Detection-Box Overlays** in the camera grid and live-view lightbox, reusing inference the pipeline already runs.
* **Trigger Screenshots** with the detection box drawn in, plus a confidence badge on the thumbnail.
* **Configurable Filmstrip Thumbnails** — reservoir sampling across the entire event plus a guaranteed final-frame slot, so long events still get full coverage.
* **Resilient by design:** cameras auto-restart on crash/disconnect; a failed AI model load keeps retrying in the background instead of going silently blind.
* **GPU-Aware Startup** detects the installed GPU (Turing through Blackwell) and self-tests FP16/cuDNN safety — same codebase runs unmodified from an RTX 2060 to an RTX 5090.
* **Accurate Timing:** wall-clock-based frame timestamps, so a network stall shows as a pause, not sped-up playback.
* **Race-Safe Model Download** for the YOLO checkpoint on first run.

### Optional AI Scene Description & Topic Classification (Ollama)
* Hands the filmstrip frames to a locally hosted **Ollama** vision model for a plain-language description.
* **Detection-aware prompt:** built dynamically around your configured YOLO classes and topics, so the model writes toward what you're actually watching for.
* **Topic classification:** your own categories (`break-in`, `accident`, `mail carrier`...) with a 0–100 confidence score each, all above-threshold topics shown and searchable. This score is the model's own self-reported guess, not a calibrated probability — a sort/filter signal, not a certainty.
* **Model picker** with tested presets plus a free-text option. Ollama has no native video input — every model receives the filmstrip image sequence, never the raw video.
* **Live connectivity badge** in Settings.
* Written as both dashboard JSON and an **Immich-compatible XMP sidecar**.
* Manual **re-analyze button** (re-runs transcription and face recognition too).

### Audio Trigger (CLAP)
* Triggers purely from **sound**, independent of visual detection — for anything outside the frame or in the dark.
* [CLAP](https://github.com/LAION-AI/CLAP) compares live audio against **freely typed categories**, not a fixed class list.
* Runs in its own background thread per camera — can never block or delay recording.
* Live-editable categories, no restart needed.

### Speech Transcription (Whisper)
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper), `tiny` through `large-v3`, GPU-accelerated with CPU fallback.
* Separate from the audio trigger — CLAP recognizes sound *categories*, this transcribes actual words.
* Coordinated (not parallel) with description/topic analysis so they never clobber the same metadata file.

### Face Recognition (InsightFace)
* Detects and embeds faces on the same filmstrip frames the description stage already uses — no extra extraction.
* **Model pack is your choice:** `buffalo_s/m/l` or `antelopev2`.
* Unmatched faces grouped via **DBSCAN clustering** — no need to pre-specify how many people to expect.
* Dedicated **People** section: named people with representative photo, unnamed clusters you can name or merge.
* **Fully correctable** at the individual-face level; false detections rejectable; unhelpful clusters hideable without deleting anything.

### Semantic Search
* Searches descriptions, topics, transcripts, **and named people** — by exact text *and* by meaning (`person carrying a box` also finds `individual holding a package`).
* Powered by a local sentence-embedding model (`all-MiniLM-L6-v2`) in a lightweight SQLite index — no external vector database needed.
* Falls back to plain text search if the embedding model isn't installed.
* Archive has its own dedicated search bar, same combined matching.

### Daily & Weekly Summaries
* Natural-language recap of a day/week's events, generated from existing AI descriptions — pure text summarization, same Ollama endpoint, no new infrastructure.
* Dashboard button or cronjob (example provided in the Summaries card).

### Home Assistant / MQTT (optional)
* Per-camera "Recording" motion sensor and "Last Event" description sensor to any MQTT broker.
* Home Assistant MQTT Discovery — entities appear automatically, no YAML.
* Fire-and-forget: publishing never delays or affects recording, even if the broker is down. See [HOME_ASSISTANT.md](./HOME_ASSISTANT.md).

### Anomaly Detection (optional)
* Flags recordings that are statistical outliers versus a camera's own recent history.
* Isolation Forest per camera, reusing the semantic-search embedding already computed for every event — no new model, no extra GPU cost.
* Needs a baseline: at least 15 analyzed recordings per camera in the lookback window (default 30 days); cameras with less history are skipped, not force-trained.

### External API (Remote Control)
* Submit video for the same processing pipeline as a live recording (codec handling, filmstrip, description, faces), with per-job topics overriding the global setting.
* Job-based: instant job ID, poll for status or get a webhook callback (with retry) when done.
* Delivers the processed video, an arbitrary time-range clip (`?start=&end=`, stream-copy, no re-encoding), and full enriched metadata.
* Own API-key auth, separate from the dashboard session — generate/revoke from the dashboard, keys stored as a hash. See [REMOTE_API.md](./REMOTE_API.md).

### Agent Control (optional, off by default)
* An AI agent can operate the pipeline directly through the same API — toggle cameras, tune settings, start/stop recording, search.
* **Two-layer permission gate:** a master switch plus a per-capability toggle (search, cameras, pipeline, settings) — all manageable from the dashboard, no config-file editing needed.
* **Settings changes are further restricted to a fixed allowlist enforced in code** — credentials, camera URLs, and export destinations are never reachable through this capability, regardless of the toggle.
* **Delete and export are intentionally not implemented** for agent use — no route exists, so there's nothing a config flag could accidentally enable.
* A `GET /capabilities` orientation endpoint lets an agent discover what it's allowed to do in one call instead of guessing via trial and error. See [AGENT_CONFIG.md](./AGENT_CONFIG.md).

### Export
* Bundles video, screenshot, all sidecar metadata, and the filmstrip folder into one named folder per event — or a shared subfolder when exporting several at once.
* Local path (direct copy) or remote `user@host:/path` (via `rsync`, assumes passwordless SSH already set up).
* Choose exactly what's included (video / metadata / large thumbs / small thumbs), remembered for next time.
* Optional delete-original-after-export, only ever after a confirmed successful export.
* Blocked automatically while a recording is still in progress.

### Web Dashboard
* CCTV-style layout: pipeline bar, live previews, and Recent Recordings up front; Settings, Hardware Status, People, Log, and Archive tucked into collapsible sections.
* Cameras, all AI features, and Agent Control managed entirely from the dashboard — no hand-editing config files.
* Live previews with a configurable refresh rate; REC badges everywhere, including the browser tab title.
* Card-style thumbnails with duration, day-grouped lists, hardware stats (CPU/RAM/VRAM/GPU temp/NVENC-NVDEC), dark and day themes.
* CSRF-protected, non-blocking settings changes; `/health` endpoint + watchdog script for external monitoring.

### Utilities
* `backfill_thumbnails.py`, `backfill_filmstrips.py`, `backfill_search_index.py` — retroactively generate thumbnails/filmstrips/search entries for older recordings.
* `cluster_faces.py` — on-demand face grouping.
* None of these touch the live pipeline — safe to run anytime, safe to re-run.

</details>

<details>
<summary><strong>Model comparisons (YOLO, Face Recognition)</strong></summary>

### YOLO

| Model | Best for | Trade-offs |
| :--- | :--- | :--- |
| **YOLOv10** | Lean, predictable real-time detection | Lower peak accuracy than v12/26, but very consistent latency (NMS-free) |
| **YOLOv12** | Maximum accuracy, GPU headroom to spare | Higher VRAM/CPU cost; Ultralytics itself recommends v11/v26 for most production workloads |
| **YOLO26** | Best all-round default, especially on constrained hardware | Newest of the three, so less battle-tested; up to ~43% faster CPU inference than the previous generation |

**Rule of thumb:** YOLO26 by default. YOLOv10 for the simplest latency profile. YOLOv12 only with GPU headroom to spare.

### Face Recognition

| Model Pack | Best for | Trade-offs |
| :--- | :--- | :--- |
| **buffalo_s** | Fastest, lowest resource use | May miss more at odd angles or in poor light |
| **buffalo_m** | Balanced default | Middle ground on speed vs. accuracy |
| **buffalo_l** | Best accuracy of the buffalo packs | Larger, more compute per frame |
| **antelopev2** | Highest accuracy overall | Largest/slowest; IDguard PRO auto-fixes a known packaging quirk on first load |

</details>

<details>
<summary><strong>Tech stack</strong></summary>

* **Language:** Python 3.x
* **Inference:** PyTorch with CUDA
* **Computer Vision:** Ultralytics (YOLOv10/v12/26), OpenCV, PyAV
* **Video I/O:** PyAV/ffmpeg — NVDEC decode, packet-copy recording where possible, NVENC fallback otherwise
* **Web:** Flask
* **AI Analysis:** [Ollama](https://ollama.com)
* **Audio Trigger:** [CLAP](https://github.com/LAION-AI/CLAP) via `transformers`
* **Transcription:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
* **Face Recognition:** [InsightFace](https://github.com/deepinsight/insightface) + `onnxruntime`, `scikit-learn` (DBSCAN + Isolation Forest)
* **Semantic Search:** `sentence-transformers` (`all-MiniLM-L6-v2`) + SQLite
* **Export:** local copy or `rsync`
* **Process management:** threading, multiprocessing, subprocess

</details>

<details>
<summary><strong>Project structure</strong></summary>

* `web_ui.py`: Flask dashboard — routes, settings, camera management, search, export, People API.
* `recorder_pipeline.py`: Core detection/recording — one process per camera, GPU-aware startup, packet-copy recording, filmstrip capture.
* `watch_folder.py`: Optional folder-based import.
* `daily_summary.py`: Daily/weekly narrative summaries.
* `mqtt_client.py`: MQTT / Home Assistant integration.
* `anomaly_detection.py`: Per-camera anomaly detection.
* `mam_api.py`: External API — job submission, status, webhooks, plus gated agent-control routes.
* `agent_permissions.py`, `agent_config.json`: Agent Control permission gate.
* `postprocess.py`: Sequences description/topics, transcription, and face recognition for a finished recording.
* `ai_analyze.py`: Ollama scene analysis and topic classification; writes dashboard metadata + Immich XMP.
* `audio_trigger.py`: CLAP-based audio trigger.
* `transcribe_audio.py`: Whisper transcription.
* `face_recognize.py`, `faces_db.py`, `cluster_faces.py`: Face detection, storage, and clustering.
* `search_index.py`: SQLite-backed full-text + semantic search index.
* `backfill_thumbnails.py`, `backfill_filmstrips.py`, `backfill_search_index.py`: Retroactive utilities.
* `helpers.py`, `config.py`: Shared utilities and system-wide configuration.
* `templates/dashboard.html`, `static/`: Dashboard UI, dark and day themes.
* `start_detached.sh`, `stop.sh`, `watchdog.sh`: Pipeline lifecycle scripts.
* `Dockerfile`, `docker-compose.yml`: Containerized setup.
* `alerts/`: Recorded events + metadata, with an `archive/` subfolder.
* `logs/`: Application and system logs.
* `search_index.db`, `faces.db`, `streams.json`, `pipeline_settings.json`, `stream_overrides.json`: Local, gitignored runtime data.

</details>

---

## Hardware Requirements

* **GPU:** NVIDIA, CUDA 12.8+ recommended (RTX 20-series through 50-series) — auto-detects capability and degrades gracefully.
* **OS:** Linux (Ubuntu recommended).
* **Memory:** 8GB+ RAM.
* **Optional:** [Ollama](https://ollama.com) with a vision model for AI descriptions — `llava` is the most broadly reliable choice.

**Tested configurations:**

| System | Specs | Load | Status |
| :--- | :--- | :--- | :--- |
| Intel NUC 11 Enthusiast | 32 GB RAM, RTX 2060 (6 GB) | 4 streams, 30 FPS FullHD | Stable |
| High-End Workstation | Core Ultra 9 285K, 64 GB RAM, RTX 5090 (32 GB) | 8 streams, 30 FPS FullHD | Stable |

---

## Acknowledgements

Built on [YOLOv10](https://github.com/THU-MIG/yolov10), [YOLOv12](https://github.com/sunsmarterjie/yolov12), YOLO26, and [Ultralytics](https://github.com/ultralytics/ultralytics). Optional components: [Ollama](https://ollama.com), [CLAP](https://github.com/LAION-AI/CLAP), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [InsightFace](https://github.com/deepinsight/insightface), [sentence-transformers](https://www.sbert.net/).

Parts of this code were written with AI (Google Gemini, Claude, Claude Code, and local models). The architecture and the work of making it all run reliably was a person's job.

<details>
<summary>Citation</summary>

```bibtex
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```
</details>

## Disclaimer

Intended for educational and private security purposes. You're responsible for ensuring your use of surveillance technology — especially AI description, transcription, and face recognition — complies with local privacy and data protection law. Face recognition and transcription carry meaningfully higher privacy stakes than object detection alone.
