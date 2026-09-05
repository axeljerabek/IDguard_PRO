# vigil

<img src="vigil-logo.svg" alt="vigil logo" width="1000">

**Most camera systems record video and make you scroll through it. Vigil watches, understands, and tells you what happened — in plain English, searchable by meaning, and now operable by an AI agent.**

It hears glass breaking in a room the camera can't see. It notices when someone falls, or stays too long in the same spot, or moves unusually fast. It writes an actual sentence describing what happened instead of a bounding box. It flags the one recording that doesn't look like anything else on that camera, without you ever defining what "weird" means. And once it's learned who someone is, it doesn't forget them just because you cleaned up old footage.

Instead of one AI trying to do everything, vigil chains together several small, specialized models — a detector, an audio listener, a pose analyzer, a vision-language model, a face model, a text-embedding model — each doing the one job it's good at, with automatic handoffs between them. Everything runs on your own GPU. No cloud, no subscription, nothing leaves your network.

![Architecture Overview](architecture_overview.png)

[→ Full detailed architecture](ARCHITECTURE.html)

---

## What makes vigil different

**It hears what it can't see.** An independent audio model (CLAP) listens continuously, with zero fixed categories — type "glass breaking" or "dog barking" into the dashboard in plain English and it's live immediately, no retraining. A break-in two rooms away, in the dark, out of frame — vigil still knows.

**It notices when someone's in trouble.** One pose model, reused at zero extra GPU cost from a person already spotted by the main detector, reads six independent behavior signals: a fall, a raised-hands distress gesture, loitering, unusually fast movement, sustained close proximity between people, and which way someone's facing. Each is its own switch with its own threshold — turn on just fall detection, or all six.

**It finds the recording that doesn't fit — without you writing a single rule.** A per-camera Isolation Forest model learns what "normal" looks like over time and flags statistical outliers automatically, recycling the same embedding semantic search already computes. Zero extra GPU cycles.

**It learns — and doesn't forget.** Once a face is identified, its recognition data is archived independently of any single video. Clean up old footage a year later, and vigil still recognizes that person tomorrow — the system is designed to keep learning, not to quietly lose what it already knew.

**An AI agent can run it for you.** This is the one nobody else has. A native MCP server exposes 21 tools — the same ones any script can call over the REST API — so an agent like Claude can check what a camera saw, toggle cameras, kick off a recording, search past events, and read system health, through a real typed interface instead of screen-scraping a dashboard. A layered, off-by-default permission system decides exactly what it's allowed to touch; deletion is intentionally unreachable no matter what's toggled on.

**It records on demand, pipeline running or not.** Ask for thirty seconds from a specific camera right now, and it just happens — connects directly, records the exact duration, runs the same AI analysis afterward — whether or not the detection pipeline is even active.

**You can search it by meaning, not keywords.** "Person carrying a box" finds "individual holding a package." Same semantic search across descriptions, transcripts, topics, and named people together.

---

## vigil vs. Motion vs. Frigate

| | Motion / MotionEye | Frigate NVR | vigil |
| :--- | :--- | :--- | :--- |
| Trigger | Pixel difference | Object detection (YOLO) | YOLO + audio (CLAP) + pose/behavior |
| Model interaction | None | Isolated / parallel | Detection feeds the description prompt |
| Face recognition | None | External / third-party | In-process; identity survives video deletion |
| Anomaly detection | None | None | Isolation Forest, zero extra GPU cost |
| Agent control | None | None | Native MCP server, 21 tools |
| Ad-hoc recording | — | — | Works even with the pipeline stopped |
| Home Assistant | Basic webhooks | Deep UI integration | Native MQTT auto-discovery |
| Photo manager export | Standard MP4 | Internal DB/API | Immich-compatible XMP sidecars |

**[Motion/MotionEye](https://motion-project.github.io/)** triggers on pixel differences — reliable, decades-old, but it can't tell a cloud shadow from an intruder. Choose it if you want a simple daemon on minimal hardware with no GPU.

**[Frigate](https://frigate.video/)** is the closer, more mature comparison — it already does AI detection, semantic search, scene descriptions, transcription, and face recognition, with a bigger community and wider hardware support (Coral, Hailo, Apple Silicon). If you want a battle-tested, broadly-compatible NVR, **Frigate is very likely the better choice today.**

Choose vigil if you want one strong GPU doing real reasoning about what it sees and hears, with the option to hand an AI agent the keys.

---

## What it can do

**Understand**

- YOLO object detection (v10/v12/26, your choice) as the trigger — not blind pixel-diffing
- CLAP audio triggers on sound alone, matched against categories *you* type
- **Six independent pose/behavior signals** from one model, reused at zero extra cost: fall detection, raised-hands distress gesture, loitering, fast movement, sustained close proximity, and head/gaze orientation — each its own toggle and threshold
- Ollama vision model writes a plain-language description of every clip, aware of what you're actually watching for
- User-defined topic classification (`break-in`, `mail carrier`, anything) with confidence scores
- Whisper transcribes speech; InsightFace recognizes and groups people — **identity persists even after the source video is deleted**
- Isolation Forest anomaly detection flags recordings that don't match a camera's usual pattern — no need to know in advance what you're looking for

**Connect & control**

- 🆕 **MCP Server** — 21 native tools exposing the full agent-control surface to any MCP-compatible client (Claude Desktop, etc.), not just raw HTTP
- **External API** — submit video for processing, get a webhook when it's done, pull a video segment or the full enriched metadata back
- **Quick Record** — ask for N seconds from any camera right now, independent of whether the detection pipeline is even running
- **Agent Control** — let an AI agent toggle cameras, tune settings, start/stop recording, search, retrain anomaly baselines, and get proactively notified via webhook — through a permission system with a master switch and per-capability toggles. **Off by default.**
- Home Assistant / MQTT integration with auto-discovery

**Find & review**

- Semantic search across descriptions, topics, transcripts, and people — by meaning, not just keywords
- Daily/weekly AI-written summaries of what happened — generate, regenerate, or delete any of them from the dashboard
- Star ratings, personal notes, a user-chosen profile photo per person, and full Immich-compatible XMP export

**Capture**

- RTMP, RTSP, MJPEG, USB/V4L2 webcams, or a watched import folder — any mix, per camera
- Packet-copy recording where possible (~1000× cheaper than re-encoding), automatic real-encoding fallback for cameras that need it
- Automatic HEVC/codec fixing so everything actually plays back in a browser

Every AI feature above is **off by default** and independently toggleable. Run it as a pure detection-triggered recorder, or turn on everything.

---

## Documentation

| | |
| :--- | :--- |
| [INSTALL.md](./INSTALL.md) | Setup, virtual environment, dependencies |
| [DOCKER.md](./DOCKER.md) | Containerized install |
| [REMOTE_API.md](./REMOTE_API.md) | External API — job submission, webhooks, video/segment delivery |
| [AGENT_CONFIG.md](./AGENT_CONFIG.md) | Agent Control — permissions, endpoints, quick-reference, rollout |
| [MCP_SERVER.md](./MCP_SERVER.md) | MCP server setup — 21 tools, Claude Desktop config |
| [HOME_ASSISTANT.md](./HOME_ASSISTANT.md) | MQTT setup, entities, example automations |

---

<details>
<summary><strong>Full feature details (click to expand)</strong></summary>

### Camera Input
* **RTMP, RTSP, MJPEG, and local USB/V4L2 webcams** (`/dev/video0` etc.) — just point a camera's URL field at any of these, no separate configuration needed. RTSP connections use TCP transport by default (more robust against packet loss than the ffmpeg default of UDP).
* **Automatic recording-path selection per camera:** cameras whose source codec plays back reliably in a browser (H.264, VP9, AV1) are recorded via packet copy — the already-compressed stream is written straight to disk, no re-encoding, ~1000× cheaper per frame. Cameras that don't (MJPEG, most raw USB webcam feeds) are recorded via real encoding instead (NVENC-accelerated where available, software fallback otherwise). Decided automatically per camera at connection time.
* **Watchfolder Import (optional):** point vigil at a folder instead of a camera, and it treats every video file dropped in there as a finished recording — waits for the file to stop growing, then imports it: renamed, filmstrip generated, run through the same AI post-processing as a live camera event.
* **Automatic playback-codec fix on import:** if an imported file's codec won't play reliably in a browser (HEVC being the common offender), it's transcoded to H.264 automatically — GPU-accelerated where available. Already-compatible video is only re-wrapped, never needlessly re-encoded.
* **Quick Record:** an ad-hoc, fixed-duration recording from any camera, completely independent of the detection pipeline's state — works even with the pipeline stopped or that camera disabled everywhere else. Connects directly, records for exactly the requested duration, generates a filmstrip and thumbnail, and runs the same AI analysis as a normal event afterward.

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
* **Manual trigger/stop:** force-start or immediately end a recording on a specific camera on demand — separate from, and more precise than, simply enabling/disabling the camera.

### Pose Estimation & Behavior Detection (optional)
* One small pose model, run only on frames where a person is already detected by the main detector — no extra GPU cost on empty scenes.
* **Fall detection:** torso-angle heuristic (near-horizontal vs. upright), confirmed over several consecutive frames so a brief bend — tying shoes, picking something up — doesn't read as a fall. Angle threshold is adjustable.
* **Raised-hands distress signal:** both wrists held well above the shoulders, confirmed over a shorter window than a fall (a deliberate gesture doesn't need as long to confirm). One hand alone is treated as an ordinary gesture and ignored.
* **Loitering:** a person staying in roughly the same spot for longer than a configurable duration — pure position tracking, no pose model needed. Resets the moment the person leaves the frame or moves on, so a normal walk-through never triggers it.
* **Fast movement:** speed measured in "body heights per second," not pixels — the same real-world pace reads the same whether the person is close to or far from the camera, no calibration needed.
* **Close proximity:** flags two or more people staying near each other for a sustained period, not a brief pass-by.
* **Head orientation:** facing the camera vs. facing away, based on whether facial keypoints are visible at all — informational, not an alert.
* Fall, distress, fast movement, and sustained proximity all force a recording to start if one isn't already running; head orientation is logged only.

### Optional AI Scene Description & Topic Classification (Ollama)
* Hands the filmstrip frames to a locally hosted **Ollama** vision model for a plain-language description.
* **Detection-aware prompt:** built dynamically around your configured YOLO classes and topics, so the model writes toward what you're actually watching for.
* **Topic classification:** your own categories (`break-in`, `accident`, `mail carrier`...) with a 0–100 confidence score each, all above-threshold topics shown and searchable. This score is the model's own self-reported guess, not a calibrated probability — a sort/filter signal, not a certainty.
* **Model picker** with tested presets plus a free-text option. Ollama has no native video input — every model receives the filmstrip image sequence, never the raw video.
* **Live connectivity badge** in Settings.
* Written as both dashboard JSON and an **Immich-compatible XMP sidecar**.
* Manual **re-analyze button** (re-runs transcription and face recognition too), also available to an agent via the API.

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
* **Identity persists independently of any single video.** The moment a face is assigned to a named person, their photo is archived permanently and their recognition embedding is preserved even if every video that ever showed them gets deleted later — the system keeps learning, it doesn't quietly forget someone because old footage was cleaned up.
* Dedicated **People** section: named people with a representative photo you can pick yourself from any of their recognized faces, unnamed clusters you can name or merge.
* **Fully correctable** at the individual-face level: reject a false detection, unassign a face back to the unassigned pool, or select specific faces within a mixed cluster to merge into a *different* existing person instead of always naming the whole group at once.
* **Two levels of removing a person:** "un-name" un-assigns their faces for re-clustering but keeps all data; "delete permanently" removes their recognition data and archived photos entirely, for people you genuinely don't need in the system anymore.

### Semantic Search
* Searches descriptions, topics, transcripts, **and named people** — by exact text *and* by meaning (`person carrying a box` also finds `individual holding a package`).
* Powered by a local sentence-embedding model (`all-MiniLM-L6-v2`) in a lightweight SQLite index — no external vector database needed.
* Falls back to plain text search if the embedding model isn't installed.
* Archive has its own dedicated search bar, same combined matching.

### Daily & Weekly Summaries
* Natural-language recap of a day/week's events, generated from existing AI descriptions — pure text summarization, same Ollama endpoint, no new infrastructure.
* Dashboard button or cronjob (example provided in the Summaries card).
* Any summary can be regenerated (e.g. after re-analyzing events it covers) or deleted directly from the dashboard.

### Home Assistant / MQTT (optional)
* Per-camera "Recording" motion sensor and "Last Event" description sensor to any MQTT broker.
* Home Assistant MQTT Discovery — entities appear automatically, no YAML.
* Fire-and-forget: publishing never delays or affects recording, even if the broker is down. See [HOME_ASSISTANT.md](./HOME_ASSISTANT.md).

### Anomaly Detection (optional)
* Flags recordings that are statistical outliers versus a camera's own recent history.
* Isolation Forest per camera, reusing the semantic-search embedding already computed for every event — no new model, no extra GPU cost.
* Needs a baseline: at least 15 analyzed recordings per camera in the lookback window (default 30 days); cameras with less history are skipped, not force-trained.
* Training can be triggered from the dashboard, via cron, or by an agent through the API.

### External API (Remote Control)
* Submit video for the same processing pipeline as a live recording (codec handling, filmstrip, description, faces), with per-job topics overriding the global setting.
* Job-based: instant job ID, poll for status or get a webhook callback (with retry) when done.
* Delivers the processed video, an arbitrary time-range clip (`?start=&end=`, stream-copy, no re-encoding), and full enriched metadata.
* Own API-key auth, separate from the dashboard session — generate/revoke from the dashboard, keys stored as a hash. See [REMOTE_API.md](./REMOTE_API.md).

### Agent Control (optional, off by default)
* An AI agent can operate the pipeline directly through the same API — toggle cameras, tune settings, start/stop the pipeline, force-trigger or stop a specific recording, quick-record on demand, search, read event/summary/system-status details, and retrain anomaly baselines.
* **Two-layer permission gate:** a master switch plus a per-capability toggle — all manageable from the dashboard, no config-file editing needed.
* **Settings changes are further restricted to a fixed allowlist enforced in code** — credentials, camera URLs, and export destinations are never reachable through this capability, regardless of the toggle.
* **Delete and export are intentionally not implemented** for agent use — no route exists, so there's nothing a config flag could accidentally enable.
* **Proactive notifications:** configure a webhook URL and vigil pushes an event to it after every analyzed recording (or only for anomalies) — an agent doesn't have to poll for changes.
* A `GET /capabilities` orientation endpoint lets an agent discover what it's allowed to do in one call instead of guessing via trial and error. See [AGENT_CONFIG.md](./AGENT_CONFIG.md).

### MCP Server
* Exposes the entire Agent Control surface as 21 native MCP tools instead of raw HTTP calls — for Claude Desktop or any other MCP-compatible client.
* A thin wrapper, not a new permission surface: every tool calls the exact same gated API endpoint, with the exact same checks enforced server-side.
* Same setup as any other API client — generate a key from the dashboard, point the server at your vigil instance. See [MCP_SERVER.md](./MCP_SERVER.md).

### Export
* Bundles video, screenshot, all sidecar metadata, and the filmstrip folder into one named folder per event — or a shared subfolder when exporting several at once.
* Local path (direct copy) or remote `user@host:/path` (via `rsync`, assumes passwordless SSH already set up).
* Choose exactly what's included (video / metadata / large thumbs / small thumbs), remembered for next time.
* Optional delete-original-after-export, only ever after a confirmed successful export.
* Blocked automatically while a recording is still in progress.

### Web Dashboard
* CCTV-style layout: pipeline bar, live previews, and Recent Recordings up front; Settings, Hardware Status, People, Summaries, Anomaly Detection, Log, and Archive tucked into collapsible sections.
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
| **antelopev2** | Highest accuracy overall | Largest/slowest; vigil auto-fixes a known packaging quirk on first load |

</details>

<details>
<summary><strong>Tech stack</strong></summary>

* **Language:** Python 3.x
* **Inference:** PyTorch with CUDA
* **Computer Vision:** Ultralytics (YOLOv10/v12/26, pose models), OpenCV, PyAV
* **Video I/O:** PyAV/ffmpeg — NVDEC decode, packet-copy recording where possible, NVENC fallback otherwise
* **Web:** Flask
* **AI Analysis:** [Ollama](https://ollama.com)
* **Audio Trigger:** [CLAP](https://github.com/LAION-AI/CLAP) via `transformers`
* **Transcription:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
* **Face Recognition:** [InsightFace](https://github.com/deepinsight/insightface) + `onnxruntime`, `scikit-learn` (DBSCAN + Isolation Forest)
* **Semantic Search:** `sentence-transformers` (`all-MiniLM-L6-v2`) + SQLite
* **Agent Integration:** MCP (`mcp` SDK), REST/webhooks, MQTT
* **Export:** local copy or `rsync`
* **Process management:** threading, multiprocessing, subprocess

</details>

<details>
<summary><strong>Project structure</strong></summary>

* `web_ui.py`: Flask dashboard — routes, settings, camera management, search, export, People API.
* `recorder_pipeline.py`: Core detection/recording — one process per camera, GPU-aware startup, packet-copy recording, filmstrip capture, pose/behavior detection.
* `pose_fall_detection.py`: Fall, raised-hands, head-orientation, and pointing-gesture heuristics from pose keypoints.
* `loitering_detection.py`: Position-based loitering, movement-speed, and proximity detection (no pose model needed).
* `watch_folder.py`: Optional folder-based import.
* `quick_record.py`: Ad-hoc, pipeline-independent fixed-duration recording.
* `daily_summary.py`: Daily/weekly narrative summaries.
* `mqtt_client.py`: MQTT / Home Assistant integration.
* `agent_webhook.py`: Proactive event notifications to an agent, fire-and-forget.
* `anomaly_detection.py`: Per-camera anomaly detection.
* `mam_api.py`: External API — job submission, status, webhooks, plus gated agent-control routes.
* `mcp_server.py`: MCP server exposing the agent-control API as 21 tools.
* `agent_permissions.py`, `agent_config.json`: Agent Control permission gate.
* `postprocess.py`: Sequences description/topics, transcription, and face recognition for a finished recording.
* `ai_analyze.py`: Ollama scene analysis and topic classification; writes dashboard metadata + Immich XMP.
* `audio_trigger.py`: CLAP-based audio trigger.
* `transcribe_audio.py`: Whisper transcription.
* `face_recognize.py`, `faces_db.py`, `cluster_faces.py`: Face detection, permanent identity storage, and clustering.
* `search_index.py`: SQLite-backed full-text + semantic search index.
* `backfill_thumbnails.py`, `backfill_filmstrips.py`, `backfill_search_index.py`: Retroactive utilities.
* `helpers.py`, `config.py`: Shared utilities and system-wide configuration.
* `templates/dashboard.html`, `static/`: Dashboard UI, dark and day themes.
* `start_detached.sh`, `stop.sh`, `watchdog.sh`: Pipeline lifecycle scripts.
* `Dockerfile`, `docker-compose.yml`: Containerized setup.
* `alerts/`: Recorded events + metadata, with an `archive/` subfolder and a `.people_photos/` permanent identity archive.
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

Built on [YOLOv10](https://github.com/THU-MIG/yolov10), [YOLOv12](https://github.com/sunsmarterjie/yolov12), YOLO26, and [Ultralytics](https://github.com/ultralytics/ultralytics). Optional components: [Ollama](https://ollama.com), [CLAP](https://github.com/LAION-AI/CLAP), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [InsightFace](https://github.com/deepinsight/insightface), [sentence-transformers](https://www.sbert.net/), the [MCP](https://modelcontextprotocol.io/) SDK.

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
