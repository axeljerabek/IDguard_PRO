# 🛡️ VIGIL - CONFIGURATION MANUAL

**Version:** 3.0
**Purpose:** Complete reference for every configurable parameter — both in `config.py` and in the live dashboard Settings.

---

## 0. Where does a setting actually live? (Read this first)

This is the single most important thing that changed since v1.0 of this manual: **`config.py` is no longer where you edit most settings.**

* `config.py` still holds the **camera list** (`STREAMS`) and the **starting defaults** for everything else — these are read once, at process start.
* Almost every operational setting (detection sensitivity, recording timing, thumbnails, retention, theme, AI analysis, pose/behavior detection, anomaly detection, MQTT, agent control, export, ...) is now stored in **`pipeline_settings.json`**, edited live from the **dashboard's Settings page**, and takes effect either immediately (no restart) or after an automatic background restart of the pipeline — the dashboard tells you which.
* `config.py` reads `pipeline_settings.json` once at import time as a *fallback/default source* for the handful of values that still require a full pipeline restart to change (`YOLO_VERSION`, `MODEL_SIZE`, `TARGET_FPS`, `PRE_ROLL_SEC`, `POST_ROLL_SEC`, `CONFIDENCE_THRESHOLD`, `DETECTION_CLASSES`, and — new — the pose-estimation toggles, since loading the pose model happens once at camera-process startup). Everything else in `pipeline_settings.json` is read directly, live, by whichever component needs it — deliberately, so those settings apply without a restart.

**Bottom line:** unless you're changing the camera list itself, use the dashboard, not this file.

---

## 1. CORE SYSTEM PATHS (AUTO-MANAGED)

All paths are resolved relative to the project root. You do not need to set these manually unless you want to override the default directory structure.

*   **`PROJECT_ROOT`**: The absolute path of your installation folder.
*   **`ALERTS_DIR`**: Destination for all `.mp4` event files, trigger screenshots, and their metadata sidecars.
    - *Location*: `{PROJECT_ROOT}/alerts/`
    - Contains an `archive/` subfolder with the identical structure for permanently kept recordings.
    - Contains a `.people_photos/` subfolder: permanent, video-independent storage for named people's face photos (see §9). Contains `.summaries/`, `.detections/`, `.thumbs/`, `.triggers/`, `.stops/`, and `.quick_record_jobs/` — all internal, safe to ignore.
*   **`LOG_DIR`**: Storage for system and per-camera logs.
    - *Location*: `{PROJECT_ROOT}/logs/`
*   **`MODEL_PATH`**: The absolute path to your active YOLO weights (`.pt`), auto-downloaded on first use if missing.
*   **`BROWSER_COMPATIBLE_VIDEO_CODECS`**: `{"h264", "vp9", "av1"}` — codecs a camera's source stream can have and still be recorded via cheap packet-copy. Anything else (MJPEG, most raw USB feeds) falls back to real encoding automatically. Not meant to be edited; documented here because it's referenced throughout this manual and the README.
*   **`OVERRIDE_F`**: `{PROJECT_ROOT}/stream_overrides.json` — which cameras are currently toggled on/off from the dashboard (independent of the `enabled` default in `STREAMS`).
*   **`SETTINGS_F`**: `{PROJECT_ROOT}/pipeline_settings.json` — where essentially everything described in this manual actually lives. Safe to inspect directly; edit it through the dashboard, not by hand, to avoid an invalid combination of values.

---

## 2. STREAM CONFIGURATION (`STREAMS` in `config.py`)

The `STREAMS` list defines the active cameras. This is the one thing you still edit directly in `config.py`. Each entry is a dictionary:

| Key | Type | Description |
| :--- | :--- | :--- |
| **`name`** | `string` | Unique ID (used in logs, filenames, alerts, dashboard grouping, and MQTT/agent-API topic paths). |
| **`url`** | `string` | The RTMP/RTSP/MJPEG stream origin URL, a `/dev/videoX` path for a local USB webcam, **or a YouTube/Twitch/Vimeo/Facebook/Dailymotion/Kick URL** — resolved automatically via `yt-dlp` into a direct stream URL, re-resolved fresh on every (re)connect since these are typically time-limited/signed. Requires `yt-dlp` installed (`pip install yt-dlp`). If the channel isn't currently live, vigil waits 30s before trying again rather than reconnecting every 5s like a normal camera. |
| **`enabled`**| `boolean`| Default state; can be overridden per-camera live from the dashboard or the agent API (stored in `stream_overrides.json`/`streams.json`, takes precedence over this default). Toggling it live takes up to ~15s to actually start/stop the camera's process — the pipeline reconciles against the file on a periodic cycle, not instantly on save. |
| **`type`** | `string` | Currently supports `"VIDEO"`. |
| **`audio_enabled`** | `boolean` | Whether this camera's audio is used for the CLAP trigger (§7b). Default `true` for existing entries without this field. |
| **`notify_only`** | `boolean` | If `true`, YOLO keeps detecting on this camera but never auto-starts a recording — it only reports what it saw (§10, agent API `/detections`). Recording then requires an explicit trigger, e.g. from an agent. Default `false`. Takes effect on the camera's next process restart. |

**Example Configuration:**
```python
STREAMS = [
    {"name": "Entrance_Main", "url": "rtmp://192.168.1.50/pi/test", "enabled": True, "type": "VIDEO", "audio_enabled": True, "notify_only": False},
    {"name": "Garden_North",  "url": "rtmp://192.168.1.50/garden/live", "enabled": False, "type": "VIDEO"}
]
```

---

## 3. AI MODEL SELECTION (dashboard: Settings → KI-Modell)

*   **`YOLO_VERSION`**: `"v10"`, `"v12"`, or `"v26"`. See the model comparison table in `README.md` for the trade-offs between them. Changing this restarts the pipeline (a different model needs to be loaded).
*   **`MODEL_SIZE`**: `"n"`, `"s"`, `"m"`, `"b"` (v10 only), `"l"`, or `"x"` — Nano through Extra-Large. Bigger sizes are more accurate but slower and use more VRAM.
*   The pipeline auto-detects your GPU generation at startup and safely enables/disables FP16 and cuDNN via a staged self-test — no manual tuning needed here regardless of which card you run (RTX 2060 through RTX 5090).

---

## 4. DETECTION & AI PARAMETERS (dashboard: Settings → Erkennung)

Controls the sensitivity of the detection engine.

*   **`DETECTION_CLASSES`**: **[CRITICAL]** A list of integer IDs representing the objects you want to track (COCO classes). Picked via the category checklist in the dashboard, not typed by hand.
    - `0`: Person (Default)
    - `1`: Bicycle
    - `2`: Car
    - *Example*: `[0, 2]` triggers alerts for both humans and vehicles.
    - Also changeable by an agent through the API (§11), within the same settings allowlist.
*   **`CONFIDENCE_THRESHOLD`**: A float (`0.0` to `1.0`). How certain the AI must be before it triggers an alert.
*   **`SHOW_DETECTION_BOXES`**: Whether live camera previews and the live-view lightbox draw detection boxes in real time (color-coded per class). Purely cosmetic, reuses inference the pipeline already runs — no extra GPU cost either way.

---

## 5. TEMPORAL & RECORDING LOGIC (dashboard: Settings → Aufnahme-Timing)

Controls buffer management and video duration per event.

*   **`PRE_ROLL_SEC`**: Seconds of "pre-event" footage kept in the circular buffer, written to the *start* of the `.mp4` when an event triggers.
    - *Warning*: high values increase memory use, especially for cameras running in encode mode (MJPEG/USB) rather than packet-copy mode.
*   **`POST_ROLL_SEC`**: Seconds to keep recording **after** the target object has left the frame, to capture the "exit" phase.
*   **`TARGET_FPS`**: Target frames per second for the output video. Also used to throttle how often the pipeline runs inference at all — source frames arriving faster than this are skipped before the (comparatively expensive) BGR conversion and detection step.

---

## 6. THUMBNAILS, FILMSTRIP & DISPLAY (dashboard: Settings → Anzeige / Speicher)

*   **`THUMBNAIL_FPS`** (0.5–5): Refresh rate for camera grid previews and the live-view lightbox. Live, no restart.
*   **`FILMSTRIP_COUNT`** (0 = off): Number of small+large preview frames captured per recording, starting right after pre-roll. Small frames (with detection boxes) power the hover-to-scrub preview in the dashboard; large frames (kept raw) are what gets sent to Ollama if AI analysis is enabled. Includes a guaranteed final-frame slot so long recordings still get end-of-clip coverage.
*   **`FILMSTRIP_INTERVAL_SEC`**: Seconds between filmstrip captures.
*   **`RETENTION_DAYS`** (0 = never): Auto-delete unarchived recordings older than this. Archived recordings are never touched.
*   **`THEME`**: `"dark"` or `"light"`.

---

## 7a. OPTIONAL AI SCENE DESCRIPTION & TOPIC CLASSIFICATION (dashboard: Settings → KI-Videoanalyse)

All off by default — no Ollama instance required unless enabled.

*   **`AI_ANALYSIS_ENABLED`**: Master on/off switch.
*   **`OLLAMA_URL`**: Endpoint of your Ollama instance, e.g. `http://localhost:11434`. A live reachability badge in this settings section shows whether it's currently answering. Also the endpoint used for daily/weekly summaries (§7c) — no separate configuration needed.
*   **`OLLAMA_VISION_MODEL`**: Which model to send the filmstrip frames to. Pick from the dropdown of tested presets, or "Eigenes Modell…" for anything else you've pulled into Ollama. Note: Ollama has no native video-file input — every model gets the same image sequence (the large filmstrip frames), never the raw `.mp4`.
*   **`AI_ANALYZE_MAX_FRAMES`** (1–64): How many filmstrip frames to send per analysis. More frames = more context but more tokens/time per request.
*   **`AI_TOPICS_ENABLED`**: Turns on topic classification (separate from the description itself).
*   **`AI_TOPICS`**: Your own free-text categories (e.g. `break-in`, `mail carrier`, `accident`) — the model scores the clip against each, 0–100. Also changeable by an agent through the API, within the settings allowlist.
*   **`AI_TOPICS_THRESHOLD`**: Minimum score for a topic to count as "matched" (shown, searchable, and written to the XMP sidecar's `dc:subject`).
*   Result is written as `<recording>.ai.json` (shown in the dashboard) and `<recording>.mp4.xmp` (Immich-compatible sidecar — also carries `xmp:Rating` for the star rating and your personal note appended to `dc:description`; note Immich's XMP support for *video* files specifically is worth verifying against your own Immich version).
*   A manual re-analyze button is available per recording in the dashboard (requires `FILMSTRIP_COUNT` > 0 for that recording), also callable by an agent through the API.

## 7b. AUDIO TRIGGER (dashboard: Settings → Audio-Trigger)

*   **`AUDIO_TRIGGER_ENABLED`**: Master on/off switch. Runs in its own thread per camera regardless of state — can never block or delay recording.
*   **`AUDIO_TRIGGER_CATEGORIES`**: Free-text categories compared against live audio via CLAP (e.g. `glass breaking`, `dog barking`) — not a fixed label set, no retraining needed to add one. Live-editable, no restart.
*   Independent of `AI_TOPICS` (§7a) — this is what *starts* a recording from sound alone; topic classification happens afterward, on the finished clip.
*   Only applies to cameras with `audio_enabled: true` (§2).

## 7c. TRANSCRIPTION (dashboard: Settings → Transkription)

*   **`TRANSCRIPTION_ENABLED`**: Master on/off switch. Off by default.
*   **`WHISPER_MODEL_SIZE`**: `tiny` through `large-v3`. GPU-accelerated with automatic CPU fallback.
*   Separate from the audio trigger (§7b) — CLAP recognizes sound *categories*, this transcribes actual spoken words. Coordinated (not run in parallel) with description/topic analysis so they never write to the same metadata file at the same time.

## 7d. FACE RECOGNITION (dashboard: Settings → Gesichtserkennung)

*   **`FACE_RECOGNITION_ENABLED`**: Master on/off switch. Off by default.
*   **`FACE_MODEL_PACK`**: `buffalo_s`/`buffalo_m`/`buffalo_l`/`antelopev2` — see the comparison table in `README.md`.
*   **`FACE_MIN_CONFIDENCE`**: Minimum detection confidence for a face to be stored at all.
*   Detects/embeds on the same filmstrip frames the description stage already uses — no extra frame extraction.
*   Unmatched faces are grouped via DBSCAN (`cluster_faces.py`, runs automatically or on-demand from the dashboard's People section).
*   **Identity persists independently of any single video**: the moment a face is assigned to a named person, its photo is copied into the permanent `.people_photos/` archive and the recognition embedding is preserved even after every video that showed them is deleted. "Un-naming" a person (dashboard trash icon) un-assigns their faces for re-clustering without deleting anything; a separate, clearly-marked "delete permanently" action removes their recognition data and archived photo entirely — for people genuinely no longer needed in the system.

---

## 8. POSE ESTIMATION & BEHAVIOR DETECTION (dashboard: Settings → Pose Estimation / Fall Detection)

One shared pose model, loaded once at camera-process startup — only if at least one of the pose-based evaluations below is enabled. Runs only on frames where the main detector already found a `person`, so there's no cost on empty scenes. Loitering and movement don't need the pose model at all (they work from the ordinary detection boxes) and can be used even with the others off.

*   **`POSE_ESTIMATION_ENABLED`**: Enables fall detection specifically.
*   **`POSE_FALL_ANGLE_THRESHOLD`** (20–85°, default 55): Torso-angle-from-vertical threshold. Lower = more sensitive (more false positives from bending over). A fall must read above this threshold for several consecutive frames before it's reported — a brief bend doesn't count.
*   **`POSE_RAISED_HANDS_ENABLED`**: Both wrists held well above the shoulders — a possible wave or distress signal. One hand alone is treated as an ordinary gesture and ignored. Uses the same pose pass as fall detection, no extra cost if both are on.
*   **`POSE_LOITERING_ENABLED`**: A person staying in roughly the same spot for too long. Position-based, no pose model needed.
*   **`POSE_LOITERING_SECONDS`** (5–600, default 30): How long is "too long." Resets the moment the person leaves the frame or moves on.
*   **`POSE_MOVEMENT_ENABLED`**: Flags unusually fast movement (possibly running). Position-based. Speed is measured in body-heights-per-second, not raw pixels, so it reads consistently regardless of how close the person is to the camera — no per-camera calibration needed.
*   **`POSE_PROXIMITY_ENABLED`**: Flags two or more people staying close together for a sustained period. Position-based. A brief pass-by (e.g. on a sidewalk) doesn't trigger it.
*   **`POSE_GAZE_ENABLED`**: Logs whether a person is facing the camera or facing away, based on whether facial keypoints are visible at all. Informational only — does not start a recording or send a notification on its own.
*   **`POSE_POINTING_ENABLED`**: Logs a pointing gesture (one arm extended sideways at shoulder height, not raised). Also informational only.
*   **What actually starts a recording:** a confirmed fall, raised-hands, fast movement, or sustained proximity event all force a recording to start (or continue) even if the main detector doesn't currently see anything — the same way a manual trigger does. Gaze and pointing never do; they're logged and published (MQTT/webhook) but don't affect recording state.

---

## 9. ANOMALY DETECTION (dashboard: Settings → Anomaly Detection)

*   **`ANOMALY_DETECTION_ENABLED`**: Master on/off switch. Off by default. Reuses the semantic-search text embedding already computed for every event (§10 in README) — no new model, no extra GPU cost. Training itself (not just tagging) works regardless of this switch; it only gates whether a match actually flags a new recording.
*   Per-camera Isolation Forest, trained from the dashboard's Anomaly Detection card, via cron (`anomaly_detection.py --lookback-days 30`), or by an agent through the API.
*   Needs at least 15 analyzed recordings for that camera within the lookback window (default 30 days) — cameras with less history are skipped, not force-trained on too little data. The dashboard card shows exactly how many it currently has toward that threshold.
*   Flagged recordings get `anomaly: true` and a score written to their metadata; this also feeds the agent-webhook "anomaly only" filter (§12).

---

## 10. HOME ASSISTANT / MQTT (dashboard: Settings → Home Assistant / MQTT)

*   **`MQTT_ENABLED`**: Master on/off switch. Off by default.
*   **`MQTT_BROKER`**, **`MQTT_PORT`** (default 1883), **`MQTT_USERNAME`**, **`MQTT_PASSWORD`**: Standard broker connection details.
*   **`MQTT_TOPIC_PREFIX`** (default `vigil`): Topics publish under `<prefix>/<camera>/...` — see `HOME_ASSISTANT.md` for the full topic list, including per-camera `recording`, `last_event_summary`, `detection`, `fall_detected`, `raised_hands_detected`, `loitering_detected`, `running_detected`, and `proximity_detected`.
*   **`MQTT_HA_DISCOVERY`** (default on): Publishes Home Assistant MQTT Discovery config so entities appear automatically, no YAML needed.
*   Fire-and-forget by design: publishing runs on its own background thread and can never delay or affect recording, even if the broker is completely unreachable.

---

## 11. AGENT WEBHOOK (dashboard: Settings → Agent Webhook)

*   **`AGENT_WEBHOOK_URL`**: If set, vigil POSTs a JSON event to this URL after every analyzed recording. Empty by default (feature is off until a URL is provided).
*   **`AGENT_WEBHOOK_ANOMALY_ONLY`**: If enabled, only POSTs when the event was flagged by Anomaly Detection (§9) — useful if you only want an agent notified about the unusual cases, not every delivery. Requires Anomaly Detection to be enabled to have anything to report.
*   Same fire-and-forget guarantee as MQTT — an unreachable receiver never delays the pipeline.
*   See `AGENT_CONFIG.md` for the exact payload shape.

---

## 12. EXPORT (dashboard: Settings → Export)

*   **`EXPORT_DIR`**: Local path, or `user@host:/path` for remote export via `rsync` (assumes passwordless SSH is already set up).
*   **`EXPORT_INCLUDE_VIDEO`**, **`EXPORT_INCLUDE_METADATA`**, **`EXPORT_INCLUDE_LARGE_THUMBS`**, **`EXPORT_INCLUDE_SMALL_THUMBS`** (all default on): Exactly what gets bundled per export, remembered for next time.
*   **`EXPORT_DELETE_AFTER`** (default off): Deletes the original only after a confirmed successful export — never before, never on a partial failure.
*   Exporting several recordings at once can optionally share a named subfolder. Blocked automatically for a recording that's still in progress.

---

## 13. WATCHFOLDER IMPORT (dashboard: Settings → Watchfolder)

*   **`WATCH_FOLDER_ENABLED`**: Master on/off switch. Off by default.
*   **`WATCH_FOLDER_PATH`**: Folder to watch. Every finished video file dropped in there is treated as a completed recording once it stops growing (size-stability check).
*   **`WATCH_FOLDER_SOURCE_NAME`**: The "camera" name assigned to imports, for filenames/logs/grouping.
*   **`WATCH_FOLDER_STABILITY_SEC`** (default 5): How long a file's size must stay unchanged before it's considered finished.
*   **`WATCH_FOLDER_DELETE_SOURCE`**: Whether to remove the original file after import (copy vs. move).
*   **`WATCH_FOLDER_RUN_DETECTION`**: If enabled, runs YOLO once on the imported file and discards it if none of `DETECTION_CLASSES` are found.
*   **`WATCH_FOLDER_LIVE_MODE_ENABLED`**: If enabled, a newly-discovered growing file is probed once (`mp4_probe.py`) to check whether it's a streamable container (MPEG-TS, or "fast-start"/fragmented MP4 with the index written first). If so, it's read live — via a tailing FIFO (`live_tail.py`) feeding a real `CameraAgent` instance — through the same detection/recording pipeline as an actual camera, instead of waiting for it to finish. A classic MP4 (index written only at the end, the common case for most recording tools) can't be read this way at all; it falls back to the normal wait-for-completion behavior automatically, file by file. The live source is stopped automatically once its underlying file stops growing for 60s or disappears.

---

## 13b. PLATFORM SOURCES (YouTube, Twitch, Vimeo, and others)

A camera's `url` field (§2) accepts a YouTube/Twitch/Vimeo/Facebook/Dailymotion/Kick link directly, alongside the usual rtsp://, rtmp://, and /dev/videoX values — no separate setting or camera "type" needed.

*   Requires `yt-dlp` installed (`pip install yt-dlp`) and reachable via `ffmpeg` on the same machine.
*   A persistent background bridge process (`platform_bridge.py`) keeps a `yt-dlp | ffmpeg` pipeline feeding a local FIFO in an always-streamable format (MPEG-TS), restarting that pipeline on its own if the channel goes offline or the connection drops — the same tailing/FIFO mechanism as Watchfolder mode 1 (§13). vigil's own camera-connection logic just reads the local FIFO, so it's never affected by a platform's own time-limited/signed stream URLs expiring mid-recording.
*   No additional settings — the bridge starts automatically the moment a platform URL is detected for a camera, and is stopped/cleaned up when that camera is disabled or its process stops.
*   Codec handling matches live cameras: browser-incompatible codecs (HEVC being the common case) are transcoded automatically, GPU-accelerated where available; already-compatible files are only re-wrapped.
*   Runs as its own background process, independent of camera recording processes.

---

## 14. AGENT CONTROL & MCP SERVER

Not a `pipeline_settings.json` block — its own file, `agent_config.json`, at the project root. Covers whether an external agent (Hermes, Claude via the MCP server, or any other API caller) can operate the system beyond the plain job-submission API.

*   **`agent_control_enabled`**: Master switch. Off by default — nothing below does anything until this is `true`.
*   Per-capability toggles: `search`, `cameras_toggle`, `pipeline_control`, `manual_trigger` (covers trigger/stop/quick_record/notify_only/detections), `settings_change` (restricted to a fixed code-level allowlist regardless of this toggle), `delete` and `export` (listed as placeholders only — no route exists for either, so enabling the flag does nothing).
*   Manageable entirely from the dashboard's External API card — no hand-editing the JSON file required, though it's a plain, readable file if you want to.
*   `mcp_server.py` is a separate, optional process exposing the same gated API as 21 MCP tools instead of raw HTTP, for Claude Desktop or any other MCP client. See `MCP_SERVER.md` for setup, and `AGENT_CONFIG.md` for the full permission model, endpoint reference, and a quick-reference table of which tool to use for which task.

---

## 15. OPERATIONAL MODES

*   **`RECORDING_MODE = "EVENT_DRIVEN"`** (default mode for a normal camera)
    - **Workflow**: `IDLE` → *Detection Found* → `RECORDING` (Pre-roll + Event + Post-roll) → `IDLE`.
    - Efficient by design — only consumes disk/CPU/GPU encode time while something is actually happening.
*   **Notify-only mode** (per camera, §2's `notify_only` field): detection still runs, but never auto-starts a recording — only reports what it saw. Recording then requires an explicit external trigger (dashboard, cron, or an agent through the API).
*   **Quick Record** (`quick_record.py`, agent API `/quick_record`): a third, independent mode — bypasses detection and the state machine entirely, connects directly to a camera and records a fixed duration on demand, regardless of whether the main pipeline is even running or that camera is enabled anywhere else.

---

## 16. LOGGING & MONITORING

Several distinct log files exist:

1.  **`logs/system_main.log`**: The `system_logger` — orchestrator lifecycle, GPU detection at startup, global errors, and the periodic reconciliation of camera enabled/disabled state (§2).
2.  **`logs/{camera_name}.log`**: Per-camera detail logs — connection state, detection events, encoding, and (if enabled) pose/behavior detection results.
3.  **`logs/pipeline_runtime.log`**: Combined stdout/stderr of the recording pipeline process when started via `start_detached.sh` — this is what the dashboard's built-in **Log** panel displays (last 100 lines, auto-refreshing while open), and where the optional AI-analysis subprocess's own output lands too.
4.  **`logs/watchdog.log`**: Written only if you've set up `watchdog.sh` via cron — records restart attempts of the dashboard.

For quick health checks without opening a log file at all: the dashboard's Hardware & System Status panel shows CPU/RAM/VRAM/GPU temp/disk/NVENC/NVDEC live (also available to an agent via `/system_status`), and `/health` is an unauthenticated endpoint suitable for external monitoring.

---

*Manual updated to reflect the current dashboard-first configuration model, including pose/behavior detection, anomaly detection, MQTT, agent control, and the MCP server. Partly generated with AI assistance, consistent with the rest of this project.*
