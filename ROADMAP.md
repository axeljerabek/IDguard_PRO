# IDguard PRO — Roadmap

Ongoing, prioritized list. Worked through incrementally, not all at once.

---

## Tier 4 — Agent Control (builds on the External API)

- [ ] Agent can call the existing External API to toggle cameras, change settings, start/stop the pipeline, and search — through the same endpoints as any other API client
- [ ] Per-capability permission config (see AGENT_CONFIG.md) — off by default, enabled selectively
- [ ] Delete and export intentionally excluded from agent capabilities for now
- [ ] Longer-term: MCP server wrapper around the API for direct agent access

## Low priority / opportunistic

- [ ] **YOLO Pose Estimation** — fall detection, posture classification. Fits the architecture cleanly (YOLO already runs), but self-rated low priority.
- [ ] *(deferred)* **VAE/autoencoder on raw frames** for anomaly detection — only worth pursuing if the Isolation Forest approach hits real limits. Would be the first custom-trained model in the system, needs a GPU training pipeline, meaningfully more effort.

## Open from earlier sessions

- [ ] Watchfolder mode 1 (read a growing file as a live stream) — technically uncertain for MP4 sources (moov atom placement); only worth evaluating against a real sample file
- [ ] Memory-usage safeguard for encode-mode cameras (MJPEG/USB) — ~1.7GB/camera at 1080p/10s pre-roll; flagged, no decision made yet
- [ ] MJPEG/USB camera encoding path — built and unit-tested, not yet verified against real hardware

---

## Done

RTMP+RTSP, MJPEG/USB encode path, Watchfolder import (mode 2), HEVC re-transcode (GPU-accelerated), process orchestration fixes (restart detection, systemd service, worker naming), post-process watchdog (a stuck Ollama/GPU no longer blocks the whole queue), per-video notes (XMP export), shared export subfolders, export content checkboxes (video/metadata/thumbs), delete-after-export option, star rating (xmp:Rating), daily/weekly LLM summaries (dashboard card + cron example), Home Assistant/MQTT integration (HA discovery, HOME_ASSISTANT.md), Isolation Forest anomaly detection (dashboard card + cron example), README logo, External API for remote control (job submission, API-key auth, status polling, webhook callbacks with retry, video/segment export, REMOTE_API.md), UI style cleanup (reduced padding/radius, more screen space for media).
