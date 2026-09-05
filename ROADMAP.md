# vigil — Roadmap

Ongoing, prioritized list. Worked through incrementally, not all at once.

---

## Low priority / opportunistic

- [ ] *(deferred)* **VAE/autoencoder on raw frames** for anomaly detection — only worth pursuing if the Isolation Forest approach hits real limits. Would be the first custom-trained model in the system, needs a GPU training pipeline, meaningfully more effort.

## Open from earlier sessions

- [ ] Watchfolder mode 1 (read a growing file as a live stream) — technically uncertain for MP4 sources (moov atom placement); only worth evaluating against a real sample file
- [ ] Memory-usage safeguard for encode-mode cameras (MJPEG/USB) — ~1.7GB/camera at 1080p/10s pre-roll; flagged, no decision made yet
- [ ] MJPEG/USB camera encoding path — built and unit-tested, not yet verified against real hardware

---

## Done

**Rename:** IDguard PRO → vigil, across the codebase, GUI, docs, Docker setup, and architecture diagram.

**Agent Control:** camera toggle, settings (allowlisted), pipeline start/stop, search, manual trigger/stop, quick-record (pipeline-independent ad-hoc recording), event details, summaries, system status, reanalyze, anomaly training. Per-capability permission config, master switch, off by default. Proactive notifications (agent webhook) after each analyzed event, with an anomaly-only filter. Delete and export intentionally excluded. Fixed a real bug where enabling/disabling a camera through the API never actually started/stopped its process — the master pipeline now reconciles against `streams.json` periodically instead of only reading it at startup.

**MCP Server:** 21 tools wrapping the full Agent Control API for Claude Desktop and other MCP clients, tested against the real MCP protocol (not just the underlying functions). `MCP_SERVER.md`.

**Pose Estimation & Behavior Detection** (turned out to be much more than just fall detection): fall detection (torso-angle heuristic, temporal confirmation), raised-hands distress signal, loitering, fast-movement/running detection, sustained close-proximity detection, and head-orientation/gaze logging — six independent, individually-toggleable signals from one shared pose model at zero extra GPU cost. Fall/distress/movement/proximity all force a recording; gaze and pointing are informational only.

**Face recognition — critical persistence fix:** named people were silently losing their recognition embedding once every video that ever showed them was deleted (the centroid got wiped to `NULL`). Fixed by archiving a person's face photo permanently the moment they're identified, and by no longer deleting a named person's face rows/embeddings when their source video is removed — only truly unassigned/cluster faces get cleaned up. Also added: user-chosen profile photo per person, a genuine "delete permanently" action (distinct from the existing soft "un-name"), and the ability to name/merge a *selected subset* of faces within a mixed cluster instead of always the whole group.

**Docs overhaul:** README rewritten twice (structure, then again leading with a hook and differentiators before features/architecture); `MANUAL.md` fully brought current (pose/behavior, anomaly detection, MQTT, agent webhook, export, watchfolder, MCP — grew from 9 to 20 sections); `AGENT_CONFIG.md` gained a quick-reference "right tool for the job" table after a real mix-up between `trigger`/`quick_record`/`stop`/`disable`; `config.py.example` fixed (stale "IDENTITY-GUARD PRO" header, a duplicated `BROWSER_COMPATIBLE_VIDEO_CODECS` line, a German leftover in `COCO_CLASS_NAMES`); a full technical blog post written.

**Earlier sessions:** RTMP+RTSP, MJPEG/USB encode path, Watchfolder import (mode 2), HEVC re-transcode (GPU-accelerated), process orchestration fixes (restart detection, systemd service, worker naming), post-process watchdog (a stuck Ollama/GPU no longer blocks the whole queue), per-video notes (XMP export), shared export subfolders, export content checkboxes (video/metadata/thumbs), delete-after-export option, star rating (xmp:Rating), daily/weekly LLM summaries (dashboard card + cron example, plus regenerate/delete), Home Assistant/MQTT integration (HA discovery, `HOME_ASSISTANT.md`), Isolation Forest anomaly detection (dashboard card + cron example), README logo, External API for remote control (job submission, API-key auth, status polling, webhook callbacks with retry, video/segment export, `REMOTE_API.md`), UI style cleanup (reduced padding/radius, more screen space for media).
