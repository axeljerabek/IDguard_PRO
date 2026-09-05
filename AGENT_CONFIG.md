# Agent Control

Lets an AI agent (Hermes, OpenClaw, or anything else calling the same API) operate IDguard PRO directly — toggle cameras, tune settings, start/stop the pipeline, and search — instead of only submitting jobs. Built on top of the [External API](./REMOTE_API.md); uses the same API keys and the same `/api/v1/` base.

**Off by default.** Nothing in this section does anything until you explicitly turn it on in `agent_config.json`.

## The permission model

`agent_config.json` has one master switch and a per-capability toggle:

```json
{
  "agent_control_enabled": false,
  "capabilities": {
    "search": { "enabled": true },
    "cameras_toggle": { "enabled": true },
    "pipeline_control": { "enabled": true },
    "settings_change": { "enabled": true },
    "delete": { "enabled": false },
    "export": { "enabled": false }
  }
}
```

Both the master switch **and** the specific capability must be `true` for a call to succeed — turning the master switch on doesn't retroactively grant every capability, each stays off unless you also flip it. Editing the file takes effect immediately, no restart needed (read fresh on every request).

| Capability | Risk | What it allows |
| :--- | :--- | :--- |
| `search` | Low | Read-only. Search recordings by description, topic, transcript, person. |
| `cameras_toggle` | Low | Enable/disable individual cameras. Never touches URLs or credentials. |
| `pipeline_control` | Medium | Start/stop the whole recording pipeline. While stopped, nothing records. |
| `settings_change` | Medium | Change a fixed allowlist of tuning settings (see below). Credentials, URLs, and export destinations are never reachable through this capability — enforced in code, not just by convention. |
| `delete` | High | **Not implemented.** No route exists for it. Listed here as a placeholder for the decision, not a working toggle. |
| `export` | Medium | **Not implemented.** Same as above. |

## Settings allowlist

`settings_change` can only touch: `CONFIDENCE_THRESHOLD`, `TARGET_FPS`, `PRE_ROLL_SEC`, `POST_ROLL_SEC`, `DETECTION_CLASSES`, `AI_TOPICS`, `AI_TOPICS_THRESHOLD`, `AI_TOPICS_ENABLED`, `AI_ANALYZE_MAX_FRAMES`, `ANOMALY_DETECTION_ENABLED`, `FACE_MIN_CONFIDENCE`.

Anything outside this list — MQTT credentials, camera URLs, export paths, watchfolder paths — is rejected with a 403, even if `settings_change` is enabled. A request that mixes an allowed and a disallowed key is rejected entirely; nothing partially applies.

## Endpoints

All under `/api/v1/agent/`, same `Authorization: Bearer <key>` / `X-API-Key` auth as the rest of the External API.

| Method & path | Capability | Notes |
| :--- | :--- | :--- |
| `GET /cameras` | `cameras_toggle` | Name, enabled, audio_enabled — URL never included. |
| `POST /cameras/<name>/enable` | `cameras_toggle` | |
| `POST /cameras/<name>/disable` | `cameras_toggle` | |
| `GET /settings` | `settings_change` | Only allowlisted keys are returned, even reading. |
| `POST /settings` | `settings_change` | JSON body of key/value pairs, allowlist enforced. |
| `GET /pipeline/status` | `pipeline_control` | |
| `POST /pipeline/start` | `pipeline_control` | |
| `POST /pipeline/stop` | `pipeline_control` | |
| `GET /search?q=...` | `search` | Same underlying search as the dashboard. |

## Why delete and export aren't here yet

Deliberate. Both are meaningfully higher-stakes than the rest (delete is irreversible; export can move data off the box). They're listed in the config as a placeholder for the decision, but no route implements them — enabling the flag today does nothing, by design. If/when they're built, they'll ship with the same allowlist-and-explicit-testing approach as everything else here, not bolted on quickly.

## Example (curl)

```bash
# Check what's currently allowed to run
curl https://your-idguard-host:19473/api/v1/agent/pipeline/status \
  -H "Authorization: Bearer idg_xxxxxxxxxxxx"

# Disable a camera
curl -X POST https://your-idguard-host:19473/api/v1/agent/cameras/Backyard/disable \
  -H "Authorization: Bearer idg_xxxxxxxxxxxx"

# Search
curl "https://your-idguard-host:19473/api/v1/agent/search?q=delivery" \
  -H "Authorization: Bearer idg_xxxxxxxxxxxx"
```
