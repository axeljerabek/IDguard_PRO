"""
agent_permissions.py — permission checks for /api/v1/agent/* routes.

Separate from the plain job-submission API (mam_api.py's /api/v1/jobs),
which has always been reachable by any valid API key. Agent-control
routes additionally require agent_config.json to have the master switch
on AND the specific capability enabled.
"""
import os
import json

DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(DIR, "agent_config.json")

DEFAULT_CONFIG = {
    "agent_control_enabled": False,
    "capabilities": {
        "search": {"enabled": True},
        "cameras_toggle": {"enabled": True},
        "pipeline_control": {"enabled": True},
        "settings_change": {"enabled": True},
        "delete": {"enabled": False},
        "export": {"enabled": False},
    },
}


def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return DEFAULT_CONFIG


def is_capability_allowed(capability):
    """True only if the master switch is on AND this specific capability
    is enabled. Missing/unknown capability names are treated as not
    allowed (fail closed, not fail open)."""
    config = load_config()
    if not config.get("agent_control_enabled", False):
        return False
    return bool(config.get("capabilities", {}).get(capability, {}).get("enabled", False))


# Settings keys the "settings_change" capability may touch, regardless of
# whether it's enabled -- credentials, URLs, and destinations are never
# reachable through this capability, full stop.
SETTINGS_ALLOWLIST = {
    "CONFIDENCE_THRESHOLD", "TARGET_FPS", "PRE_ROLL_SEC", "POST_ROLL_SEC",
    "DETECTION_CLASSES", "AI_TOPICS", "AI_TOPICS_THRESHOLD",
    "AI_TOPICS_ENABLED", "AI_ANALYZE_MAX_FRAMES", "ANOMALY_DETECTION_ENABLED",
    "FACE_MIN_CONFIDENCE",
}
