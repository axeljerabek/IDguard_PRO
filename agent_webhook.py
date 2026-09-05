"""
agent_webhook.py — proaktive Benachrichtigung eines Agenten (Hermes o.ä.)
bei neuen Events, statt dass er aktiv pollen muss ("Autonomous Watchdog"-
Idee). Bewusst nach demselben Muster wie mqtt_client.py: fire-and-forget,
läuft in einem eigenen Hintergrund-Thread, kann die Aufnahme-/Analyse-
Pipeline unter keinen Umständen verzögern oder blockieren.

Off by default -- ohne konfigurierte AGENT_WEBHOOK_URL passiert schlicht
nichts, kein Fehler, kein Log-Spam.
"""
import json
import time
import threading
import urllib.request
import urllib.error

try:
    from config import SETTINGS_F
except ImportError:
    SETTINGS_F = "pipeline_settings.json"

_settings_cache = {}
_settings_cache_time = 0.0
_SETTINGS_CACHE_TTL = 5.0


def _get_settings():
    global _settings_cache, _settings_cache_time
    now = time.time()
    if now - _settings_cache_time > _SETTINGS_CACHE_TTL:
        try:
            with open(SETTINGS_F) as f:
                _settings_cache = json.load(f)
        except Exception:
            _settings_cache = {}
        _settings_cache_time = now
    return _settings_cache


def _post_worker(url, payload):
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"⚠️ [AgentWebhook] Zustellung an {url} fehlgeschlagen (Agent erreichbar?): {e}")


def notify_event(camera, filename, description, topics, anomaly, anomaly_score=None):
    """Wird nach abgeschlossener KI-Analyse aufgerufen (aus postprocess.py).
    Respektiert AGENT_WEBHOOK_ANOMALY_ONLY -- falls gesetzt, wird nur bei
    tatsächlich erkannter Anomalie benachrichtigt, nicht bei jedem
    x-beliebigen Event (sonst würde ein Agent bei viel Kamera-Traffic in
    einer Flut von Benachrichtigungen ertrinken)."""
    settings = _get_settings()
    url = (settings.get("AGENT_WEBHOOK_URL") or "").strip()
    if not url:
        return
    if settings.get("AGENT_WEBHOOK_ANOMALY_ONLY", False) and not anomaly:
        return
    payload = {
        "event": "anomaly" if anomaly else "recording_analyzed",
        "camera": camera,
        "filename": filename,
        "description": description,
        "topics": topics or {},
        "anomaly": bool(anomaly),
        "anomaly_score": anomaly_score,
        "timestamp": time.time(),
    }
    threading.Thread(target=_post_worker, args=(url, payload), daemon=True).start()
