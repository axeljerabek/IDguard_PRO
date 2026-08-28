#!/usr/bin/env python3
"""
ai_analyze.py - Analysiert die 'large' Filmstrip-Frames eines fertigen
Recordings per Ollama-Vision-Modell und schreibt das Ergebnis als:
  1) <video>.ai.json  - fürs Dashboard (Recent/Archive-Anzeige)
  2) <video>.mp4.xmp  - Immich-kompatibles XMP-Sidecar (dc:description)

Aufruf (von recorder_pipeline.py per subprocess.Popen, fire-and-forget,
blockiert also NICHT den Detection-Loop):
    python3 ai_analyze.py <video_basename> <base_dir>
"""
import sys
import os
import json
import glob
import base64
import time
import urllib.request

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
try:
    from config import SETTINGS_F
except ImportError:
    SETTINGS_F = "pipeline_settings.json"


def _load_settings():
    try:
        with open(SETTINGS_F) as f:
            return json.load(f)
    except Exception:
        return {}


_settings = _load_settings()
OLLAMA_URL = os.environ.get("OLLAMA_URL", _settings.get("OLLAMA_URL", "http://localhost:11434"))
OLLAMA_MODEL = os.environ.get("OLLAMA_VISION_MODEL", _settings.get("OLLAMA_VISION_MODEL", "llava:latest"))
MAX_FRAMES = int(os.environ.get("AI_ANALYZE_MAX_FRAMES", _settings.get("AI_ANALYZE_MAX_FRAMES", 12)))

PROMPT = (
    "Das sind aufeinanderfolgende Standbilder aus einer Sicherheitskamera-"
    "Aufnahme, in zeitlicher Reihenfolge. Beschreibe in 1-2 knappen Sätzen "
    "auf Deutsch, was passiert: wer/was zu sehen ist und was diese Person "
    "oder dieses Objekt tut. Nur was tatsächlich sichtbar ist, keine "
    "Spekulation."
)


def _xml_escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&apos;"))


def _pick_frames(frame_dir, max_frames):
    files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if len(files) > max_frames:
        step = len(files) / max_frames
        files = [files[int(i * step)] for i in range(max_frames)]
    return files


def analyze(video_basename, base_dir):
    pending_path = os.path.join(base_dir, f"{video_basename}.ai.pending")
    try:
        open(pending_path, 'w').close()
    except Exception:
        pass
    try:
        _analyze_inner(video_basename, base_dir)
    finally:
        if os.path.exists(pending_path):
            try:
                os.remove(pending_path)
            except Exception:
                pass


def _analyze_inner(video_basename, base_dir):
    frame_dir = os.path.join(base_dir, ".thumbs", video_basename, "large")
    if not os.path.isdir(frame_dir):
        print(f"ℹ️ Kein Filmstrip für {video_basename} vorhanden, überspringe AI-Analyse.")
        return

    files = _pick_frames(frame_dir, MAX_FRAMES)
    if not files:
        return

    images_b64 = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                images_b64.append(base64.b64encode(fh.read()).decode("utf-8"))
        except Exception:
            pass
    if not images_b64:
        return

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": PROMPT,
        "images": images_b64,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        description = result.get("response", "").strip()
    except Exception as e:
        print(f"❌ Ollama-Analyse fehlgeschlagen für {video_basename}: {e}")
        return

    if not description:
        print(f"⚠️ Ollama lieferte keine Beschreibung für {video_basename}.")
        return

    # 1) Eigene JSON-Metadatei — vom Dashboard gelesen
    meta_path = os.path.join(base_dir, f"{video_basename}.ai.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "description": description,
                "model": OLLAMA_MODEL,
                "frame_count": len(images_b64),
                "ts": time.time()
            }, f)
    except Exception as e:
        print(f"❌ Konnte {meta_path} nicht schreiben: {e}")

    # 2) XMP-Sidecar für Immich (dc:description)
    xmp_path = os.path.join(base_dir, f"{video_basename}.mp4.xmp")
    xmp = f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:description>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{_xml_escape(description)}</rdf:li>
    </rdf:Alt>
   </dc:description>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    try:
        with open(xmp_path, "w", encoding="utf-8") as f:
            f.write(xmp)
    except Exception as e:
        print(f"❌ Konnte {xmp_path} nicht schreiben: {e}")

    print(f"✅ AI-Analyse fertig für {video_basename}: {description}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ai_analyze.py <video_basename> <base_dir>")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2])
