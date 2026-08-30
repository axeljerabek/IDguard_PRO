#!/usr/bin/env python3
"""
transcribe_audio.py - Extrahiert die Audiospur einer fertigen Aufnahme per
ffmpeg und transkribiert sie per faster-whisper (lokal, GPU-fähig mit
automatischem CPU-Fallback). Ergebnis landet im selben <video>.ai.json wie
die Vision-Analyse (gemergt, nicht überschrieben) und im Suchindex.

WICHTIG: läuft NIE parallel zu ai_analyze.py für dasselbe Video — beide
werden von postprocess.py sequenziell aufgerufen, genau um das Race beim
gemeinsamen Lesen/Schreiben von <video>.ai.json zu vermeiden (sonst könnte
je nachdem, wer zuletzt schreibt, die Beschreibung ODER das Transkript
verloren gehen).

Aufruf: python3 transcribe_audio.py <video_basename> <base_dir>
"""
import os
import sys
import json
import subprocess
import tempfile

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
try:
    from config import SETTINGS_F
except ImportError:
    SETTINGS_F = "pipeline_settings.json"
try:
    import search_index
except ImportError:
    search_index = None  # Optionales Feature — Transkription läuft unverändert ohne Suche


def _load_settings():
    try:
        with open(SETTINGS_F) as f:
            return json.load(f)
    except Exception:
        return {}


_settings = _load_settings()
TRANSCRIPTION_ENABLED = bool(_settings.get("TRANSCRIPTION_ENABLED", False))
WHISPER_MODEL_SIZE = _settings.get("WHISPER_MODEL_SIZE", "small")
TRANSCRIPTION_LANGUAGE = (_settings.get("TRANSCRIPTION_LANGUAGE", "") or "").strip() or None  # None = Whisper erkennt automatisch

_model = None
_model_load_failed = False


def _get_model():
    global _model, _model_load_failed
    if _model is not None:
        return _model
    if _model_load_failed:
        return None
    try:
        from faster_whisper import WhisperModel
        # GPU zuerst versuchen, sauberer Fallback auf CPU falls CUDA/cuDNN im
        # jeweiligen Setup nicht nutzbar ist — kein hartes Abhängigkeits-Muss.
        try:
            _model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
        except Exception:
            _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        return _model
    except Exception as e:
        print(f"⚠️ Whisper-Modell konnte nicht geladen werden (pip install faster-whisper nötig?) — Transkription bleibt inaktiv: {e}")
        _model_load_failed = True
        return None


def transcribe(video_basename, base_dir):
    if not TRANSCRIPTION_ENABLED:
        return
    video_path = os.path.join(base_dir, f"{video_basename}.mp4")
    if not os.path.exists(video_path):
        return

    model = _get_model()
    if model is None:
        return

    wav_path = None
    text = ""
    detected_language = None
    try:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120
        )
        if result.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            print(f"ℹ️ Keine Audiospur in {video_basename} gefunden — überspringe Transkription.")
            return

        segments, info = model.transcribe(wav_path, language=TRANSCRIPTION_LANGUAGE, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected_language = getattr(info, "language", TRANSCRIPTION_LANGUAGE)
    except subprocess.TimeoutExpired:
        print(f"⚠️ ffmpeg-Audioextraktion für {video_basename} hat das Timeout (120s) überschritten.")
        return
    except Exception as e:
        print(f"❌ Transkription für {video_basename} fehlgeschlagen: {e}")
        return
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

    if not text:
        print(f"ℹ️ Keine gesprochene Sprache in {video_basename} erkannt.")
        return

    # In dieselbe .ai.json wie ai_analyze.py schreiben — lesen, mergen,
    # schreiben. NIE blind überschreiben, sonst geht eine evtl. schon
    # vorhandene Vision-Beschreibung/Themen-Liste verloren.
    meta_path = os.path.join(base_dir, f"{video_basename}.ai.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    meta["transcript"] = text
    meta["transcript_language"] = detected_language
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except Exception as e:
        print(f"❌ Konnte {meta_path} nicht schreiben: {e}")

    if search_index is not None:
        try:
            search_index.index_event(f"{video_basename}.mp4", base_dir, transcript=text)
        except Exception as e:
            print(f"⚠️ Suchindex-Update (Transkript) fehlgeschlagen für {video_basename}: {e}")

    print(f"✅ Transkription fertig für {video_basename} ({detected_language}): {text[:120]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: transcribe_audio.py <video_basename> <base_dir>")
        sys.exit(1)
    transcribe(sys.argv[1], sys.argv[2])
