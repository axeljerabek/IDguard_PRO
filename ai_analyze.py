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
import urllib.error

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
try:
    from config import SETTINGS_F, DETECTION_CLASSES, COCO_CLASS_NAMES
except ImportError:
    SETTINGS_F = "pipeline_settings.json"
    DETECTION_CLASSES = [0]
    COCO_CLASS_NAMES = {0: "Person"}
try:
    import search_index
except ImportError:
    search_index = None  # Optionales Feature — Analyse läuft unverändert ohne Suche


def _load_settings():
    try:
        with open(SETTINGS_F) as f:
            return json.load(f)
    except Exception:
        return {}


_settings = _load_settings()
OLLAMA_URL = os.environ.get("OLLAMA_URL", _settings.get("OLLAMA_URL", "http://localhost:11434"))
OLLAMA_MODEL = os.environ.get("OLLAMA_VISION_MODEL", _settings.get("OLLAMA_VISION_MODEL", "llava:latest"))


def _describe_ollama_error_detailed(e):
    """Liest den HTTPError-Body GENAU EINMAL (ein Response-Body-Stream lässt
    sich nur einmal lesen) und liefert (ist_kontext_overflow, lesbare_
    Fehlerbeschreibung) zurück.

    WICHTIG: is_overflow kommt IMMER aus einer simplen Text-Muster-Prüfung
    auf dem vollständigen Rohtext — das ist die einzige Wahrheitsquelle.
    Bei Axel zeigte sich, dass Ollama (bzw. ein Proxy/Gateway davor) die
    Fehlerantwort in eine WECHSELNDE Anzahl von {"error": "..."}-Schichten
    verpackt (mal zwei, mal drei) — jeder Versuch, eine FESTE Verschachte-
    lungstiefe anzunehmen, ging früher oder später wieder schief und
    überschrieb die eigentlich korrekte Text-Erkennung mit einem falschen
    False. Das strukturierte JSON-Parsen unten (mit einer Schleife statt
    fester Tiefe) dient jetzt NUR NOCH dazu, eine schönere Detail-Nachricht
    zu extrahieren, kann is_overflow aber nie mehr beeinflussen."""
    if not isinstance(e, urllib.error.HTTPError):
        return False, str(e)
    try:
        raw_text = e.read().decode("utf-8")
    except Exception:
        return False, str(e)
    if not raw_text:
        return False, str(e)

    # Einzige Wahrheitsquelle für is_overflow — unabhängig von der
    # Verschachtelungstiefe der JSON-Struktur.
    is_overflow = (
        "exceed_context_size_error" in raw_text
        or "exceeds the available context size" in raw_text
    )

    # Best-effort: sich durch beliebig viele {"error": "<JSON-String>"}-
    # Schichten hindurcharbeiten, um eine kurze, lesbare "message" zu
    # finden — rein kosmetisch, verändert is_overflow nicht mehr.
    detail = raw_text
    current = raw_text
    for _ in range(5):  # großzügige Obergrenze gegen kaputte/endlose Verschachtelung
        try:
            parsed = json.loads(current)
        except Exception:
            break
        if isinstance(parsed, dict) and "message" in parsed and isinstance(parsed.get("message"), str):
            detail = parsed["message"]
            break
        err = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(err, dict):
            if isinstance(err.get("message"), str):
                detail = err["message"]
            else:
                detail = str(err)
            break
        elif isinstance(err, str) and err.strip().startswith("{"):
            current = err  # eine Schicht tiefer, nächste Runde
            continue
        else:
            break

    return is_overflow, f"{e} — {detail}"




def _describe_ollama_error(e):
    """Abwärtskompatibler Wrapper für Aufrufer, die nur den Text brauchen,
    nicht das Kontext-Overflow-Flag."""
    return _describe_ollama_error_detailed(e)[1]


MAX_FRAMES = int(os.environ.get("AI_ANALYZE_MAX_FRAMES", _settings.get("AI_ANALYZE_MAX_FRAMES", 12)))
# Ollamas Server-Standard für die Kontextgröße ist oft künstlich klein (klassisch
# 4096) — viele Modelle vertragen deutlich mehr, wenn man's explizit per Request
# anfragt ("num_ctx"). Bei wenig VRAM (z.B. NUC mit nur 6GB) ist ein größerer
# Wert aber ein echter Hardware-Trade-off (mehr Kontext = mehr GPU-Speicher fürs
# KV-Cache) — deshalb einstellbar statt fix hochgesetzt. 0 = Ollamas eigenen
# Standard unangetastet lassen (kein num_ctx im Request).
OLLAMA_CONTEXT_SIZE = int(os.environ.get("OLLAMA_CONTEXT_SIZE", _settings.get("OLLAMA_CONTEXT_SIZE", 0)))
AI_TOPICS_ENABLED = bool(_settings.get("AI_TOPICS_ENABLED", False))
AI_TOPICS = [t.strip() for t in _settings.get("AI_TOPICS", []) if isinstance(t, str) and t.strip()]
AI_TOPICS_THRESHOLD = float(_settings.get("AI_TOPICS_THRESHOLD", 50))

def _build_prompt():
    """Baut den Beschreibungs-Prompt dynamisch, statt eines starren Texts —
    nennt explizit, auf welche Objektklassen das YOLO-Modell gerade
    eingestellt ist (z.B. nur "Person"), und bittet gezielt um mehr Detail
    GENAU dazu, statt dass das Vision-Modell nur allgemein den Raum
    beschreibt und die eigentlich überwachte Objektart nur beiläufig
    erwähnt. Konfigurierte Topics fließen ebenfalls als Aufmerksamkeits-
    punkte mit ein, falls vorhanden — 'bessere KI-Verknüpfung' zwischen den
    drei Signalen (YOLO-Klassen, Vision-Beschreibung, Topics), statt dass
    sie unabhängig voneinander laufen."""
    watched = [COCO_CLASS_NAMES.get(c, str(c)) for c in DETECTION_CLASSES]
    watched_str = ", ".join(watched) if watched else "Personen"

    focus_line = (
        f"Diese Kamera überwacht gezielt: {watched_str}. Leg den Schwerpunkt "
        f"der Beschreibung klar darauf — was genau tut {'die erkannte Person/das erkannte Objekt' if len(watched) == 1 else 'das erkannte Objekt'} "
        "(Bewegungsrichtung, Handlung, mitgeführte Gegenstände, ungefähres "
        "Aussehen/Kleidung falls erkennbar), nicht nur der Raum drumherum. "
        "Den Raum/Kontext trotzdem kurz mit einordnen, aber das erkannte "
        "Objekt bleibt der Kern der Beschreibung."
    )

    topics_line = ""
    if AI_TOPICS_ENABLED and AI_TOPICS:
        topics_str = ", ".join(f'"{t}"' for t in AI_TOPICS)
        topics_line = (
            f" Falls die Szene zu einer dieser Kategorien passt, erwähne das "
            f"explizit in der Beschreibung: {topics_str}."
        )

    return (
        "Das sind aufeinanderfolgende Standbilder aus einer Sicherheitskamera-"
        "Aufnahme, in zeitlicher Reihenfolge. Beschreibe auf Deutsch in 3-5 Sätzen "
        "detailliert, was passiert. " + focus_line + topics_line +
        " Wie sich die Szene über die Bildfolge hinweg verändert (Ankunft, "
        "Handlung, Abgang) gehört ebenfalls dazu. Nur was tatsächlich sichtbar "
        "ist, keine Spekulation über Absichten oder Identität."
    )


PROMPT = _build_prompt()


def _xml_escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&apos;"))


def _pick_frames(frame_dir, max_frames):
    files = glob.glob(os.path.join(frame_dir, "*.jpg"))
    # Reservoir Sampling (recorder_pipeline.py) vergibt Slot-Nummern nicht mehr
    # zeitlich geordnet — timestamps.json (im übergeordneten Ordner) verrät die
    # echte Chronologie. Fehlt sie (alte Aufnahmen von vor diesem Fix), einfach
    # nach Dateiname sortieren wie bisher.
    ts_path = os.path.join(os.path.dirname(frame_dir), "timestamps.json")
    if os.path.exists(ts_path):
        try:
            with open(ts_path) as tf:
                ts_map = json.load(tf)
            files.sort(key=lambda p: ts_map.get(os.path.splitext(os.path.basename(p))[0].lstrip('0') or '0', 0))
        except Exception:
            files.sort()
    else:
        files.sort()

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


def _classify_topics(images_b64, topics):
    """Fragt das Vision-Modell separat, wie gut die Szene zu den in den
    Settings konfigurierten Themen passt (z.B. "break-in", "accident",
    "mail carrier").

    WICHTIG, ehrlich gesagt: das ist die Selbsteinschätzung des Sprachmodells
    per Prompt, KEINE kalibrierte Wahrscheinlichkeit wie bei YOLO (trainiert
    auf gelabelten Bounding-Boxes) oder CLAP (Cosine-Ähnlichkeit in einem
    Embedding-Raum). Ein LLM, das "80" antwortet, hat das nicht gegen echte
    Trainingsdaten kalibriert — es ist einfach die plausibelste Zahl, die das
    Modell dafür hält. Als grobe Sortierung/Filterung taugt das trotzdem gut.
    """
    if not topics:
        return {}
    topic_list = ", ".join(f'"{t}"' for t in topics)
    prompt = (
        "Look at these images again, from the same security camera recording. "
        f"For each of these categories: {topic_list} — "
        "give a number from 0 to 100 for how well the scene matches that "
        "category, based only on what is visible. Respond with ONLY a JSON "
        "object mapping each category name exactly as given to its number, "
        "nothing else, no explanation, no markdown."
    )
    # Eigene, leichte Overflow-Behandlung: der Themen-Prompt kommt ZUSÄTZLICH
    # zu den Bildern der schon erfolgreichen Beschreibung obendrauf — bei
    # vielen Themen kann das allein reichen, um erneut über die
    # Kontextgrenze zu rutschen, auch wenn die Beschreibung selbst gerade
    # noch durchgegangen ist.
    images = list(images_b64)
    parsed = None
    while images:
        payload_dict = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": images,
            "format": "json",
            "stream": False
        }
        if OLLAMA_CONTEXT_SIZE > 0:
            payload_dict["options"] = {"num_ctx": OLLAMA_CONTEXT_SIZE}
        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            raw = result.get("response", "").strip()
            if not raw:
                # Leere Antwort bei "format": "json" — in der Praxis oft ein
                # WEITERES Symptom desselben Kontext-Overflows (das Modell
                # bricht die strukturierte Ausgabe einfach ab), auch wenn
                # Ollama hier KEINEN sauberen 400er mit Fehlerdetails liefert
                # wie beim Haupt-Beschreibungs-Request. Genau der Fall, der
                # tatsächlich beobachtet wurde ("Expecting value: line 1
                # column 1"). Bewusst als Overflow behandelt (weniger Bilder,
                # erneut versuchen), da eine leere Response hier praktisch
                # nie etwas anderes bedeutet.
                raise ValueError("empty response from Ollama (likely context overflow)")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"Erwartete ein JSON-Objekt, bekam: {type(parsed)}")
            break
        except Exception as e:
            is_overflow, detail = _describe_ollama_error_detailed(e)
            if not is_overflow and isinstance(e, ValueError) and "context overflow" in str(e):
                is_overflow = True
                detail = str(e)
            if is_overflow and len(images) > 1:
                new_count = max(1, len(images) // 2)
                print(f"⚠️ Themen-Klassifikation: Kontext gesprengt ({len(images)} Bilder) — versuche erneut mit {new_count} Bild(ern).")
                images = images[:new_count]
                continue
            print(f"⚠️ Themen-Klassifikation fehlgeschlagen: {detail}")
            return {}

    if parsed is None:
        return {}

    scores = {}
    for t in topics:
        val = parsed.get(t)
        if val is None:
            # Manche Modelle normalisieren Groß-/Kleinschreibung oder
            # Leerzeichen im zurückgegebenen Key — tolerant nachschauen,
            # statt den Treffer wegen einer Kleinigkeit zu verlieren.
            for k, v in parsed.items():
                if isinstance(k, str) and k.strip().lower() == t.lower():
                    val = v
                    break
        try:
            scores[t] = max(0.0, min(100.0, float(val)))
        except (TypeError, ValueError):
            pass
    return scores


def _encode_images(files):
    images_b64 = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                images_b64.append(base64.b64encode(fh.read()).decode("utf-8"))
        except Exception:
            pass
    return images_b64


def _analyze_inner(video_basename, base_dir):
    frame_dir = os.path.join(base_dir, ".thumbs", video_basename, "large")
    if not os.path.isdir(frame_dir):
        print(f"ℹ️ Kein Filmstrip für {video_basename} vorhanden, überspringe AI-Analyse.")
        return

    files = _pick_frames(frame_dir, MAX_FRAMES)
    if not files:
        print(f"⚠️ Filmstrip-Ordner für {video_basename} ist vorhanden, aber leer (keine .jpg-Dateien) — AI-Analyse übersprungen.")
        return

    # Retry mit halbierter Bildzahl bei einem erkannten Kontext-Overflow
    # (Ollama meldet "exceed_context_size_error", wenn zu viele/große Bilder
    # das Kontextfenster des Modells sprengen) — statt die Analyse komplett
    # aufzugeben. Andere Fehler (Modell fehlt, Verbindung weg) werden NICHT
    # wiederholt, das würde nur denselben Fehler reproduzieren.
    description = None
    images_b64 = []
    last_error = None
    while files:
        images_b64 = _encode_images(files)
        if not images_b64:
            print(f"⚠️ Konnte keine der {len(files)} Filmstrip-Bilddateien für {video_basename} lesen/kodieren — AI-Analyse übersprungen.")
            return
        payload_dict = {
            "model": OLLAMA_MODEL,
            "prompt": PROMPT,
            "images": images_b64,
            "stream": False
        }
        if OLLAMA_CONTEXT_SIZE > 0:
            payload_dict["options"] = {"num_ctx": OLLAMA_CONTEXT_SIZE}
        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            description = result.get("response", "").strip()
            if not description and len(files) > 1:
                # Leere Antwort ohne Fehler — in der Praxis oft dasselbe
                # Symptom wie ein expliziter Kontext-Overflow (das Modell
                # bricht die Ausgabe einfach ab), nur dass Ollama hier
                # gar keinen Fehler meldet. Wird bewusst genauso behandelt
                # wie ein erkannter Overflow.
                new_count = max(1, len(files) // 2)
                print(f"⚠️ Ollama lieferte eine leere Antwort ({len(files)} Bilder, {video_basename}) — versuche erneut mit {new_count} Bild(ern).")
                files = files[:new_count]
                description = None
                continue
            break
        except Exception as e:
            is_overflow, detail = _describe_ollama_error_detailed(e)
            last_error = detail
            if is_overflow and len(files) > 1:
                new_count = max(1, len(files) // 2)
                print(f"⚠️ Ollama-Kontext gesprengt ({len(files)} Bilder, {video_basename}) — versuche erneut mit {new_count} Bild(ern).")
                files = files[:new_count]
                continue
            print(f"❌ Ollama-Analyse fehlgeschlagen für {video_basename}: {detail}")
            return

    if description is None:
        print(f"❌ Ollama-Analyse fehlgeschlagen für {video_basename}: {last_error}")
        return

    if not description:
        print(f"⚠️ Ollama lieferte keine Beschreibung für {video_basename}.")
        return

    topics_result = {}
    top_topic, top_topic_score = None, None
    detected_topics = []  # alle Themen über der Schwelle, absteigend sortiert
    if AI_TOPICS_ENABLED and AI_TOPICS:
        # Dieselbe (ggf. schon reduzierte) Bildmenge wiederverwenden, statt
        # unabhängig nochmal denselben Kontext-Overflow zu riskieren.
        topics_result = _classify_topics(images_b64, AI_TOPICS)
        qualifying = {t: s for t, s in topics_result.items() if s >= AI_TOPICS_THRESHOLD}
        if qualifying:
            detected_topics = [
                {"topic": t, "score": s}
                for t, s in sorted(qualifying.items(), key=lambda kv: kv[1], reverse=True)
            ]
            top_topic = detected_topics[0]["topic"]
            top_topic_score = detected_topics[0]["score"]

    # 1) Eigene JSON-Metadatei — vom Dashboard gelesen. Erst lesen & mergen,
    # NIE blind überschreiben — sonst geht ein evtl. schon vorhandenes
    # Transkript (transcribe_audio.py schreibt in dieselbe Datei) verloren,
    # z.B. wenn ai_analyze.py isoliert nochmal läuft (etwa über einen
    # gezielten "nur Beschreibung neu"-Befehl, der bewusst NICHT auch
    # transcribe_audio.py mit aufruft). transcribe_audio.py macht das schon
    # länger korrekt so — hier hatte genau dasselbe Muster gefehlt.
    meta_path = os.path.join(base_dir, f"{video_basename}.ai.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    try:
        meta["description"] = description
        meta["model"] = OLLAMA_MODEL
        meta["frame_count"] = len(images_b64)
        meta["ts"] = time.time()
        if topics_result:
            meta["topics"] = topics_result
        if detected_topics:
            meta["detected_topics"] = detected_topics
            # top_topic/top_topic_confidence bleiben zusätzlich erhalten — z.B.
            # für den Export-Ordnernamen, der bewusst nur EIN Thema im Namen
            # trägt, sonst wird der Ordnername schnell unhandlich lang.
            meta["top_topic"] = top_topic
            meta["top_topic_confidence"] = top_topic_score
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except Exception as e:
        print(f"❌ Konnte {meta_path} nicht schreiben: {e}")

    # 2) XMP-Sidecar für Immich (dc:description + dc:subject für Themen)
    xmp_path = os.path.join(base_dir, f"{video_basename}.mp4.xmp")
    subject_block = ""
    qualifying_topics = {t: s for t, s in topics_result.items() if s >= AI_TOPICS_THRESHOLD}
    if qualifying_topics:
        tags_xml = "".join(f"<rdf:li>{_xml_escape(t)}</rdf:li>" for t in qualifying_topics)
        subject_block = f"""
   <dc:subject>
    <rdf:Bag>
     {tags_xml}
    </rdf:Bag>
   </dc:subject>"""
    xmp = f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:description>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{_xml_escape(description)}</rdf:li>
    </rdf:Alt>
   </dc:description>{subject_block}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    try:
        with open(xmp_path, "w", encoding="utf-8") as f:
            f.write(xmp)
    except Exception as e:
        print(f"❌ Konnte {xmp_path} nicht schreiben: {e}")

    if search_index is not None:
        try:
            search_index.index_event(f"{video_basename}.mp4", base_dir, description, topics=list(qualifying_topics.keys()))
        except Exception as e:
            print(f"⚠️ Suchindex-Update fehlgeschlagen für {video_basename}: {e}")

    print(f"✅ AI-Analyse fertig für {video_basename}: {description}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ai_analyze.py <video_basename> <base_dir>")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2])
