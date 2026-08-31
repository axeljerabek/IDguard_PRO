#!/usr/bin/env python3
"""
face_recognize.py - Erkennt Gesichter in den bereits vorhandenen "large"
Filmstrip-Frames einer Aufnahme (kein neuer Frame-Grab nötig), extrahiert
pro Gesicht ein 512-dim-Embedding (InsightFace), schneidet ein Thumbnail
aus und speichert beides in faces.db.

Versucht JEDES neu erkannte Gesicht sofort gegen bereits benannte Personen
zu matchen (Centroid-Vergleich) — nur was dabei nicht zugeordnet werden
kann, bleibt für den nächsten cluster_faces.py-Lauf liegen.

Modellwahl (buffalo_s/m/l, antelopev2) über WHISPER... nein, über
FACE_MODEL_PACK in den Settings, analog zu YOLO_VERSION/MODEL_SIZE.

Aufruf: python3 face_recognize.py <video_basename> <base_dir>
"""
import os
import sys
import json
import glob
import shutil
import time

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
try:
    from config import SETTINGS_F
except ImportError:
    SETTINGS_F = "pipeline_settings.json"
try:
    import faces_db
except ImportError:
    faces_db = None


def _load_settings():
    try:
        with open(SETTINGS_F) as f:
            return json.load(f)
    except Exception:
        return {}


_settings = _load_settings()
FACE_RECOGNITION_ENABLED = bool(_settings.get("FACE_RECOGNITION_ENABLED", False))
FACE_MODEL_PACK = _settings.get("FACE_MODEL_PACK", "buffalo_s")
FACE_MIN_CONFIDENCE = float(_settings.get("FACE_MIN_CONFIDENCE", 0.5))
KNOWN_PERSON_THRESHOLD = 0.5  # dasselbe Maß wie in cluster_faces.py — bewusst konsistent gehalten

_app = None
_app_load_failed = False


def _fix_antelopev2_nesting(model_pack):
    """Bekannter Bug: das antelopev2-Zip von InsightFace entpackt sich in
    einen verschachtelten antelopev2/antelopev2/-Ordner statt direkt
    antelopev2/*.onnx — der Modell-Loader findet die Dateien dann nicht und
    bricht mit 'assert detection in self.models' ab. Hier automatisch
    geglättet, damit niemand das von Hand fixen muss. Betrifft nur
    antelopev2 — die buffalo_*-Packs haben dieses Problem nicht."""
    if model_pack != "antelopev2":
        return
    home = os.path.expanduser("~/.insightface/models")
    model_dir = os.path.join(home, model_pack)
    nested = os.path.join(model_dir, model_pack)
    if os.path.isdir(nested):
        for f in os.listdir(nested):
            shutil.move(os.path.join(nested, f), os.path.join(model_dir, f))
        os.rmdir(nested)

# Ein echtes InsightFace-Modell-Pack ist immer deutlich größer als das —
# alles darunter deutet auf einen abgebrochenen/korrupten Download hin
# (Netzwerk-Aussetzer mitten im Download, o.ä.), nicht auf ein vollständiges
# Pack. Bewusst konservativ (das kleinste Pack, buffalo_s, hat trotzdem
# mehrere hundert MB).
MODEL_DIR_MIN_BYTES = 10_000_000

MAX_LOAD_ATTEMPTS = 3
RETRY_DELAY_SEC = 3


def _model_dir_size(model_pack):
    home = os.path.expanduser("~/.insightface/models")
    model_dir = os.path.join(home, model_pack)
    if not os.path.isdir(model_dir):
        return 0
    total = 0
    for root, _, files in os.walk(model_dir):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


def _get_app():
    global _app, _app_load_failed
    if _app is not None:
        return _app
    if _app_load_failed:
        return None

    last_error = None
    for attempt in range(1, MAX_LOAD_ATTEMPTS + 1):
        try:
            # Proaktiv statt reaktiv: den bekannten antelopev2-Verschachtelungs-
            # Bug schon VOR jedem Ladeversuch glätten (kein-op für andere Packs
            # und für bereits geglättete Ordner), statt erst nach einem
            # fehlgeschlagenen Versuch draufzukommen.
            _fix_antelopev2_nesting(FACE_MODEL_PACK)
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name=FACE_MODEL_PACK, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            _app = app
            if attempt > 1:
                print(f"✅ Gesichtserkennungs-Modell '{FACE_MODEL_PACK}' beim {attempt}. Versuch erfolgreich geladen.")
            return _app
        except Exception as e:
            last_error = e
            print(f"⚠️ Ladeversuch {attempt}/{MAX_LOAD_ATTEMPTS} für '{FACE_MODEL_PACK}' fehlgeschlagen: {e}")
            size = _model_dir_size(FACE_MODEL_PACK)
            if size < MODEL_DIR_MIN_BYTES:
                # Sieht nach abgebrochenem/korruptem Download aus (z.B. Netz-
                # Aussetzer mittendrin) — kompletten Ordner löschen, damit
                # InsightFace beim nächsten Versuch wirklich frisch neu
                # herunterlädt, statt an denselben kaputten Dateien hängen
                # zu bleiben (die sonst bei JEDEM künftigen Start denselben
                # Fehler wiederholt hätten, nie von selbst geheilt).
                home = os.path.expanduser("~/.insightface/models")
                model_dir = os.path.join(home, FACE_MODEL_PACK)
                if os.path.isdir(model_dir):
                    print(f"🗑️ Modell-Ordner {model_dir} sieht unvollständig aus ({size} Bytes) — lösche für Neu-Download.")
                    shutil.rmtree(model_dir, ignore_errors=True)
            if attempt < MAX_LOAD_ATTEMPTS:
                time.sleep(RETRY_DELAY_SEC)

    print(f"❌ Gesichtserkennungs-Modell '{FACE_MODEL_PACK}' konnte nach {MAX_LOAD_ATTEMPTS} Versuchen nicht geladen werden — Feature bleibt inaktiv: {last_error}")
    _app_load_failed = True
    return None


def recognize(video_basename, base_dir):
    if not FACE_RECOGNITION_ENABLED:
        return
    if faces_db is None:
        return
    frame_dir = os.path.join(base_dir, ".thumbs", video_basename, "large")
    if not os.path.isdir(frame_dir):
        return
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if not frames:
        return

    app = _get_app()
    if app is None:
        return

    import cv2
    crop_dir = os.path.join(base_dir, ".thumbs", video_basename, "faces")
    os.makedirs(crop_dir, exist_ok=True)

    centroids = faces_db.get_person_centroids()
    saved_count = 0
    matched_count = 0

    # Nicht jeden Filmstrip-Frame durchsuchen (teuer bei vielen Frames,
    # außerdem meist redundant — dieselbe Person steht mehrere Frames lang
    # im Bild). Ein Gesicht pro Frame reicht für die Zwecke hier locker.
    for i, frame_path in enumerate(frames):
        img = cv2.imread(frame_path)
        if img is None:
            continue
        try:
            faces = app.get(img)
        except Exception as e:
            print(f"⚠️ Gesichtserkennung auf Frame {frame_path} fehlgeschlagen: {e}")
            continue

        for j, face in enumerate(faces):
            if float(face.det_score) < FACE_MIN_CONFIDENCE:
                continue
            x1, y1, x2, y2 = [max(0, int(v)) for v in face.bbox]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_filename = f"{i:04d}_{j}.jpg"
            crop_path = os.path.join(crop_dir, crop_filename)
            try:
                cv2.imwrite(crop_path, crop)
            except Exception:
                continue

            embedding = face.normed_embedding  # bereits L2-normalisiert (Norm 1.0) — face.embedding ist es NICHT
            person_id = None
            if centroids:
                best_id, best_sim = None, 0.0
                for pid, (_, centroid) in centroids.items():
                    sim = float(embedding @ centroid)  # beide normalisiert -> Cosine == Skalarprodukt, kein Nachnormalisieren nötig
                    if sim > best_sim:
                        best_id, best_sim = pid, sim
                if best_sim >= KNOWN_PERSON_THRESHOLD:
                    person_id = best_id
                    matched_count += 1

            faces_db.add_face(
                f"{video_basename}.mp4", base_dir,
                os.path.join(".thumbs", video_basename, "faces", crop_filename),
                embedding.tolist(), float(face.det_score), person_id=person_id
            )
            saved_count += 1

    if saved_count:
        print(f"✅ Gesichtserkennung für {video_basename}: {saved_count} Gesicht(er) gespeichert, {matched_count} direkt bekannten Personen zugeordnet.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: face_recognize.py <video_basename> <base_dir>")
        sys.exit(1)
    recognize(sys.argv[1], sys.argv[2])
