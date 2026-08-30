#!/usr/bin/env python3
"""
backfill_filmstrips.py - Erzeugt Filmstrip-Frames (small + large, wie von
recorder_pipeline.py live erfasst) nachträglich aus bereits bestehenden
Aufnahmen, per ffmpeg gleichmäßig über die tatsächliche Videolänge verteilt.

Keine Erkennungs-Boxen auf den small-Frames (anders als bei live erfassten) —
die Erkennungsdaten von damals existieren nicht mehr, reiner Frame-Grab.

Überspringt Videos, die schon einen Filmstrip-Ordner haben (z.B. live
erfasste) — überschreibt nichts. Rein lesend/ergänzend, rührt die laufende
Pipeline nicht an.

Aufruf:
    python3 backfill_filmstrips.py
    python3 backfill_filmstrips.py --count 10
    python3 backfill_filmstrips.py --analyze     # startet danach direkt ai_analyze.py pro Video
    python3 backfill_filmstrips.py --dir /pfad/zu/alerts
"""
import os
import sys
import glob
import json
import shutil
import argparse
import subprocess

import cv2

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
try:
    from config import ALERTS_DIR, SETTINGS_F
except ImportError:
    ALERTS_DIR = "./alerts"
    SETTINGS_F = "pipeline_settings.json"

ARCHIVE_DIR = os.path.join(ALERTS_DIR, "archive")
SMALL_W = 560
LARGE_W = 1280


def _default_count():
    try:
        with open(SETTINGS_F) as f:
            return int(json.load(f).get("FILMSTRIP_COUNT", 8)) or 8
    except Exception:
        return 8


def _get_duration(video_path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True
        )
        d = float(out.stdout.strip())
        return d if d > 0 else None
    except Exception:
        return None


def _extract_frame_at(video_path, ts, out_path, width):
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
             "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except subprocess.CalledProcessError:
        return False


def backfill_filmstrip(video_path, thumbs_root, count):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    fs_dir = os.path.join(thumbs_root, basename)
    if os.path.isdir(fs_dir):
        return "skipped"  # schon vorhanden (z.B. live erfasst) — nichts überschreiben

    duration = _get_duration(video_path)
    if not duration:
        return "failed"

    small_dir = os.path.join(fs_dir, "small")
    large_dir = os.path.join(fs_dir, "large")
    os.makedirs(small_dir, exist_ok=True)
    os.makedirs(large_dir, exist_ok=True)

    timestamps = {}
    for i in range(count):
        # Mitte jedes gleich großen Segments statt exakt am Rand — vermeidet
        # Frame 0 (oft ein Encoder-Init-Artefakt) und das allerletzte Bild
        # (manchmal unvollständig).
        ts = min(duration * (i + 0.5) / count, max(0.0, duration - 0.1))
        large_path = os.path.join(large_dir, f"{i:04d}.jpg")
        if not _extract_frame_at(video_path, ts, large_path, LARGE_W):
            continue
        img = cv2.imread(large_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        small = cv2.resize(img, (SMALL_W, max(1, int(h * SMALL_W / w))))
        cv2.imwrite(os.path.join(small_dir, f"{i:04d}.jpg"), small)
        timestamps[str(i)] = round(ts, 2)

    if not timestamps:
        shutil.rmtree(fs_dir, ignore_errors=True)
        return "failed"

    with open(os.path.join(fs_dir, "timestamps.json"), "w") as f:
        json.dump(timestamps, f)
    return "done"


def run(directory, thumbs_root, count, do_analyze):
    videos = sorted(glob.glob(os.path.join(directory, "*.mp4")))
    done = skipped = failed = 0
    for v in videos:
        result = backfill_filmstrip(v, thumbs_root, count)
        name = os.path.basename(v)
        if result == "done":
            print(f"✅ {name}")
            done += 1
            if do_analyze:
                basename = os.path.splitext(name)[0]
                try:
                    subprocess.run([sys.executable, os.path.join(DIR, "ai_analyze.py"), basename, directory])
                except Exception as e:
                    print(f"⚠️ AI-Analyse für {name} fehlgeschlagen: {e}")
        elif result == "skipped":
            skipped += 1
        else:
            print(f"❌ Fehlgeschlagen: {name}")
            failed += 1
    return done, skipped, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill von Filmstrip-Frames für bestehende Aufnahmen ohne Filmstrip.")
    parser.add_argument("--count", type=int, default=None,
                         help="Anzahl Frames pro Video (Standard: aktueller FILMSTRIP_COUNT-Wert aus den Settings, sonst 8)")
    parser.add_argument("--dir", action="append",
                         help="Zusätzliches Verzeichnis (Standard: ALERTS_DIR + ALERTS_DIR/archive)")
    parser.add_argument("--analyze", action="store_true",
                         help="Startet nach jedem erzeugten Filmstrip direkt ai_analyze.py für dieses Video "
                              "(braucht eine laufende, konfigurierte Ollama-Instanz — läuft SYNCHRON, kann "
                              "bei vielen Videos eine Weile dauern)")
    args = parser.parse_args()

    count = args.count or _default_count()
    dirs = args.dir if args.dir else [ALERTS_DIR, ARCHIVE_DIR]

    total_done = total_skipped = total_failed = 0
    for d in dirs:
        if not os.path.isdir(d):
            print(f"(übersprungen, existiert nicht: {d})")
            continue
        thumbs_root = os.path.join(d, ".thumbs")
        os.makedirs(thumbs_root, exist_ok=True)
        print(f"--- {d} ({count} Frames/Video) ---")
        done, skipped, failed = run(d, thumbs_root, count, args.analyze)
        total_done += done
        total_skipped += skipped
        total_failed += failed

    print(f"\nFertig: {total_done} erzeugt, {total_skipped} übersprungen (schon vorhanden), {total_failed} fehlgeschlagen.")
    if total_done and not args.analyze:
        print("Tipp: mit --analyze direkt die KI-Beschreibung für die neuen Filmstrips mit erzeugen lassen.")
