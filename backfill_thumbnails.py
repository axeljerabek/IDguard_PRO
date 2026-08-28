#!/usr/bin/env python3
"""
backfill_thumbnails.py - Erzeugt fehlende Vorschaubilder für ältere
Aufnahmen (von vor der Live-Thumbnail-Funktion). Da für diese Videos keine
Erkennungsdaten mehr existieren, gibt es KEINE Boxen — reiner Frame-Grab
per ffmpeg aus der vorhandenen MP4-Datei.

Rein lesend/ergänzend, rührt die laufende Pipeline nicht an — sicher
jederzeit manuell ausführbar.

Aufruf:
    python3 backfill_thumbnails.py
    python3 backfill_thumbnails.py --offset 3.0
    python3 backfill_thumbnails.py --dir /pfad/zu/alerts
"""
import os
import sys
import glob
import subprocess
import argparse

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
try:
    from config import ALERTS_DIR
except ImportError:
    ALERTS_DIR = "./alerts"

ARCHIVE_DIR = os.path.join(ALERTS_DIR, "archive")


def _extract_frame(video_path, thumb_path, offset_sec):
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(offset_sec), "-i", video_path,
             "-frames:v", "1", "-q:v", "3", thumb_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
            return True
    except subprocess.CalledProcessError:
        pass
    # Fallback: allererstes Frame, falls das Video kürzer als offset_sec ist
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", "-q:v", "3", thumb_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0
    except subprocess.CalledProcessError:
        return False


def backfill(directory, offset_sec):
    videos = sorted(glob.glob(os.path.join(directory, "*.mp4")))
    done = skipped = failed = 0
    for v in videos:
        thumb = os.path.splitext(v)[0] + ".jpg"
        if os.path.exists(thumb):
            skipped += 1
            continue
        if _extract_frame(v, thumb, offset_sec):
            print(f"✅ {os.path.basename(v)}")
            done += 1
        else:
            print(f"❌ Fehlgeschlagen: {os.path.basename(v)}")
            failed += 1
    return done, skipped, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill fehlender Thumbnails für ältere Aufnahmen.")
    parser.add_argument("--offset", type=float, default=2.0,
                         help="Sekunde im Video, aus der der Screenshot gezogen wird (Standard: 2.0)")
    parser.add_argument("--dir", action="append",
                         help="Zusätzliches Verzeichnis (Standard: ALERTS_DIR + ALERTS_DIR/archive)")
    args = parser.parse_args()

    dirs = args.dir if args.dir else [ALERTS_DIR, ARCHIVE_DIR]
    total_done = total_skipped = total_failed = 0
    for d in dirs:
        if not os.path.isdir(d):
            print(f"(übersprungen, existiert nicht: {d})")
            continue
        print(f"--- {d} ---")
        done, skipped, failed = backfill(d, args.offset)
        total_done += done
        total_skipped += skipped
        total_failed += failed

    print(f"\nFertig: {total_done} erzeugt, {total_skipped} übersprungen (schon vorhanden), {total_failed} fehlgeschlagen.")
