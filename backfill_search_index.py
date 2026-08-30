#!/usr/bin/env python3
"""
backfill_search_index.py - Nimmt alle bereits existierenden .ai.json-
Beschreibungen (von vor Einführung der Suche, oder falls die Datenbank mal
gelöscht/neu aufgesetzt wird) nachträglich in den Such-Index auf.

Rein lesend/ergänzend, rührt die laufende Pipeline nicht an — sicher
jederzeit manuell ausführbar.

Aufruf:
    python3 backfill_search_index.py
    python3 backfill_search_index.py --dir /pfad/zu/alerts
"""
import os
import sys
import glob
import json
import argparse

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
try:
    from config import ALERTS_DIR
except ImportError:
    ALERTS_DIR = "./alerts"

ARCHIVE_DIR = os.path.join(ALERTS_DIR, "archive")

import search_index


def backfill(directory):
    ai_files = glob.glob(os.path.join(directory, "*.ai.json"))
    done = skipped = failed = 0
    for ai_path in ai_files:
        video_name = os.path.basename(ai_path)[:-len(".ai.json")] + ".mp4"
        video_path = os.path.join(directory, video_name)
        if not os.path.exists(video_path):
            skipped += 1
            continue
        try:
            with open(ai_path) as f:
                meta = json.load(f)
            description = meta.get("description")
            if not description:
                skipped += 1
                continue
            search_index.index_event(video_name, directory, description)
            print(f"✅ {video_name}")
            done += 1
        except Exception as e:
            print(f"❌ Fehlgeschlagen: {video_name}: {e}")
            failed += 1
    return done, skipped, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill des Such-Index aus bestehenden .ai.json-Dateien.")
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
        done, skipped, failed = backfill(d)
        total_done += done
        total_skipped += skipped
        total_failed += failed

    print(f"\nFertig: {total_done} indexiert, {total_skipped} übersprungen (keine Beschreibung/Video fehlt), {total_failed} fehlgeschlagen.")
