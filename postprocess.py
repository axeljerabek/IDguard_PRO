#!/usr/bin/env python3
"""
postprocess.py - Einstiegspunkt für die komplette Nachbearbeitung einer
fertigen Aufnahme. Ruft ai_analyze.py (Vision-Beschreibung + Themen) und
transcribe_audio.py (Sprache-zu-Text) SEQUENZIELL im selben Prozess auf.

WARUM SEQUENZIELL, NICHT ALS ZWEI PARALLELE subprocess.Popen-Aufrufe: beide
Schritte lesen und schreiben dieselbe <video>.ai.json (lesen, mit eigenem
Feld ergänzen, zurückschreiben). Liefen sie parallel, könnte je nachdem wer
zuletzt schreibt, die Beschreibung ODER das Transkript verloren gehen
(klassisches Lost-Update-Problem). Sequenziell in einem Prozess umgeht das
komplett, ohne Datei-Locking zu brauchen.

Jeder der beiden Schritte prüft selbst, ob sein Feature überhaupt aktiviert
ist (AI_ANALYSIS_ENABLED / TRANSCRIPTION_ENABLED) — hier wird bewusst immer
versucht, beide aufzurufen, kein Grund für Sonderfälle.

Aufruf (von recorder_pipeline.py per subprocess.Popen, fire-and-forget):
    python3 postprocess.py <video_basename> <base_dir>
"""
import sys
import os

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)

import ai_analyze
import transcribe_audio

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: postprocess.py <video_basename> <base_dir>")
        sys.exit(1)
    video_basename, base_dir = sys.argv[1], sys.argv[2]
    ai_analyze.analyze(video_basename, base_dir)
    transcribe_audio.transcribe(video_basename, base_dir)
