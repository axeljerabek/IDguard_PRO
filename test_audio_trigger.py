#!/usr/bin/env python3
"""
test_audio_trigger.py - Manueller Test für audio_trigger.py: prüft, ob CLAP
sauber lädt und ob ein kurzes Testsignal wie erwartet klassifiziert wird —
OHNE die volle Pipeline/eine Kamera zu brauchen.

Aufruf: python3 test_audio_trigger.py
(Braucht Internetzugang beim ersten Mal, lädt ~/.cache/huggingface befüllt.)
"""
import time
import logging
import numpy as np

from audio_trigger import AudioTrigger

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("test")

SAMPLE_RATE = 48000

trigger = AudioTrigger(logger, "TestCam", sample_rate=SAMPLE_RATE)


def fake_settings():
    return {
        "AUDIO_TRIGGER_ENABLED": True,
        "AUDIO_TRIGGER_CATEGORIES": ["whispering", "dog barking", "glass breaking", "silence"],
        "AUDIO_TRIGGER_THRESHOLD": 0.2,
        "AUDIO_TRIGGER_INTERVAL_SEC": 1.0,
    }


print("Starte Hintergrund-Thread (lädt CLAP beim ersten Settings-Check, kann etwas dauern)...")
trigger.start(fake_settings)

# 3 Sekunden Stille als Testsignal einspeisen
print("Speise 3s Stille ein...")
silence = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
trigger.feed(silence, SAMPLE_RATE)

# Auf das erste Klassifikationsergebnis warten (Modell-Download + erster
# Durchlauf können beim allerersten Mal ein bis zwei Minuten dauern)
print("Warte auf Klassifikation (kann beim ersten Mal 1-2 Min dauern, Modell-Download)...")
for i in range(120):
    triggered, label = trigger.is_triggered()
    if trigger._text_embeds is not None:
        print("\n✅ Modell geladen und Kategorien codiert.")
        print(f"Ergebnis nach {i}s: triggered={triggered}, label={label!r}")
        break
    time.sleep(1)
    print(".", end="", flush=True)
else:
    print("\n⚠️ Nach 2 Minuten immer noch nicht bereit — prüfe Internetzugang/Log-Ausgabe oben.")

trigger.stop()
print("\nFertig.")
