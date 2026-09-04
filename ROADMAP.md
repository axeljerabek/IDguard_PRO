# IDguard PRO — Roadmap

Laufende, priorisierte Liste. Wird Stück für Stück abgearbeitet, nicht alles auf einmal.

---

## Tier 1 — Kleine, schnelle Gewinne

- [ ] **Home Assistant Anbindung via MQTT** — Events (Aufnahme gestartet/fertig, Topic erkannt) an einen MQTT-Broker publishen, HA unterstützt MQTT Discovery nativ. Kleiner, isolierter Baustein.

## Tier 2 — Anomalie-Erkennung (Weg A zuerst, Weg B zurückgestellt)

- [ ] **Isolation Forest auf dem bestehenden Text-Embedding** — `sentence-transformers`-Vektor aus `search_index.db` wiederverwenden (kein CLIP-Bild-Embedding nötig, existiert in IDguard aktuell nicht). Baseline-Training per Cronjob (z.B. wöchentlich, letzte 7 Tage pro Kamera), Inferenz pro neuem Event, `[ANOMALY]`-Tag in `.ai.json` bei Ausreißer. Ein/ausschaltbar per Setting.
- [ ] *(zurückgestellt, nicht vor Tier 2 bewertet)* **VAE/Autoencoder auf Rohbildern** — nur angehen, falls Weg A in der Praxis an Grenzen stößt. Erstes selbst trainiertes Modell im System (Abweichung vom Baukasten-Prinzip), GPU-Trainings-Pipeline, deutlich höherer Aufwand und Tuning-Bedarf.

## Tier 3 — Externe API / MAM-Anbindung (großes Subsystem, in Einzelschritte zerlegt)

- [ ] Job-Annahme-Endpunkt (Upload/Referenz) mit Auftrags-spezifischen Parametern (Topics, COCO-Klassen, Audio-Trigger-Phrasen) statt globaler Settings
- [ ] API-Key-Auth für externe Aufrufer, getrennt von der bestehenden Dashboard-Session-Auth
- [ ] Async-Job-Tracking + passiver Status-Endpunkt (`GET /api/job/<id>`)
- [ ] Callback/Webhook-Zustellung bei Fertigstellung, inkl. Retry-Logik
- [ ] Video-**Segment**-Export (Ausschnitt statt ganzes Video) — existiert noch nicht, eigene Teilaufgabe
- [ ] Auslieferung: Video/Segment + angereicherte Metadaten + Gesichtsdaten über die API

## Tier 4 — Agent-Steuerung (baut auf Tier 3 auf)

- [ ] Agent kann über dieselben API-Endpunkte aus Tier 3 "manuell" Aufnahmen/Exporte auslösen — kein eigenes Subsystem, sondern Zugriff auf die MAM-API
- [ ] Perspektivisch: MCP-Server-Wrapper um die API, für direkten Agenten-Zugriff (Claude o.ä.)

## Kleine Prio / bei Gelegenheit

- [ ] **YOLO Pose Estimation** — Sturzerkennung, Haltungs-Klassifikation. Fügt sich strukturell nahtlos ein (YOLO läuft eh), aber von dir selbst als niedrige Priorität eingestuft.

## Bereits besprochen, noch offen von früheren Sessions

- [ ] Watchfolder Modus 1 (wachsende Datei live als Stream lesen) — technisch unsicher bei MP4-Quellen (moov-Atom-Problem), nur mit echter Beispieldatei sinnvoll bewertbar
- [ ] Schutzmechanismus für Encode-Modus-Speicherverbrauch (MJPEG/USB-Kameras, ~1,74GB/Kamera bei 1080p/10s Pre-Roll) — Rückmeldung von Axel noch offen
- [ ] MJPEG/USB-Kamera-Encoding-Pfad — Kernstück fertig, aber noch nicht produktiv mit echter Hardware getestet

---

## Bereits erledigt (heute)

RTMP+RTSP, MJPEG/USB-Encode-Pfad, Watchfolder-Import (Modus 2), HEVC-Retranscode (GPU-beschleunigt), Prozess-Orchestrierung-Fixes (Neustart-Erkennung, systemd-Service, Worker-Benennung), Postprocess-Watchdog (hängendes Ollama/GPU blockiert nicht mehr die ganze Warteschlange), eigenes Notizfeld (XMP-Export), Export-Unterordner für Sammelexporte, Export-Auswahl-Checkboxen (Video/Metadaten/Thumbs), Löschen-nach-Export-Option, Favorite/Sterne-Bewertung (xmp:Rating), Tages-/Wochen-Zusammenfassung per LLM (inkl. Dashboard-Karte + Cronjob-Beispiel).
