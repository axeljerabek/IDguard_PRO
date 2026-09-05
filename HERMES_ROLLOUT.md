# Hermes Agent Control — Rollout Plan

Ziel: kontrolliert, eine Fähigkeit nach der anderen, mit echter Verifikation
bei jedem Schritt — nicht alles auf einmal freischalten und hoffen.

## Schritt 0: Voraussetzungen schaffen

1. Dashboard → "External API (Remote Control)"-Karte → Key erzeugen,
   Label z.B. "Hermes". Sofort kopieren (wird nur einmal angezeigt).
2. Prüfen, ob Hermes überhaupt ein Werkzeug hat, das echte HTTP-Requests
   stellen kann (curl, ein HTTP-Tool, etc.) — nicht nur Dateien lesen.
   Ohne das kann er die API gar nicht benutzen, egal was freigeschaltet ist.
3. `agent_config.json`: `agent_control_enabled` bleibt vorerst `false`.
   Noch nichts einschalten.
4. Hermes `GET /api/v1/agent/capabilities` aufrufen lassen — funktioniert
   auch bei ausgeschaltetem Master, zeigt ihm auf einen Blick, was es
   überhaupt gibt und was davon (noch) aus ist. Guter erster Test, ob
   sein HTTP-Werkzeug grundsätzlich funktioniert, ohne dass er dabei
   irgendwas anfassen kann.

## Schritt 1: Nur `search` — read-only, niedrigstes Risiko

1. `agent_control_enabled: true` setzen
2. `capabilities.search.enabled: true` — alles andere bleibt `false`
3. Hermes bitten, etwas Konkretes zu suchen (z.B. "such nach 'Paket'")
4. **Verifizieren:** stimmen die Treffer mit dem überein, was auch die
   Dashboard-Suche liefert? Meldet er Ergebnisse, die es nicht gibt?
5. Erst weiter, wenn das sauber und mehrfach funktioniert.

## Schritt 2: `cameras_toggle` dazu

1. `capabilities.cameras_toggle.enabled: true`
2. Hermes bitten, eine bestimmte Kamera auszuschalten
3. **Verifizieren im Dashboard selbst** (nicht nur Hermes' Wort nehmen) —
   steht der Schalter dort jetzt wirklich auf aus?
4. Wieder einschalten lassen, nochmal im Dashboard bestätigen.

## Schritt 3: `pipeline_control` dazu

1. `capabilities.pipeline_control.enabled: true`
2. Status abfragen lassen, dann bewusst stop/start testen
3. **Verifizieren:** `ps aux | grep recorder_pipeline` — läuft/läuft nicht
   tatsächlich, wie gemeldet?

## Schritt 4: `settings_change`, falls gewünscht

1. `capabilities.settings_change.enabled: true`
2. Testen mit einem unkritischen Wert (z.B. `TARGET_FPS`)
3. Testen, ob ein Versuch außerhalb der Allowlist (z.B. Export-Pfad)
   korrekt mit 403 abgelehnt wird — das MUSS fehlschlagen

## Nie freischalten (aktuell nicht mal gebaut)

`delete`, `export` — keine Route existiert dafür, das Flag tut nichts.
Bleibt so, bis das bewusst separat entschieden wird.

## Bei jedem Schritt

- Erst EIN Capability-Flag ändern, nicht mehrere gleichzeitig
- Immer im Dashboard/System selbst verifizieren, nicht nur Hermes glauben
- Bei irgendetwas Unerwartetem: `agent_control_enabled: false` sofort
  zurücksetzen, das kappt alles auf einmal
