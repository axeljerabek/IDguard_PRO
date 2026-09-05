"""
loitering_detection.py — erkennt, ob eine Person über längere Zeit an
derselben Stelle im Bild verharrt ("herumlungern"). Bewusst NICHT auf den
Pose-Keypoints aufgebaut wie fall/raised_hands -- reine Positions-Tracking-
Logik über die normale Detection-Bounding-Box, kein Pose-Modell nötig.

Eigenständiges, reines Funktions-/Klassen-Modul, unabhängig testbar mit
simulierten Zeitstempeln (keine echte Kamera/Uhrzeit nötig).
"""


def box_center(bbox):
    """(x1, y1, x2, y2) -> (cx, cy)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


class LoiteringTracker:
    """Verfolgt EINE Person-Position über Zeit (bewusst vereinfacht auf
    einen einzelnen "Anker" statt vollständigem Multi-Personen-Tracking mit
    IDs -- für den typischen Zuhause-Fall von 0-1 Personen im Bild reicht
    das; bei mehreren Personen wird die räumlich nächste zum aktuellen
    Anker weiterverfolgt, eine neue Person weit weg setzt einen neuen
    Anker). Bewegt sich die verfolgte Position zu weit vom ursprünglichen
    Punkt weg, gilt das als "weitergegangen", der Anker wird neu gesetzt --
    Herumlungern heißt "bleibt in etwa an einem Fleck", nicht "verlässt nie
    das Bild"."""

    def __init__(self, position_tolerance=0.15, min_duration_sec=30):
        # position_tolerance: erlaubte Positionsabweichung, als Anteil der
        # jeweiligen Bild-Dimension (0.15 = 15% von Breite/Höhe) --
        # relative statt absolute Pixel, damit dieselbe Einstellung bei
        # unterschiedlichen Kamera-Auflösungen vergleichbar bleibt.
        self.position_tolerance = position_tolerance
        self.min_duration_sec = min_duration_sec
        self._anchor_pos = None
        self._anchor_time = None
        self._confirmed = False

    def update(self, person_center, frame_width, frame_height, now):
        """person_center: (x, y) in Pixeln, oder None, falls gerade keine
        Person im Bild ist. now: aktueller Zeitstempel (time.time() o.ä.,
        als Zahl -- Einheit egal, solange konsistent mit min_duration_sec).
        Gibt True zurück GENAU im Moment der Erstbestätigung, danach False,
        bis reset() (z.B. weil die Person das Bild verlassen hat)."""
        if person_center is None:
            self.reset()
            return False

        if self._anchor_pos is None:
            self._anchor_pos = person_center
            self._anchor_time = now
            return False

        dx = abs(person_center[0] - self._anchor_pos[0]) / max(frame_width, 1)
        dy = abs(person_center[1] - self._anchor_pos[1]) / max(frame_height, 1)
        if dx > self.position_tolerance or dy > self.position_tolerance:
            # Zu weit vom Anker weg -- Person ist weitergegangen, neuer Anker.
            self._anchor_pos = person_center
            self._anchor_time = now
            self._confirmed = False
            return False

        elapsed = now - self._anchor_time
        if elapsed >= self.min_duration_sec and not self._confirmed:
            self._confirmed = True
            return True
        return False

    def reset(self):
        self._anchor_pos = None
        self._anchor_time = None
        self._confirmed = False

    @property
    def is_tracking(self):
        """Für Diagnose/Status-Anzeige: verfolgt der Tracker gerade
        überhaupt jemanden (auch wenn noch nicht als Loitering bestätigt)."""
        return self._anchor_pos is not None
