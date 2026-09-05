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


class MovementTracker:
    """Schätzt grob, ob eine Person geht oder rennt -- Geschwindigkeit in
    "Körperhöhen pro Sekunde" statt Pixel/Sekunde, damit dieselbe Person
    nah an der Kamera (groß im Bild) und weit weg (klein im Bild) bei
    gleichem TATSÄCHLICHEN Lauftempo vergleichbare Werte liefert, ohne
    Kalibrierung auf eine bestimmte Kamera-Distanz. Faustregel-Schwellwerte
    (~2 Körperhöhen/s ≈ Rennen, ~0,5-2 ≈ Gehen), keine gemessene Kalibrierung."""

    def __init__(self, running_threshold=1.8, walking_threshold=0.5):
        self.running_threshold = running_threshold
        self.walking_threshold = walking_threshold
        self._last_center = None
        self._last_time = None
        self._last_height = None

    def update(self, center, bbox_height, now):
        """Gibt "running"|"walking"|"stationary"|None zurück (None = noch
        nicht genug Datenpunkte, braucht mindestens einen vorherigen
        Aufruf, um eine Geschwindigkeit zu berechnen)."""
        if center is None:
            self.reset()
            return None
        if self._last_center is None or self._last_time is None:
            self._last_center, self._last_time, self._last_height = center, now, bbox_height
            return None
        dt = now - self._last_time
        if dt <= 0:
            return None
        distance = ((center[0] - self._last_center[0]) ** 2 + (center[1] - self._last_center[1]) ** 2) ** 0.5
        avg_height = (bbox_height + self._last_height) / 2
        self._last_center, self._last_time, self._last_height = center, now, bbox_height
        if avg_height <= 0:
            return None
        speed = (distance / avg_height) / dt
        if speed >= self.running_threshold:
            return "running"
        if speed >= self.walking_threshold:
            return "walking"
        return "stationary"

    def reset(self):
        self._last_center = None
        self._last_time = None
        self._last_height = None


def detect_close_proximity(person_boxes, proximity_ratio=0.6):
    """Prüft alle Paare erkannter Personen -- True, wenn mindestens zwei
    einander näher sind als proximity_ratio * ihre durchschnittliche
    Bounding-Box-Höhe (Bewegungsraum-Metrik statt kalibrierter Distanz --
    eine Box-Höhe entspricht ungefähr einer Körperlänge)."""
    if len(person_boxes) < 2:
        return False
    for i in range(len(person_boxes)):
        for j in range(i + 1, len(person_boxes)):
            b1, b2 = person_boxes[i], person_boxes[j]
            c1, c2 = box_center(tuple(b1[:4])), box_center(tuple(b2[:4]))
            h1, h2 = b1[3] - b1[1], b2[3] - b2[1]
            avg_height = (h1 + h2) / 2
            if avg_height <= 0:
                continue
            distance = ((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2) ** 0.5
            if distance <= avg_height * proximity_ratio:
                return True
    return False


class ProximityTracker:
    """Bestätigt Personen-Nähe erst nach mehreren aufeinanderfolgenden
    Frames -- zwei Personen, die sich normal aneinander vorbei bewegen
    (z.B. auf dem Gehweg), sind für einen Moment nah beieinander, das ist
    kein Grund zur Meldung. Anhaltende Nähe dagegen schon eher."""

    def __init__(self, required_consecutive=5):
        self.required_consecutive = required_consecutive
        self._consecutive_count = 0
        self._confirmed = False

    def update(self, is_close):
        if is_close:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0
            self._confirmed = False
            return False
        if self._consecutive_count >= self.required_consecutive and not self._confirmed:
            self._confirmed = True
            return True
        return False

    def reset(self):
        self._consecutive_count = 0
        self._confirmed = False
