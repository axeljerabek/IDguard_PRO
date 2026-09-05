"""
pose_fall_detection.py — Sturz-Heuristik auf Basis von YOLO-Pose-Keypoints
(COCO-17-Format). Bewusst als eigenständiges, reines Funktions-Modul gebaut,
damit die Kernlogik unabhängig von der restlichen Pipeline getestet werden
kann, bevor sie in recorder_pipeline.py eingehängt wird.

COCO-Keypoint-Reihenfolge (Index -> Körperteil):
0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle

Heuristik, bewusst einfach gehalten (kein trainiertes Klassifikations-
modell): der Winkel der "Wirbelsäulen-Achse" (Schulter-Mittelpunkt zu
Hüft-Mittelpunkt) gegenüber der Vertikalen. Stehen/Sitzen -> nahe an 0°
(senkrecht). Liegen/gestürzt -> nahe an 90° (waagerecht). Das ist ein
Startpunkt, kein kalibriertes medizinisches Werkzeug -- Schwellwert bewusst
einstellbar, nicht hart codiert.
"""
import math

KP_NOSE = 0
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12

MIN_KEYPOINT_CONFIDENCE = 0.3  # unterhalb dessen gilt ein Keypoint als "nicht gesehen"
DEFAULT_FALL_ANGLE_THRESHOLD = 55.0  # Grad von der Vertikalen -- Startwert, nicht kalibriert


def _kp_ok(keypoints, idx):
    """True, wenn dieser Keypoint mit ausreichender Konfidenz erkannt wurde."""
    if idx >= len(keypoints):
        return False
    x, y, conf = keypoints[idx]
    return conf >= MIN_KEYPOINT_CONFIDENCE


def _midpoint(keypoints, idx_a, idx_b):
    """Mittelpunkt zweier Keypoints, oder der einzelne, falls nur einer
    davon sicher genug erkannt wurde -- robuster gegen Verdeckung
    (z.B. eine Schulter durch Möbel verdeckt) als beide zu verlangen."""
    a_ok, b_ok = _kp_ok(keypoints, idx_a), _kp_ok(keypoints, idx_b)
    if a_ok and b_ok:
        return ((keypoints[idx_a][0] + keypoints[idx_b][0]) / 2,
                (keypoints[idx_a][1] + keypoints[idx_b][1]) / 2)
    if a_ok:
        return (keypoints[idx_a][0], keypoints[idx_a][1])
    if b_ok:
        return (keypoints[idx_b][0], keypoints[idx_b][1])
    return None


def torso_angle_from_vertical(keypoints):
    """Winkel der Schulter-Hüft-Achse gegenüber der Vertikalen, in Grad.
    0° = senkrecht (stehend/sitzend), 90° = waagerecht (liegend). Gibt
    None zurück, wenn nicht genug Keypoints sicher genug erkannt wurden,
    um eine sinnvolle Aussage zu treffen -- lieber "weiß nicht" als eine
    Zahl aus zu wenig Information zu erfinden."""
    shoulder_mid = _midpoint(keypoints, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    hip_mid = _midpoint(keypoints, KP_LEFT_HIP, KP_RIGHT_HIP)
    if shoulder_mid is None or hip_mid is None:
        return None
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    if dx == 0 and dy == 0:
        return None
    # atan2(dx, dy) statt atan2(dy, dx): 0° soll "senkrecht" bedeuten
    # (dy dominant, dx klein), nicht "waagerecht" -- Bildkoordinaten haben
    # y nach unten wachsend, das passt trotzdem, da nur der WINKEL
    # zwischen dem Vektor und der Vertikalen zählt, nicht die Richtung.
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)


def detect_fall(keypoints, bbox=None, angle_threshold=DEFAULT_FALL_ANGLE_THRESHOLD):
    """Prüft, ob eine Person-Pose auf einen Sturz/liegende Position hindeutet.

    keypoints: Liste von 17 (x, y, confidence)-Tupeln, COCO-Reihenfolge.
    bbox: optional (x1, y1, x2, y2) -- als Zusatz-Indiz genutzt (breiter als
          hoch stützt die Sturz-Einschätzung zusätzlich), nicht zwingend.
    angle_threshold: ab welchem Winkel (Grad von der Vertikalen) als Sturz
          gilt -- einstellbar statt hart codiert, da unkalibriert.

    Gibt ein Dict zurück: {"fall_detected": bool, "angle": float|None,
    "confidence": "low"|"medium"|"high", "reason": str} -- "confidence" ist
    KEINE trainierte Wahrscheinlichkeit, sondern ein grobes Signal, wie viele
    unabhängige Indizien übereinstimmen (Winkel + Seitenverhältnis)."""
    angle = torso_angle_from_vertical(keypoints)
    if angle is None:
        return {"fall_detected": False, "angle": None, "confidence": "low",
                "reason": "Not enough confidently-detected keypoints (shoulders/hips) to judge."}

    angle_says_fall = angle >= angle_threshold
    bbox_says_fall = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        if height > 0:
            bbox_says_fall = (width / height) >= 1.2  # breiter als hoch

    if not angle_says_fall:
        return {"fall_detected": False, "angle": angle, "confidence": "medium",
                "reason": f"Torso angle {angle:.1f}° is within normal standing/sitting range."}

    if bbox_says_fall is None:
        confidence = "medium"
        reason = f"Torso angle {angle:.1f}° exceeds threshold ({angle_threshold}°); no bounding box to cross-check."
    elif bbox_says_fall:
        confidence = "high"
        reason = f"Torso angle {angle:.1f}° exceeds threshold, and the bounding box is wider than tall -- both signals agree."
    else:
        confidence = "medium"
        reason = f"Torso angle {angle:.1f}° exceeds threshold, but the bounding box shape doesn't confirm it -- could be a bend, not a fall."

    return {"fall_detected": True, "angle": angle, "confidence": confidence, "reason": reason}


class FallTracker:
    """Verlangt mehrere aufeinanderfolgende 'liegt'-Einschätzungen, bevor ein
    Sturz tatsächlich gemeldet wird -- ein einzelner Frame reicht nicht,
    sonst würde sich kurzes Bücken (Schuhe binden, etwas aufheben) genauso
    anfühlen wie ein Sturz, da beides für EINEN Frame ähnlich aussieht. Ein
    echter Sturz bleibt dagegen typischerweise mehrere Sekunden liegen --
    genau dieser zeitliche Unterschied wird hier genutzt, nicht nur die Pose
    selbst. Pro Kamera eine eigene Instanz nötig (Zustand ist nicht global)."""

    def __init__(self, required_consecutive=5):
        self.required_consecutive = required_consecutive
        self._consecutive_count = 0
        self._confirmed = False

    def update(self, fall_result):
        """Mit dem Ergebnis von detect_fall() für den aktuellen Frame
        aufrufen. Gibt True zurück, GENAU in dem Moment, in dem ein Sturz
        neu bestätigt wird (nicht bei jedem weiteren Frame danach erneut --
        die aufrufende Seite soll nur einmal benachrichtigt werden)."""
        if fall_result.get("fall_detected"):
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
