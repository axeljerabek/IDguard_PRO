#!/usr/bin/env python3
"""
cluster_faces.py - Gruppiert alle noch nicht zugeordneten Gesichter per
DBSCAN. Läuft on-demand (Dashboard-Button "Re-cluster faces") oder als Cron,
NIE automatisch nach jeder einzelnen Aufnahme (DBSCAN über die komplette
Menge macht bei wenigen neuen Gesichtern keinen Sinn und wäre teuer, wenn es
nach jedem Event neu berechnet würde).

Cosine-Distanz statt euklidisch, weil die Embeddings bereits L2-normalisiert
sind (InsightFace liefert das so) — für normalisierte Vektoren ist Cosine-
Ähnlichkeit äquivalent zum Skalarprodukt, DBSCAN mit metric='cosine' passt
direkt.

Aufruf: python3 cluster_faces.py [--eps 0.4] [--min-samples 2]
"""
import sys
import os
import argparse
import numpy as np
from sklearn.cluster import DBSCAN

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
import faces_db

# eps ist eine Cosine-DISTANZ (1 - Ähnlichkeit), kein Winkel — 0.4 heißt
# "mindestens 60% Cosine-Ähnlichkeit, um als dieselbe Person zu gelten".
# Das ist ein Startwert, kein für alle Gesichter/Beleuchtungen validierter
# Wert; je nach eurer Kamera-Bildqualität ggf. anpassen.
DEFAULT_EPS = 0.4
DEFAULT_MIN_SAMPLES = 2

# Ab dieser Cosine-Ähnlichkeit gilt ein unzugeordnetes Gesicht als "eindeutig
# dieselbe Person" wie ein bereits bekannter Centroid — bewusst strenger als
# der DBSCAN-eps-Wert, da eine automatische Zuordnung zu einem NAMEN eine
# höhere Sicherheit verdient als eine reine Cluster-Vorschlag-Gruppierung.
KNOWN_PERSON_THRESHOLD = 0.5


def match_against_known_people(embedding, centroids):
    """Vergleicht ein Embedding gegen alle bekannten Personen-Centroide.
    Gibt (person_id, similarity) der besten Übereinstimmung zurück, oder
    (None, 0.0) falls keine über der Schwelle liegt."""
    best_id, best_sim = None, 0.0
    for person_id, (_, centroid) in centroids.items():
        sim = float(np.dot(embedding, centroid))  # beide normalisiert -> Cosine == Skalarprodukt
        if sim > best_sim:
            best_id, best_sim = person_id, sim
    if best_sim >= KNOWN_PERSON_THRESHOLD:
        return best_id, best_sim
    return None, 0.0


def run(eps=DEFAULT_EPS, min_samples=DEFAULT_MIN_SAMPLES):
    # Schritt 1: unzugeordnete Gesichter nochmal gegen inzwischen evtl. neu
    # benannte Personen prüfen — jemand könnte seit dem letzten Lauf benannt
    # worden sein, dann sollten dessen übrige Gesichter direkt zugeordnet
    # werden, statt im Cluster-Pool zu bleiben.
    centroids = faces_db.get_person_centroids()
    unassigned = faces_db.get_unassigned_faces()

    still_unassigned = []
    auto_matched = 0
    if centroids:
        for face in unassigned:
            person_id, sim = match_against_known_people(face["embedding"], centroids)
            if person_id is not None:
                faces_db.assign_faces_to_person(person_id, [face["id"]])
                auto_matched += 1
            else:
                still_unassigned.append(face)
    else:
        still_unassigned = unassigned

    if auto_matched:
        print(f"✅ {auto_matched} Gesicht(er) automatisch bereits bekannten Personen zugeordnet.")

    if len(still_unassigned) < min_samples:
        print(f"ℹ️ Nur noch {len(still_unassigned)} unzugeordnete Gesichter — zu wenig für ein Clustering (min_samples={min_samples}).")
        return

    # Schritt 2: DBSCAN über den Rest
    embeddings = np.array([f["embedding"] for f in still_unassigned])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_

    face_cluster_map = {f["id"]: int(label) for f, label in zip(still_unassigned, labels)}
    faces_db.set_cluster_ids(face_cluster_map)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"✅ Clustering fertig: {n_clusters} Gruppe(n) gefunden, {n_noise} Gesicht(er) ohne klare Gruppe (Rauschen).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gruppiert unzugeordnete Gesichter per DBSCAN.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS,
                         help=f"Cosine-Distanz-Schwelle (Standard: {DEFAULT_EPS} = 60%% Mindest-Ähnlichkeit)")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                         help=f"Mindestanzahl Gesichter pro Gruppe (Standard: {DEFAULT_MIN_SAMPLES})")
    args = parser.parse_args()
    run(eps=args.eps, min_samples=args.min_samples)
