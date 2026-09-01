"""
faces_db.py - SQLite-Speicher für erkannte Gesichter, geclusterte Gruppen und
benannte Personen. Eigene Datenbank (faces.db), getrennt von search_index.db.

Modell: face_recognize.py erkennt Gesichter + Embeddings pro Aufnahme und
versucht SOFORT eine Zuordnung zu bereits benannten Personen (Vergleich
gegen deren Centroid-Embedding). Was dabei nicht zugeordnet werden kann,
bleibt "unassigned" bis cluster_faces.py (DBSCAN) es in Gruppen einteilt,
die der Nutzer im Dashboard benennt oder korrigiert.
"""
import os
import sqlite3
import struct
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, "faces.db")

EMBEDDING_DIM = 512  # InsightFace buffalo_*/antelopev2 liefern alle 512-dim Embeddings


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            centroid BLOB,
            representative_face_id INTEGER,
            created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            base_dir TEXT NOT NULL,
            crop_path TEXT NOT NULL,
            embedding BLOB NOT NULL,
            detection_confidence REAL,
            person_id INTEGER,
            cluster_id INTEGER,
            rejected INTEGER DEFAULT 0,
            created_at REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id)")
    return conn


def _pack(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob):
    n = len(blob) // 4
    return struct.unpack(f"{n}f", blob)


def get_faces_summary_for_video(filename):
    """Für die Event-Karten in Recent/Archived: alle Gesichter DIESES
    Videos, benannte Personen (mit der konkreten Gesichts-ID aus diesem
    Video, nicht dem allgemeinen Titelbild der Person) getrennt von der
    reinen Anzahl noch unbenannter Erkennungen."""
    try:
        conn = _connect()
        named = conn.execute("""
            SELECT p.id, p.name, MIN(f.id) as face_id
            FROM faces f JOIN people p ON p.id = f.person_id
            WHERE f.filename = ? AND f.rejected = 0
            GROUP BY p.id, p.name
        """, (filename,)).fetchall()
        unnamed_count = conn.execute("""
            SELECT COUNT(*) FROM faces
            WHERE filename = ? AND rejected = 0 AND person_id IS NULL
        """, (filename,)).fetchone()[0]
        conn.close()
        return {
            "people": [{"id": r[0], "name": r[1], "face_id": r[2]} for r in named],
            "unnamed_count": unnamed_count
        }
    except Exception as e:
        print(f"⚠️ Konnte Gesichts-Zusammenfassung für Video {filename} nicht laden: {e}")
        return {"people": [], "unnamed_count": 0}


def update_base_dir(filename, new_base_dir):
    """Beim Archivieren aufrufen. Die Crop-Bilder wandern physisch mit (sie
    liegen im selben .thumbs/<basename>/-Ordner, der beim Archivieren
    komplett verschoben wird) — aber ohne diesen Aufruf würde /face_crop/<id>
    weiterhin den ALTEN Pfad (ALERTS_DIR) versuchen und ins Leere laufen,
    weil die Datei dort nicht mehr liegt. Spiegelt search_index.py's
    update_location() für denselben Anwendungsfall."""
    try:
        conn = _connect()
        conn.execute("UPDATE faces SET base_dir = ? WHERE filename = ?", (new_base_dir, filename))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Konnte base_dir für Gesichter von {filename} nicht aktualisieren: {e}")


def remove_faces_for_video(filename):
    """Beim endgültigen Löschen eines Videos aufrufen. Sonst blieben
    verwaiste Gesichts-Datensätze (mit Verweis auf inzwischen gelöschte
    Crop-Bilder) für immer in der Datenbank — u.U. sogar sichtbar im
    unzugeordneten Cluster-Pool, für ein Video, das gar nicht mehr
    existiert. Betroffene Personen-Centroide werden danach neu berechnet,
    falls eines der gelöschten Gesichter einer Person zugeordnet war."""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT DISTINCT person_id FROM faces WHERE filename = ? AND person_id IS NOT NULL", (filename,)
        ).fetchall()
        affected_people = [r[0] for r in rows]
        conn.execute("DELETE FROM faces WHERE filename = ?", (filename,))
        conn.commit()
        conn.close()
        for person_id in affected_people:
            _recompute_centroid(person_id)
    except Exception as e:
        print(f"⚠️ Konnte Gesichter für {filename} nicht löschen: {e}")


def get_face(face_id):
    """Ein einzelnes Gesicht per ID nachschlagen — z.B. um dessen Bild-Datei
    auszuliefern, ohne dass der Aufrufer die interne DB-Verbindung anfassen muss."""
    try:
        conn = _connect()
        row = conn.execute("SELECT base_dir, crop_path FROM faces WHERE id = ?", (face_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return {"base_dir": row[0], "crop_path": row[1]}
    except Exception as e:
        print(f"⚠️ Konnte Gesicht {face_id} nicht laden: {e}")
        return None


def add_face(filename, base_dir, crop_path, embedding, confidence, person_id=None):
    """Speichert ein neu erkanntes Gesicht. Falls person_id direkt mitgegeben
    wird (automatische Zuordnung zu einer bekannten Person beim Erkennen),
    wird gleich zugeordnet, sonst bleibt es 'unassigned' fürs Clustering."""
    try:
        conn = _connect()
        cur = conn.execute(
            "INSERT INTO faces (filename, base_dir, crop_path, embedding, detection_confidence, "
            "person_id, created_at) VALUES (?, ?, ?, ?, ?, ?, strftime('%s','now'))",
            (filename, base_dir, crop_path, _pack(embedding), float(confidence), person_id)
        )
        face_id = cur.lastrowid
        conn.commit()
        conn.close()
        if person_id is not None:
            _recompute_centroid(person_id)
        return face_id
    except Exception as e:
        print(f"⚠️ Konnte Gesicht nicht speichern: {e}")
        return None


def get_person_centroids():
    """{person_id: (name, embedding_als_numpy_array)} — nur Personen mit
    mindestens einem zugeordneten Gesicht (centroid IS NOT NULL)."""
    try:
        conn = _connect()
        rows = conn.execute("SELECT id, name, centroid FROM people WHERE centroid IS NOT NULL").fetchall()
        conn.close()
        return {pid: (name, np.array(_unpack(centroid), dtype=np.float32)) for pid, name, centroid in rows}
    except Exception as e:
        print(f"⚠️ Konnte Personen-Centroide nicht laden: {e}")
        return {}


def _recompute_centroid(person_id):
    """Durchschnitts-Embedding aller (nicht abgelehnten) Gesichter dieser
    Person neu berechnen — nach jeder Zuordnungsänderung aufgerufen."""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT embedding FROM faces WHERE person_id = ? AND rejected = 0", (person_id,)
        ).fetchall()
        if not rows:
            conn.execute("UPDATE people SET centroid = NULL WHERE id = ?", (person_id,))
        else:
            vecs = np.array([_unpack(r[0]) for r in rows], dtype=np.float32)
            centroid = vecs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)  # normalisieren, für Cosine-Vergleich
            conn.execute("UPDATE people SET centroid = ? WHERE id = ?", (_pack(centroid.tolist()), person_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Konnte Centroid für Person {person_id} nicht neu berechnen: {e}")


def create_person(name, face_ids):
    """Neue Person aus einer Menge von Gesichts-IDs (z.B. beim Benennen
    eines Clusters). Erste Face-ID wird als Titelbild verwendet."""
    if not face_ids:
        return None
    try:
        conn = _connect()
        cur = conn.execute(
            "INSERT INTO people (name, representative_face_id, created_at) VALUES (?, ?, strftime('%s','now'))",
            (name, face_ids[0])
        )
        person_id = cur.lastrowid
        conn.executemany(
            "UPDATE faces SET person_id = ?, cluster_id = NULL WHERE id = ?",
            [(person_id, fid) for fid in face_ids]
        )
        conn.commit()
        conn.close()
        _recompute_centroid(person_id)
        return person_id
    except Exception as e:
        print(f"⚠️ Konnte Person nicht anlegen: {e}")
        return None


def assign_faces_to_person(person_id, face_ids):
    """Bestehende Gesichter (z.B. aus einem Cluster) einer bereits benannten
    Person zuordnen — 'Merge in bestehende Person'."""
    if not face_ids:
        return
    try:
        conn = _connect()
        conn.executemany(
            "UPDATE faces SET person_id = ?, cluster_id = NULL WHERE id = ?",
            [(person_id, fid) for fid in face_ids]
        )
        conn.commit()
        conn.close()
        _recompute_centroid(person_id)
    except Exception as e:
        print(f"⚠️ Konnte Gesichter nicht zuordnen: {e}")


def unassign_face(face_id):
    """Ein einzelnes Gesicht aus seiner Person herauslösen (Korrektur einer
    Fehlzuordnung) — geht zurück in den unzugeordneten Pool."""
    try:
        conn = _connect()
        row = conn.execute("SELECT person_id FROM faces WHERE id = ?", (face_id,)).fetchone()
        old_person_id = row[0] if row else None
        conn.execute("UPDATE faces SET person_id = NULL, cluster_id = NULL WHERE id = ?", (face_id,))
        conn.commit()
        conn.close()
        if old_person_id is not None:
            _recompute_centroid(old_person_id)
    except Exception as e:
        print(f"⚠️ Konnte Gesicht nicht lösen: {e}")


def unassign_faces(face_ids):
    """Bulk-Variante von unassign_face() — z.B. zum schnellen Aufräumen
    eines großen unzugeordneten Pools. Ein DB-Write für alle IDs auf
    einmal, Centroid-Neuberechnung pro betroffener Person nur EINMAL,
    nicht pro einzelnem Gesicht."""
    if not face_ids:
        return
    try:
        conn = _connect()
        placeholders = ','.join('?' * len(face_ids))
        rows = conn.execute(
            f"SELECT DISTINCT person_id FROM faces WHERE id IN ({placeholders}) AND person_id IS NOT NULL",
            face_ids
        ).fetchall()
        affected_people = [r[0] for r in rows]
        conn.execute(f"UPDATE faces SET person_id = NULL, cluster_id = NULL WHERE id IN ({placeholders})", face_ids)
        conn.commit()
        conn.close()
        for person_id in affected_people:
            _recompute_centroid(person_id)
    except Exception as e:
        print(f"⚠️ Konnte Gesichter nicht lösen: {e}")


def reject_face(face_id):
    """Nutzer markiert: das war gar kein Gesicht (Fehlerkennung) — zählt
    nirgends mehr mit, taucht auch nicht mehr im Clustering auf."""
    try:
        conn = _connect()
        row = conn.execute("SELECT person_id FROM faces WHERE id = ?", (face_id,)).fetchone()
        old_person_id = row[0] if row else None
        conn.execute("UPDATE faces SET rejected = 1, person_id = NULL, cluster_id = NULL WHERE id = ?", (face_id,))
        conn.commit()
        conn.close()
        if old_person_id is not None:
            _recompute_centroid(old_person_id)
    except Exception as e:
        print(f"⚠️ Konnte Gesicht nicht ablehnen: {e}")


def reject_faces(face_ids):
    """Bulk-Variante von reject_face()."""
    if not face_ids:
        return
    try:
        conn = _connect()
        placeholders = ','.join('?' * len(face_ids))
        rows = conn.execute(
            f"SELECT DISTINCT person_id FROM faces WHERE id IN ({placeholders}) AND person_id IS NOT NULL",
            face_ids
        ).fetchall()
        affected_people = [r[0] for r in rows]
        conn.execute(f"UPDATE faces SET rejected = 1, person_id = NULL, cluster_id = NULL WHERE id IN ({placeholders})", face_ids)
        conn.commit()
        conn.close()
        for person_id in affected_people:
            _recompute_centroid(person_id)
    except Exception as e:
        print(f"⚠️ Konnte Gesichter nicht ablehnen: {e}")


def get_unassigned_faces():
    """Alle Gesichter, die weder einer Person zugeordnet noch abgelehnt
    wurden — Grundlage für den nächsten Clustering-Lauf."""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, filename, base_dir, crop_path, embedding FROM faces "
            "WHERE person_id IS NULL AND rejected = 0"
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "filename": r[1], "base_dir": r[2], "crop_path": r[3],
             "embedding": np.array(_unpack(r[4]), dtype=np.float32)}
            for r in rows
        ]
    except Exception as e:
        print(f"⚠️ Konnte unzugeordnete Gesichter nicht laden: {e}")
        return []


def set_cluster_ids(face_cluster_map):
    """face_cluster_map: {face_id: cluster_label}. cluster_label == -1
    (DBSCAN-Rauschen, keine Gruppe gefunden) wird als NULL gespeichert."""
    try:
        conn = _connect()
        conn.executemany(
            "UPDATE faces SET cluster_id = ? WHERE id = ?",
            [(int(c) if c != -1 else None, fid) for fid, c in face_cluster_map.items()]
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Konnte Cluster-IDs nicht speichern: {e}")


def list_clusters():
    """Alle aktuellen (unbenannten) Cluster mit ihren Gesichtern, gruppiert."""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, cluster_id, filename, base_dir, crop_path FROM faces "
            "WHERE cluster_id IS NOT NULL AND person_id IS NULL AND rejected = 0 "
            "ORDER BY cluster_id"
        ).fetchall()
        conn.close()
        clusters = {}
        for face_id, cluster_id, filename, base_dir, crop_path in rows:
            clusters.setdefault(cluster_id, []).append({
                "id": face_id, "filename": filename, "base_dir": base_dir, "crop_path": crop_path
            })
        return clusters
    except Exception as e:
        print(f"⚠️ Konnte Cluster nicht laden: {e}")
        return {}


def list_people():
    """Alle benannten Personen mit Gesichts-Anzahl + Titelbild-Pfad."""
    try:
        conn = _connect()
        rows = conn.execute("""
            SELECT p.id, p.name, p.representative_face_id, f.crop_path,
                   (SELECT COUNT(*) FROM faces WHERE person_id = p.id AND rejected = 0) as face_count
            FROM people p
            LEFT JOIN faces f ON f.id = p.representative_face_id
            ORDER BY p.name
        """).fetchall()
        conn.close()
        return [
            {"id": r[0], "name": r[1], "representative_face_id": r[2], "crop_path": r[3], "face_count": r[4]}
            for r in rows
        ]
    except Exception as e:
        print(f"⚠️ Konnte Personen-Liste nicht laden: {e}")
        return []


def get_faces_for_person(person_id):
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, filename, base_dir, crop_path FROM faces WHERE person_id = ? AND rejected = 0",
            (person_id,)
        ).fetchall()
        conn.close()
        return [{"id": r[0], "filename": r[1], "base_dir": r[2], "crop_path": r[3]} for r in rows]
    except Exception as e:
        print(f"⚠️ Konnte Gesichter für Person {person_id} nicht laden: {e}")
        return []


def rename_person(person_id, new_name):
    try:
        conn = _connect()
        conn.execute("UPDATE people SET name = ? WHERE id = ?", (new_name, person_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Konnte Person nicht umbenennen: {e}")


def delete_person(person_id):
    """Löscht die Person selbst; ihre Gesichter bleiben erhalten, werden aber
    unzugeordnet (landen wieder im Clustering-Pool statt gelöscht zu werden)."""
    try:
        conn = _connect()
        conn.execute("UPDATE faces SET person_id = NULL WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Konnte Person nicht löschen: {e}")
