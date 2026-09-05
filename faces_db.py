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
import shutil
import sqlite3
import struct
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, "faces.db")

try:
    from config import ALERTS_DIR
except ImportError:
    ALERTS_DIR = os.path.join(DIR, "alerts")

# Permanenter Foto-Speicher für benannte Personen -- UNABHÄNGIG vom Ordner
# des Quellvideos. Vorher lagen Gesichts-Crops nur im videospezifischen
# Ordner (base_dir/crop_path) -- wurde das Video gelöscht, verschwand damit
# auch das Foto UND (schlimmer) die Zeile in der faces-Tabelle, was
# _recompute_centroid() dazu brachte, den Centroid einer Person komplett auf
# NULL zu setzen, sobald keine ihrer Quellvideos mehr existierten. Damit
# war die Person zwar noch benannt, aber nie wieder automatisch erkennbar --
# genau das Problem, das dieser permanente Speicher behebt: sobald ein
# Gesicht einer Person zugeordnet wird, wird sein Foto hierher kopiert und
# die Datenbank-Zeile bleibt bestehen, auch wenn das Quellvideo später
# gelöscht wird.
PEOPLE_PHOTOS_DIR = os.path.join(ALERTS_DIR, ".people_photos")


def _archive_face_photo(base_dir, crop_path, face_id):
    """Kopiert das Foto eines Gesichts in den permanenten Personen-Speicher,
    falls es dort nicht schon liegt. Gibt (neuer_base_dir, neuer_crop_path)
    zurück -- im selben base_dir/crop_path-Format wie der Rest der Tabelle,
    damit get_face() unverändert funktioniert. Bei einem Fehler (z.B. Quelle
    schon weg) wird (base_dir, crop_path) unverändert zurückgegeben -- lieber
    die alte, möglicherweise brüchige Referenz behalten als abzustürzen."""
    try:
        os.makedirs(PEOPLE_PHOTOS_DIR, exist_ok=True)
        source_path = os.path.join(base_dir, crop_path)
        if not os.path.exists(source_path):
            return base_dir, crop_path  # nichts zu kopieren, alte Referenz behalten
        ext = os.path.splitext(crop_path)[1] or ".jpg"
        new_filename = f"face_{face_id}{ext}"
        dest_path = os.path.join(PEOPLE_PHOTOS_DIR, new_filename)
        if os.path.abspath(source_path) == os.path.abspath(dest_path):
            return PEOPLE_PHOTOS_DIR, new_filename  # schon archiviert
        shutil.copy2(source_path, dest_path)
        return PEOPLE_PHOTOS_DIR, new_filename
    except Exception as e:
        print(f"⚠️ Konnte Gesichtsfoto {face_id} nicht permanent archivieren: {e}")
        return base_dir, crop_path

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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ignored_clusters (
            cluster_id INTEGER PRIMARY KEY,
            ignored_at REAL
        )
    """)
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
    """Beim endgültigen Löschen eines Videos aufrufen. Wichtig: Gesichter,
    die bereits einer BENANNTEN Person zugeordnet sind (person_id IS NOT
    NULL), werden NICHT gelöscht -- ihr Foto wurde beim Zuordnen bereits
    permanent archiviert (siehe create_person/assign_faces_to_person), aber
    das eigentliche Embedding lebt in genau dieser Zeile. Würde sie hier mit
    gelöscht, verlöre die Person ihr Wiedererkennungs-Embedding, sobald alle
    ihre Quellvideos gelöscht sind -- sie bliebe zwar benannt, würde aber nie
    wieder automatisch erkannt (das war der eigentliche Bug). Nur
    unzugeordnete/Cluster-Gesichter (person_id IS NULL) werden wie bisher
    entfernt, die waren ohnehin nie einer Identität zugewiesen."""
    try:
        conn = _connect()
        conn.execute("DELETE FROM faces WHERE filename = ? AND person_id IS NULL", (filename,))
        conn.commit()
        conn.close()
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
    eines Clusters). Erste Face-ID wird als Titelbild verwendet. Alle
    zugeordneten Fotos werden sofort permanent archiviert (siehe
    PEOPLE_PHOTOS_DIR) -- ab diesem Moment übersteht die Person das
    Löschen ihrer Quellvideos."""
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
        for fid in face_ids:
            row = conn.execute("SELECT base_dir, crop_path FROM faces WHERE id = ?", (fid,)).fetchone()
            if row:
                new_base_dir, new_crop_path = _archive_face_photo(row[0], row[1], fid)
                conn.execute("UPDATE faces SET base_dir = ?, crop_path = ? WHERE id = ?", (new_base_dir, new_crop_path, fid))
        conn.commit()
        conn.close()
        _recompute_centroid(person_id)
        return person_id
    except Exception as e:
        print(f"⚠️ Konnte Person nicht anlegen: {e}")
        return None


def assign_faces_to_person(person_id, face_ids):
    """Bestehende Gesichter (z.B. aus einem Cluster) einer bereits benannten
    Person zuordnen — 'Merge in bestehende Person'. Fotos werden dabei
    ebenfalls permanent archiviert, siehe create_person()."""
    if not face_ids:
        return
    try:
        conn = _connect()
        conn.executemany(
            "UPDATE faces SET person_id = ?, cluster_id = NULL WHERE id = ?",
            [(person_id, fid) for fid in face_ids]
        )
        conn.commit()
        for fid in face_ids:
            row = conn.execute("SELECT base_dir, crop_path FROM faces WHERE id = ?", (fid,)).fetchone()
            if row:
                new_base_dir, new_crop_path = _archive_face_photo(row[0], row[1], fid)
                conn.execute("UPDATE faces SET base_dir = ?, crop_path = ? WHERE id = ?", (new_base_dir, new_crop_path, fid))
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


def find_orphaned_faces():
    """Gesichter, deren Crop-Bild-Datei physisch nicht mehr existiert —
    typischerweise Altlasten aus einer Zeit, bevor Archivieren/Löschen
    korrekt mit der Datenbank synchron gehalten wurden (siehe
    update_base_dir/remove_faces_for_video). Betrifft nur Datensätze von
    VOR diesem Fix — neue Archivierungen/Löschungen halten base_dir schon
    korrekt aktuell."""
    try:
        conn = _connect()
        rows = conn.execute("SELECT id, base_dir, crop_path FROM faces WHERE rejected = 0").fetchall()
        conn.close()
        orphaned = []
        for face_id, base_dir, crop_path in rows:
            full_path = os.path.join(base_dir, crop_path)
            if not os.path.exists(full_path):
                orphaned.append(face_id)
        return orphaned
    except Exception as e:
        print(f"⚠️ Konnte verwaiste Gesichter nicht ermitteln: {e}")
        return []


def remove_orphaned_faces():
    """Entfernt Gesichter, deren Crop-Bild nicht mehr existiert, endgültig
    aus der Datenbank — nicht nur als 'rejected' markieren wie bei
    reject_faces(), denn es gibt hier schlicht kein Bild mehr, das man
    sich je wieder ansehen könnte. Gibt die Anzahl der entfernten
    Einträge zurück."""
    orphaned_ids = find_orphaned_faces()
    if not orphaned_ids:
        return 0
    try:
        conn = _connect()
        placeholders = ','.join('?' * len(orphaned_ids))
        rows = conn.execute(
            f"SELECT DISTINCT person_id FROM faces WHERE id IN ({placeholders}) AND person_id IS NOT NULL",
            orphaned_ids
        ).fetchall()
        affected_people = [r[0] for r in rows]
        conn.execute(f"DELETE FROM faces WHERE id IN ({placeholders})", orphaned_ids)
        conn.commit()
        conn.close()
        for person_id in affected_people:
            _recompute_centroid(person_id)
        return len(orphaned_ids)
    except Exception as e:
        print(f"⚠️ Konnte verwaiste Gesichter nicht entfernen: {e}")
        return 0


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
    """Alle aktuellen (unbenannten) Cluster mit ihren Gesichtern, gruppiert.
    Jeder Cluster trägt zusätzlich ein 'ignored'-Flag -- ignorierte Cluster
    werden NICHT weggelassen (das Frontend entscheidet per Standard-Filter
    + "Show ignored clusters"-Umschalter, was angezeigt wird), damit ein
    einziger Aufruf für beide Ansichten reicht."""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, cluster_id, filename, base_dir, crop_path FROM faces "
            "WHERE cluster_id IS NOT NULL AND person_id IS NULL AND rejected = 0 "
            "ORDER BY cluster_id"
        ).fetchall()
        ignored_ids = {r[0] for r in conn.execute("SELECT cluster_id FROM ignored_clusters").fetchall()}
        conn.close()
        clusters = {}
        for face_id, cluster_id, filename, base_dir, crop_path in rows:
            if cluster_id not in clusters:
                clusters[cluster_id] = {"faces": [], "ignored": cluster_id in ignored_ids}
            clusters[cluster_id]["faces"].append({
                "id": face_id, "filename": filename, "base_dir": base_dir, "crop_path": crop_path
            })
        return clusters
    except Exception as e:
        print(f"⚠️ Konnte Cluster nicht laden: {e}")
        return {}


def ignore_cluster(cluster_id):
    """Blendet einen Cluster aus der Standardansicht aus (nur noch über
    'Show ignored clusters' sichtbar) -- löscht keine Gesichter, rein
    kosmetisch/organisatorisch, jederzeit über unignore_cluster() rückgängig
    zu machen."""
    try:
        conn = _connect()
        import time
        conn.execute(
            "INSERT OR REPLACE INTO ignored_clusters (cluster_id, ignored_at) VALUES (?, ?)",
            (cluster_id, time.time())
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"⚠️ Konnte Cluster {cluster_id} nicht ignorieren: {e}")
        return False


def unignore_cluster(cluster_id):
    try:
        conn = _connect()
        conn.execute("DELETE FROM ignored_clusters WHERE cluster_id = ?", (cluster_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"⚠️ Konnte Cluster {cluster_id} nicht zurückholen: {e}")
        return False


def list_people():
    """Alle benannten Personen mit Gesichts-Anzahl + Titelbild-Pfad.

    Prüft, ob die Titelbild-Datei tatsächlich noch existiert — z.B. wenn das
    zugehörige Video (und damit sein .thumbs-Ordner) längst gelöscht wurde,
    aber die Personen-Zuordnung selbst noch besteht. Fällt in dem Fall
    automatisch auf das nächste noch existierende Gesicht derselben Person
    zurück, statt ein kaputtes Bild im Dashboard anzuzeigen — und aktualisiert
    representative_face_id gleich dauerhaft in der DB (selbstheilend, kein
    wiederholtes Nachprüfen bei jedem künftigen Aufruf nötig)."""
    try:
        conn = _connect()
        rows = conn.execute("""
            SELECT p.id, p.name, p.representative_face_id, f.crop_path, f.base_dir,
                   (SELECT COUNT(*) FROM faces WHERE person_id = p.id AND rejected = 0) as face_count
            FROM people p
            LEFT JOIN faces f ON f.id = p.representative_face_id
            ORDER BY p.name
        """).fetchall()

        result = []
        for r in rows:
            person_id, name, rep_face_id, crop_path, base_dir, face_count = r
            full_path = os.path.join(base_dir, crop_path) if (base_dir and crop_path) else None
            needs_fallback = (not full_path) or (not os.path.exists(full_path))
            if needs_fallback:
                # Titelbild-Datei fehlt ODER representative_face_id zeigt auf
                # eine Face-ID, die in der faces-Tabelle gar nicht mehr existiert
                # (z.B. komplett entfernt statt nur rejected) -- in BEIDEN Fällen
                # nächstes noch existierendes Gesicht derselben Person suchen.
                candidates = conn.execute(
                    "SELECT id, crop_path, base_dir FROM faces WHERE person_id = ? AND rejected = 0",
                    (person_id,)
                ).fetchall()
                rep_face_id, crop_path, base_dir = None, None, None
                for cand_id, cand_crop, cand_base in candidates:
                    cand_full = os.path.join(cand_base, cand_crop) if (cand_base and cand_crop) else None
                    if cand_full and os.path.exists(cand_full):
                        rep_face_id, crop_path, base_dir = cand_id, cand_crop, cand_base
                        conn.execute(
                            "UPDATE people SET representative_face_id = ? WHERE id = ?",
                            (cand_id, person_id)
                        )
                        break
            result.append({
                "id": person_id, "name": name, "representative_face_id": rep_face_id,
                "crop_path": crop_path, "face_count": face_count
            })
        conn.commit()
        conn.close()
        return result
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


def set_representative_face(person_id, face_id):
    """Legt fest, welches Foto als Titelbild der Person angezeigt wird --
    muss ein Gesicht SEIN, das bereits dieser Person zugeordnet ist (sonst
    wird ein Foto verwendet, das gar nicht zu ihr gehört). Da Fotos benannter
    Personen beim Zuordnen bereits permanent archiviert werden, ist jede
    Auswahl hier dauerhaft sicher vor dem Löschen von Quellvideos."""
    try:
        conn = _connect()
        row = conn.execute("SELECT person_id FROM faces WHERE id = ?", (face_id,)).fetchone()
        if not row or row[0] != person_id:
            conn.close()
            return False, "This face doesn't belong to this person."
        conn.execute("UPDATE people SET representative_face_id = ? WHERE id = ?", (face_id, person_id))
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        print(f"⚠️ Konnte Titelbild nicht setzen: {e}")
        return False, str(e)


def delete_person(person_id):
    """Löscht die Person selbst; ihre Gesichter bleiben erhalten, werden aber
    unzugeordnet (landen wieder im Clustering-Pool statt gelöscht zu werden).
    Das ist die 'Un-name'-Aktion im Dashboard -- für Korrekturen (falsch
    benannt), nicht zum endgültigen Entsorgen. Siehe delete_person_permanently()
    für Letzteres."""
    try:
        conn = _connect()
        conn.execute("UPDATE faces SET person_id = NULL WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Konnte Person nicht löschen: {e}")


def delete_person_permanently(person_id):
    """Löscht eine Person UND alle ihre Gesichts-Daten unwiderruflich --
    Datenbank-Zeilen, Embeddings, UND die permanent archivierten Fotos auf
    der Platte. Anders als delete_person() (= sanftes 'Un-name', Gesichter
    bleiben für spätere Neuzuordnung erhalten) ist das hier endgültig --
    für Personen, die wirklich nicht mehr im System gebraucht werden."""
    try:
        conn = _connect()
        rows = conn.execute("SELECT base_dir, crop_path FROM faces WHERE person_id = ?", (person_id,)).fetchall()
        conn.execute("DELETE FROM faces WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        conn.commit()
        conn.close()
        people_photos_abs = os.path.abspath(PEOPLE_PHOTOS_DIR)
        for base_dir, crop_path in rows:
            full_path = os.path.abspath(os.path.join(base_dir, crop_path))
            # Sicherheitsnetz: nur Dateien löschen, die wirklich im
            # permanenten Foto-Ordner liegen -- niemals versehentlich eine
            # Datei innerhalb eines noch existierenden Video-Ordners
            # anfassen, die gehört dem Video, nicht der Personenverwaltung.
            try:
                if os.path.commonpath([full_path, people_photos_abs]) == people_photos_abs:
                    os.remove(full_path)
            except Exception:
                pass
        return True, None
    except Exception as e:
        print(f"⚠️ Konnte Person nicht endgültig löschen: {e}")
        return False, str(e)
