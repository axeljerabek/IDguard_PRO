import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes the database schema."""
        os_dir = os.path.dirname(self.db_path)
        if not os.path.exists(os_dir):
            os.makedirs(os_dir, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table for tracking all detection events
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    stream_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    video_path TEXT,
                    status TEXT DEFAULT 'recorded'
                )
            ''')
            conn.commit()

    def log_alert(self, stream_name, event_type, video_path):
        """Logs a new alert event into the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts (stream_name, event_type, video_path)
                    VALUES (?, ?, ?)
                ''', (stream_name, event_type, video_path))
                conn.commit()
            return True
        except Exception as e:
            print(f"❌ [DB_ERROR] Failed to log alert: {e}")
            return False

    def get_all_alerts(self):
        """Retrieates all logged alerts."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
                return cursor.fetchall()
        except Exception as e:
            print(f"❌ [DB_ERROR] Failed to retrieve alerts: {e}")
            return []

if __name__ == "__main__":
    # Test run logic
    test_db = "/opt/data/IDguard_PRO_FINAL/database/alerts_test.db"
    db = DatabaseManager(test_db)
    success = db.log_alert("Test-Stream", "TEST_EVENT", "/tmp/test.mp4")
    print(f"✅ DB Test successful: {success}")
