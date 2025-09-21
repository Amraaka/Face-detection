from __future__ import annotations

from typing import Optional
from uuid import uuid4
from datetime import datetime, timezone
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import PyMongoError


class StatsDB:
    def __init__(self, uri: str, db_name: str = "driver_monitor", coll_name: str = "stats"):
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        # quick ping to ensure connectivity (mirrors testdb.py style)
        self.client.admin.command('ping')
        self.db = self.client[db_name]
        self.coll = self.db[coll_name]

    def insert(self, doc: dict) -> Optional[str]:
        try:
            res = self.coll.insert_one(doc)
            return str(res.inserted_id)
        except PyMongoError as e:
            print(f"[warn] Mongo insert failed: {e}")
            return None


class TimeSeriesStore:
    """
    MongoDB storage for time-series friendly schema:
      - sessions: one per app run
      - windows_5s: one per 5-second window

    Fields and indexes follow the requested structure.
    """

    def __init__(
        self,
        uri: str,
        db_name: str = "driver_monitor",
        sessions_coll: str = "sessions",
        windows_coll: str = "windows_5s",
    ) -> None:
        self.client = MongoClient(uri, server_api=ServerApi("1"))
        # Ensure connectivity
        self.client.admin.command("ping")
        self.db = self.client[db_name]
        self.sessions = self.db[sessions_coll]
        self.windows = self.db[windows_coll]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        try:
            # windows_5s indexes
            self.windows.create_index([("session_id", 1), ("window_start_dt", 1)], name="sid_start_dt")
            self.windows.create_index([("window_end_dt", -1)], name="end_dt_desc")
            # sessions index
            self.sessions.create_index([("started_at_dt", 1)], name="started_dt")
        except PyMongoError as e:
            print(f"[warn] Failed to ensure indexes: {e}")

    def start_session(self, session_tz: str = "UTC") -> str:
        session_id = str(uuid4())
        now_utc = datetime.now(timezone.utc)
        doc = {
            "session_id": session_id,
            "session_tz": session_tz,
            "started_at_dt": now_utc,
            "ended_at_dt": None,
        }
        try:
            self.sessions.insert_one(doc)
        except PyMongoError as e:
            print(f"[warn] Failed to create session: {e}")
        return session_id

    def end_session(self, session_id: str) -> None:
        try:
            self.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"ended_at_dt": datetime.now(timezone.utc)}}
            )
        except PyMongoError as e:
            print(f"[warn] Failed to end session {session_id}: {e}")

    def insert_window(self, session_id: str, window_doc: dict) -> Optional[str]:
        """
        window_doc must contain (seconds since epoch):
          - window_start_s (float)
          - window_end_s (float)
          - duration_sec (float)
          - blinks_count (int)
          - eyes_closed_over_3s_episodes (int)
          - eyes_closed_over_3s_total_sec (float)
          - head_over_3s: {left/right/up/down: {episodes:int, total_sec:float}}
          - phone_episodes (int)
          - phone_total_sec (float)
        """
        try:
            start_s = float(window_doc.get("window_start_s"))
            end_s = float(window_doc.get("window_end_s"))
        except Exception:
            print("[warn] insert_window missing or invalid start/end seconds")
            return None

        start_dt = datetime.fromtimestamp(start_s, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_s, tz=timezone.utc)

        doc = {
            "session_id": session_id,
            "window_start_dt": start_dt,
            "window_end_dt": end_dt,
            "window_start_s": start_s,
            "window_end_s": end_s,
            "duration_sec": float(window_doc.get("duration_sec", end_s - start_s)),

            # Eye
            "blinks_count": int(window_doc.get("blinks_count", 0) or 0),
            "eyes_closed_over_3s_episodes": int(window_doc.get("eyes_closed_over_3s_episodes", 0) or 0),
            "eyes_closed_over_3s_total_sec": float(window_doc.get("eyes_closed_over_3s_total_sec", 0.0) or 0.0),

            # Head
            "head_over_3s": window_doc.get("head_over_3s", {
                "left": {"episodes": 0, "total_sec": 0.0},
                "right": {"episodes": 0, "total_sec": 0.0},
                "up": {"episodes": 0, "total_sec": 0.0},
                "down": {"episodes": 0, "total_sec": 0.0},
            }),

            # Phone
            "phone_episodes": int(window_doc.get("phone_episodes", 0) or 0),
            "phone_total_sec": float(window_doc.get("phone_total_sec", 0.0) or 0.0),
        }

        try:
            res = self.windows.insert_one(doc)
            return str(res.inserted_id)
        except PyMongoError as e:
            print(f"[warn] Mongo insert window failed: {e}")
            return None
