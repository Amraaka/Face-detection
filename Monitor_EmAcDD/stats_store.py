from __future__ import annotations

from typing import Optional
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import PyMongoError


class StatsDB:
    def __init__(self, uri: str, db_name: str = "driver_monitor", coll_name: str = "stats"):
        self.client = MongoClient(uri, server_api=ServerApi('1'))
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
