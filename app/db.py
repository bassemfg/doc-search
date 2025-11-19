"""MongoDB helpers with optional in-memory mocks for tests."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Optional

from pymongo.collection import Collection
from pymongo.database import Database

from .config import get_settings

settings = get_settings()

try:  # pragma: no cover - import happens once
    if settings.use_mock_db:
        from mongomock import MongoClient  # type: ignore
    else:
        from pymongo import MongoClient  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("MongoClient dependency is missing") from exc


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    """Return a cached Mongo client (real or mocked)."""

    return MongoClient(settings.mongodb_uri)


def get_db() -> Database:
    return get_client()[settings.mongodb_db_name]


def get_collection() -> Collection:
    return get_db()[settings.mongodb_collection_name]


def get_sessions_collection() -> Collection:
    """Collection used to store lightweight session history."""

    return get_db()[f"{settings.mongodb_collection_name}_sessions"]


def set_timestamps_for_insert(doc: dict) -> dict:
    now = datetime.utcnow()
    doc.setdefault("created_at", now)
    doc.setdefault("updated_at", now)
    return doc


def set_timestamps_for_update(update: dict) -> dict:
    now = datetime.utcnow()
    if "$set" not in update:
        update["$set"] = {}
    update["$set"]["updated_at"] = now
    return update
