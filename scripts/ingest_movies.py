"""Simple ingestion helper for the sample_movies dataset."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.db import get_collection, set_timestamps_for_insert
from app.embeddings import embed_text

settings = get_settings()
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_movies.json"


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found at {DATA_PATH}")

    docs = json.loads(DATA_PATH.read_text())
    collection = get_collection()
    created = 0
    for record in docs:
        body = {
            "title": record.get("title"),
            "content": record.get("plot", ""),
            "tags": record.get("genres", []),
            "source": "sample_movies",
            "metadata": {"year": record.get("year")},
            "embedding": embed_text(f"{record.get('title')} - {record.get('plot', '')}"),
        }
        set_timestamps_for_insert(body)
        collection.insert_one(body)
        created += 1
    print(f"Inserted {created} documents into {settings.mongodb_collection_name}")


if __name__ == "__main__":
    main()
