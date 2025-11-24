"""Backfill embeddings for existing movies that lack them."""

from __future__ import annotations

import sys
from pathlib import Path

from pymongo import UpdateOne

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.db import get_collection, set_timestamps_for_update
from app.embeddings import embed_text

settings = get_settings()


def main(limit: int | None = None) -> None:
    collection = get_collection()

    query = {"embedding": {"$exists": False}}
    if limit:
        cursor = collection.find(query, limit=limit)
    else:
        cursor = collection.find(query)

    bulk_ops = []
    processed = 0
    for doc in cursor:
        text = f"{doc.get('title', '')} - {doc.get('plot', '')}"
        embedding = embed_text(text)
        update_doc = {"$set": {"embedding": embedding}}
        set_timestamps_for_update(update_doc)
        bulk_ops.append(UpdateOne({"_id": doc["_id"]}, update_doc))
        processed += 1

        # Execute in chunks to avoid oversized bulk requests
        if len(bulk_ops) >= 100:
            collection.bulk_write(bulk_ops)
            bulk_ops.clear()
            print(f"Updated {processed} documents so far...")

    if bulk_ops:
        collection.bulk_write(bulk_ops)

    print(f"Embedding backfill complete. Total updated: {processed}")
    print(
        "Docs with embedding:",
        collection.count_documents({"embedding": {"$exists": True}}),
    )


if __name__ == "__main__":
    # Optional: pass a limit as CLI arg
    arg_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=arg_limit)
