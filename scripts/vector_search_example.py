"""Quick smoke test for vector search against the Atlas movies collection."""

from __future__ import annotations

from app.config import get_settings
from app.db import get_collection
from app.embeddings import embed_text

settings = get_settings()


def main() -> None:
    query = "Fight Club"
    embedding = embed_text(query)
    collection = get_collection()

    pipeline = [
        {
            "$vectorSearch": {
                "index": settings.mongodb_vector_index_name,
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": settings.num_candidates,
                "limit": settings.default_top_k,
            }
        },
        {
            "$project": {
                "title": 1,
                "plot": 1,
                "genres": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    docs = list(collection.aggregate(pipeline))
    if not docs:
        print("No results returned. Check that embeddings exist and the vector index is built.")
        return

    print(f"Top {len(docs)} results for query '{query}':")
    for i, doc in enumerate(docs, 1):
        title = doc.get("title")
        score = doc.get("score")
        print(f"{i}. {title} (score={score})")


if __name__ == "__main__":
    main()
