"""LangGraph orchestration for multi-agent CRUD + semantic search."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

from bson import ObjectId
from langgraph.graph import END, START, StateGraph
from pymongo.collection import Collection

from .config import get_settings
from .db import (
    get_collection,
    get_sessions_collection,
    set_timestamps_for_insert,
    set_timestamps_for_update,
)
from .embeddings import embed_text

settings = get_settings()


class SearchState(TypedDict, total=False):
    operation: Literal["search", "create", "read", "update", "delete"]
    query: Optional[str]
    top_k: int
    session_id: Optional[str]
    document_in: Optional[Dict[str, Any]]
    document_update: Optional[Dict[str, Any]]
    doc_id: Optional[str]
    query_embedding: Optional[List[float]]
    results: Optional[List[Dict[str, Any]]]
    crud_result: Optional[Dict[str, Any]]
    history: List[Dict[str, Any]]
    error: Optional[str]


# --------- Agents ---------


def router_agent(state: SearchState) -> SearchState:
    """No-op router placeholder to mirror supervised agentic setups."""

    return {}


def session_memory_agent(state: SearchState) -> SearchState:
    """Load previous session interactions to influence downstream agents."""

    session_id = state.get("session_id")
    if not session_id:
        return {"history": []}

    col = get_sessions_collection()
    doc = col.find_one({"session_id": session_id})
    history = doc.get("interactions", []) if doc else []
    return {"history": history}


def embedding_agent(state: SearchState) -> SearchState:
    query = state.get("query")
    if not query:
        return {"error": "Missing query for semantic search"}

    embedding = embed_text(query)
    return {"query_embedding": embedding}


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    length = min(len(vec_a), len(vec_b))
    dot = sum(vec_a[i] * vec_b[i] for i in range(length))
    norm_a = sum(vec_a[i] ** 2 for i in range(length)) ** 0.5
    norm_b = sum(vec_b[i] ** 2 for i in range(length)) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vector_search_agent(state: SearchState) -> SearchState:
    collection = get_collection()
    embedding = state.get("query_embedding")
    if not embedding:
        return {"error": "Missing query embedding"}

    top_k = state.get("top_k", settings.default_top_k)

    if settings.use_mock_db:
        docs = []
        for doc in collection.find({}):
            doc_embedding = doc.get("embedding") or []
            score = _cosine_similarity(embedding, doc_embedding)
            doc_copy = dict(doc)
            doc_copy["score"] = score
            docs.append(doc_copy)
        docs.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return {"results": docs[:top_k]}

    pipeline = [
        {
            "$vectorSearch": {
                "index": settings.mongodb_vector_index_name,
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": settings.num_candidates,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "title": 1,
                "content": 1,
                "tags": 1,
                "source": 1,
                "metadata": 1,
                "created_at": 1,
                "updated_at": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    docs = list(collection.aggregate(pipeline))
    return {"results": docs}


def memory_write_agent(state: SearchState) -> SearchState:
    session_id = state.get("session_id")
    if not session_id:
        return {}
    col = get_sessions_collection()
    entry = {
        "timestamp": datetime.utcnow(),
        "query": state.get("query"),
        "result_ids": [str(doc.get("_id")) for doc in state.get("results", [])],
    }
    col.update_one(
        {"session_id": session_id},
        {
            "$setOnInsert": {"session_id": session_id},
            "$push": {"interactions": entry},
        },
        upsert=True,
    )
    history = state.get("history", []) + [entry]
    return {"history": history}


def crud_agent(state: SearchState) -> SearchState:
    collection = get_collection()
    op = state["operation"]

    if op == "create":
        doc_in = state.get("document_in") or {}
        embedding = embed_text(doc_in.get("content", ""))
        to_insert = {
            "title": doc_in.get("title"),
            "content": doc_in.get("content"),
            "tags": doc_in.get("tags", []),
            "source": doc_in.get("source"),
            "metadata": doc_in.get("metadata", {}),
            "embedding": embedding,
        }
        set_timestamps_for_insert(to_insert)
        res = collection.insert_one(to_insert)
        doc = collection.find_one({"_id": res.inserted_id})
        return {"crud_result": doc}

    if op == "read":
        doc_id = state.get("doc_id")
        if not doc_id:
            return {"error": "doc_id is required"}
        doc = collection.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return {"error": "Document not found"}
        return {"crud_result": doc}

    if op == "update":
        doc_id = state.get("doc_id")
        if not doc_id:
            return {"error": "doc_id is required"}
        patch = state.get("document_update") or {}
        update_fields: Dict[str, Any] = {}
        if "content" in patch and patch["content"] is not None:
            update_fields["content"] = patch["content"]
            update_fields["embedding"] = embed_text(patch["content"])
        if "title" in patch and patch["title"] is not None:
            update_fields["title"] = patch["title"]
        if "tags" in patch and patch["tags"] is not None:
            update_fields["tags"] = patch["tags"]
        if "source" in patch and patch["source"] is not None:
            update_fields["source"] = patch["source"]
        if "metadata" in patch and patch["metadata"] is not None:
            update_fields["metadata"] = patch["metadata"]
        if not update_fields:
            return {"error": "No valid fields provided for update"}
        update_doc = {"$set": update_fields}
        set_timestamps_for_update(update_doc)
        collection.update_one({"_id": ObjectId(doc_id)}, update_doc)
        doc = collection.find_one({"_id": ObjectId(doc_id)})
        return {"crud_result": doc}

    if op == "delete":
        doc_id = state.get("doc_id")
        if not doc_id:
            return {"error": "doc_id is required"}
        res = collection.delete_one({"_id": ObjectId(doc_id)})
        return {"crud_result": {"deleted_count": res.deleted_count}}

    return {"error": f"Unsupported operation {op}"}


# --------- Graph assembly ---------


def build_graph():
    graph = StateGraph(SearchState)
    graph.add_node("router", router_agent)
    graph.add_node("session_memory", session_memory_agent)
    graph.add_node("embedding_agent", embedding_agent)
    graph.add_node("vector_search_agent", vector_search_agent)
    graph.add_node("memory_write_agent", memory_write_agent)
    graph.add_node("crud_agent", crud_agent)

    graph.set_entry_point("router")

    def route(state: SearchState) -> str:
        op = state.get("operation")
        if op == "search":
            return "search"
        if op in {"create", "read", "update", "delete"}:
            return "crud"
        return "unknown"

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route,
        {
            "search": "session_memory",
            "crud": "crud_agent",
            "unknown": END,
        },
    )

    graph.add_edge("session_memory", "embedding_agent")
    graph.add_edge("embedding_agent", "vector_search_agent")
    graph.add_edge("vector_search_agent", "memory_write_agent")
    graph.add_edge("memory_write_agent", END)
    graph.add_edge("crud_agent", END)

    return graph.compile()


GRAPH = build_graph()
