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
    # Operation and inputs provided by the caller/API.
    operation: Literal["search", "create", "read", "update", "delete"]
    query: Optional[str]
    top_k: int
    session_id: Optional[str]
    keyword: Optional[str]
    filters: Optional[Dict[str, Any]]
    document_in: Optional[Dict[str, Any]]
    document_update: Optional[Dict[str, Any]]
    doc_id: Optional[str]
    # Intermediate values computed by agents.
    query_embedding: Optional[List[float]]
    # Outputs written by agents.
    results: Optional[List[Dict[str, Any]]]
    crud_result: Optional[Dict[str, Any]]
    answer: Optional[str]
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
    # Simple cosine similarity used only when running against the in-memory mock DB.
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
    # Route to Atlas vector search (or in-memory cosine) using the query embedding.
    collection = get_collection()
    embedding = state.get("query_embedding")
    if not embedding:
        return {"error": "Missing query embedding"}

    top_k = state.get("top_k", settings.default_top_k)

    if settings.use_mock_db:
        # Local-only scoring when mongomock is used: compute cosine manually.
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
        }
    ]

    # Hybrid filter: optional keyword regex + caller-provided filters after vector similarity.
    match_filters: Dict[str, Any] = {}
    keyword = state.get("keyword")
    if keyword:
        match_filters.setdefault("$or", []).extend(
            [
                {"title": {"$regex": keyword, "$options": "i"}},
                {"content": {"$regex": keyword, "$options": "i"}},
                {"plot": {"$regex": keyword, "$options": "i"}},
            ]
        )
    if state.get("filters"):
        match_filters.update(state["filters"])

    if match_filters:
        pipeline.append({"$match": match_filters})

    pipeline.append(
        {
            "$project": {
                "title": 1,
                "content": 1,
                "plot": 1,
                "tags": 1,
                "genres": 1,
                "source": 1,
                "metadata": 1,
                "created_at": 1,
                "updated_at": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    )

    try:
        docs = list(collection.aggregate(pipeline))
    except Exception as exc:  # pragma: no cover
        return {"error": f"Vector search failed: {exc}"}
    return {"results": docs}


def rerank_agent(state: SearchState) -> SearchState:
    """Lightweight reranker that boosts keyword matches on top of vector scores."""

    docs = state.get("results", []) or []
    keyword = (state.get("keyword") or "").lower()

    def score(doc: Dict[str, Any]) -> float:
        base = float(doc.get("score") or 0.0)
        bonus = 0.0
        if keyword:
            blob = " ".join([
                str(doc.get("title", "")),
                str(doc.get("content", "")),
                str(doc.get("plot", "")),
            ]).lower()
            if keyword in blob:
                bonus += 0.1
        return base + bonus

    reranked = sorted(docs, key=score, reverse=True)
    return {"results": reranked}


def answer_agent(state: SearchState) -> SearchState:
    # RAG answerer: call LLM (if available) with citations; fallback to extractive summary.
    docs = state.get("results", []) or []
    query = state.get("query", "")
    if not docs:
        return {"answer": "No documents found to answer this query."}

    try:
        answer = generate_answer(query, docs)
    except Exception as exc:  # pragma: no cover
        parts = []
        for doc in docs[:3]:
            title = doc.get("title") or doc.get("metadata", {}).get("title", "(untitled)")
            body = doc.get("content") or doc.get("plot") or ""
            snippet = (body[:220] + "...") if len(body) > 220 else body
            parts.append(f"- {title}: {snippet}")
        answer = "\n".join(parts) + f"\n(Note: LLM answer failed: {exc})"

    return {"answer": answer}


def memory_write_agent(state: SearchState) -> SearchState:
    # Append a retrieval trace to the session collection so future turns can reuse history.
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
    # CRUD handler keeps embeddings in sync on create/update and returns hydrated docs.
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
        try:
            res = collection.insert_one(to_insert)
            doc = collection.find_one({"_id": res.inserted_id})
            return {"crud_result": doc}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Create failed: {exc}"}

    if op == "read":
        doc_id = state.get("doc_id")
        if not doc_id:
            return {"error": "doc_id is required"}
        try:
            doc = collection.find_one({"_id": ObjectId(doc_id)})
            if not doc:
                return {"error": "Document not found"}
            return {"crud_result": doc}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Read failed: {exc}"}

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
        try:
            collection.update_one({"_id": ObjectId(doc_id)}, update_doc)
            doc = collection.find_one({"_id": ObjectId(doc_id)})
            return {"crud_result": doc}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Update failed: {exc}"}

    if op == "delete":
        doc_id = state.get("doc_id")
        if not doc_id:
            return {"error": "doc_id is required"}
        try:
            res = collection.delete_one({"_id": ObjectId(doc_id)})
            return {"crud_result": {"deleted_count": res.deleted_count}}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Delete failed: {exc}"}

    return {"error": f"Unsupported operation {op}"}


# --------- Graph assembly ---------


def build_graph():
    # Assemble the LangGraph: router -> search path (session -> embed -> search -> log) or CRUD path.
    graph = StateGraph(SearchState)
    graph.add_node("router", router_agent)
    graph.add_node("session_memory", session_memory_agent)
    graph.add_node("embedding_agent", embedding_agent)
    graph.add_node("vector_search_agent", vector_search_agent)
    graph.add_node("rerank_agent", rerank_agent)
    graph.add_node("answer_agent", answer_agent)
    graph.add_node("memory_write_agent", memory_write_agent)
    graph.add_node("crud_agent", crud_agent)

    graph.set_entry_point("router")

    def route(state: SearchState) -> str:
        # Minimal branching: semantic search goes through the embedding + vector path; everything else is CRUD.
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

    # Search chain: load memory -> embed -> vector search -> rerank -> LLM answer -> log session, then terminate.
    graph.add_edge("session_memory", "embedding_agent")
    graph.add_edge("embedding_agent", "vector_search_agent")
    graph.add_edge("vector_search_agent", "rerank_agent")
    graph.add_edge("rerank_agent", "answer_agent")
    graph.add_edge("answer_agent", "memory_write_agent")
    graph.add_edge("memory_write_agent", END)
    # CRUD chain: go straight to CRUD agent and terminate.
    graph.add_edge("crud_agent", END)

    return graph.compile()


GRAPH = build_graph()
