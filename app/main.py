"""FastAPI surface for the agentic document search workflow."""

from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4

from bson import json_util
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .agents import GRAPH
from .config import get_settings
from .models import (
    CRUDResponse,
    DocumentIn,
    DocumentOut,
    DocumentUpdate,
    SearchRequest,
    SearchResponse,
)
from .monitoring import monitor

settings = get_settings()

app = FastAPI(
    title="Agentic Document Search API",
    description="LangGraph multi-agent workflow on top of MongoDB Atlas Vector Search",
    version="0.1.0",
)


def _bson_to_dict(doc: Dict[str, Any]) -> Dict[str, Any]:
    data = json_util.loads(json_util.dumps(doc))
    if "_id" in data:
        _id = data["_id"]
        if isinstance(_id, dict) and "$oid" in _id:
            data["_id"] = _id["$oid"]
        else:
            try:
                data["_id"] = str(_id)
            except Exception:
                pass
    return data


def _documents_from_results(docs: List[Dict[str, Any]]) -> List[DocumentOut]:
    converted: List[DocumentOut] = []
    for doc in docs:
        _id = doc.get("_id")
        if isinstance(_id, dict) and "$oid" in _id:
            _id = _id["$oid"]
        elif _id is not None:
            try:
                _id = str(_id)
            except Exception:
                pass
        converted.append(
            DocumentOut(
                **{
                    "_id": _id,
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "tags": doc.get("tags", []),
                    "source": doc.get("source"),
                    "metadata": doc.get("metadata", {}),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                    "score": doc.get("score"),
                }
            )
        )
    return converted


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "vector_index": settings.mongodb_vector_index_name,
        "database": settings.mongodb_db_name,
    }


@app.post("/search", response_model=SearchResponse)
def semantic_search(body: SearchRequest) -> SearchResponse:
    session_id = body.session_id or str(uuid4())
    initial_state: Dict[str, Any] = {
        "operation": "search",
        "query": body.query,
        "top_k": body.top_k or settings.default_top_k,
        "session_id": session_id,
        "keyword": body.keyword,
        "filters": body.filters,
    }
    state = GRAPH.invoke(initial_state)

    if state.get("error"):
        monitor.log_workflow_run(
            operation="search",
            inputs=initial_state,
            outputs={"error": state["error"]},
            success=False,
            error=state["error"],
        )
        raise HTTPException(status_code=400, detail=state["error"])

    docs = [
        _bson_to_dict(doc)
        for doc in state.get("results", [])
    ]
    response = SearchResponse(
        query=body.query,
        answer=state.get("answer"),
        results=_documents_from_results(docs),
    )
    monitor.log_workflow_run(
        operation="search",
        inputs=initial_state,
        outputs=response.model_dump(),
        success=True,
        metadata={"session_id": session_id},
    )
    return response


@app.post("/documents", response_model=CRUDResponse)
def create_document(body: DocumentIn) -> CRUDResponse:
    payload = body.model_dump()
    state = GRAPH.invoke({"operation": "create", "document_in": payload})
    if state.get("error"):
        monitor.log_workflow_run(
            "create",
            inputs=payload,
            outputs={"error": state["error"]},
            success=False,
            error=state["error"],
        )
        raise HTTPException(status_code=400, detail=state["error"])
    doc = _bson_to_dict(state["crud_result"])
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    response = CRUDResponse(success=True, data=doc)
    monitor.log_workflow_run("create", inputs=payload, outputs=response.model_dump(), success=True)
    return response


@app.get("/documents/{doc_id}", response_model=CRUDResponse)
def read_document(doc_id: str) -> CRUDResponse:
    state = GRAPH.invoke({"operation": "read", "doc_id": doc_id})
    if state.get("error"):
        monitor.log_workflow_run(
            "read",
            inputs={"doc_id": doc_id},
            outputs={"error": state["error"]},
            success=False,
            error=state["error"],
        )
        raise HTTPException(status_code=404, detail=state["error"])
    doc = _bson_to_dict(state["crud_result"])
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    response = CRUDResponse(success=True, data=doc)
    monitor.log_workflow_run("read", inputs={"doc_id": doc_id}, outputs=response.model_dump(), success=True)
    return response


@app.put("/documents/{doc_id}", response_model=CRUDResponse)
def update_document(doc_id: str, body: DocumentUpdate) -> CRUDResponse:
    update_payload = body.model_dump(exclude_unset=True)
    state = GRAPH.invoke(
        {
            "operation": "update",
            "doc_id": doc_id,
            "document_update": update_payload,
        }
    )
    if state.get("error"):
        monitor.log_workflow_run(
            "update",
            inputs={"doc_id": doc_id, **update_payload},
            outputs={"error": state["error"]},
            success=False,
            error=state["error"],
        )
        raise HTTPException(status_code=400, detail=state["error"])
    doc = _bson_to_dict(state["crud_result"])
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    response = CRUDResponse(success=True, data=doc)
    monitor.log_workflow_run(
        "update",
        inputs={"doc_id": doc_id, **update_payload},
        outputs=response.model_dump(),
        success=True,
    )
    return response


@app.delete("/documents/{doc_id}", response_model=CRUDResponse)
def delete_document(doc_id: str) -> CRUDResponse:
    state = GRAPH.invoke({"operation": "delete", "doc_id": doc_id})
    if state.get("error"):
        monitor.log_workflow_run(
            "delete",
            inputs={"doc_id": doc_id},
            outputs={"error": state["error"]},
            success=False,
            error=state["error"]
        )
        raise HTTPException(status_code=400, detail=state["error"])
    response = CRUDResponse(success=True, data=state.get("crud_result"))
    monitor.log_workflow_run("delete", inputs={"doc_id": doc_id}, outputs=response.model_dump(), success=True)
    return response


@app.exception_handler(Exception)
def handle_exception(request, exc: Exception):  # pragma: no cover - FastAPI handles
    return JSONResponse(status_code=500, content={"detail": str(exc)})
