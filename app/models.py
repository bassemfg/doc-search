"""Pydantic models for API IO."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    title: str
    content: str
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentOut(BaseModel):
    id: str = Field(alias="_id")
    title: str
    content: str
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    score: Optional[float] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: Optional[str] = None
    keyword: Optional[str] = None  # optional keyword filter for hybrid retrieval
    filters: Optional[Dict[str, Any]] = None  # optional Mongo-style filters applied post vector search


class SearchResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    results: List[DocumentOut]


class CRUDResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
