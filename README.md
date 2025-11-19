# Agentic Document Search API

LangGraph-powered FastAPI service that ingests documents, embeds them, stores metadata + vectors inside MongoDB Atlas, and exposes CRUD + semantic search APIs. The orchestration mirrors the MongoDB agentic reference solutions (router + specialist agents + shared memory) and ships with monitoring hooks (LangSmith), evaluation scaffolding, and automated tests.

## Architecture

- **Multi-agent LangGraph workflow** (`app/agents.py`)
  - `router_agent` detects the requested operation.
  - `session_memory_agent` reads/maintains conversations per session.
  - `embedding_agent` calls the embedding provider (OpenAI by default, deterministic fake embeddings for tests/dev).
  - `vector_search_agent` executes MongoDB Atlas `$vectorSearch` (with a cosine similarity fallback when using the mock DB).
  - `memory_write_agent` persists interaction history for follow-up context.
  - `crud_agent` owns create/read/update/delete flows and keeps embeddings in sync.
- **FastAPI surface** (`app/main.py`) is intentionally thin – each endpoint just builds the initial graph state and hands off to LangGraph.
- **MongoDB Atlas Vector Search** stores both the canonical document and its embedding in a single collection. Session traces are stored in a sibling collection.
- **Observability + evaluations** (`app/monitoring.py`, `scripts/run_evals.py`) emit LangSmith traces/runs if the environment is configured.
- **Sample data + ingestion** (`data/sample_movies.json`, `scripts/ingest_movies.py`) provide a turnkey dataset to demo the full workflow.
- **Automated tests** (`tests/test_api.py`) run against an in-memory Mongo mock + deterministic embeddings for reproducibility.

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -e .
   ```

2. **Copy env template**
   ```bash
   cp .env.example .env
   ```
   Fill in MongoDB Atlas + OpenAI credentials. Set `USE_MOCK_DB=true` and/or `USE_FAKE_EMBEDDINGS=true` for offline development / testing.

3. **Create the Atlas vector index** (once per collection):
   ```json
   {
     "name": "kb_documents_vs_idx",
     "type": "vectorSearch",
     "definition": {
       "fields": [
         {
           "type": "vector",
           "path": "embedding",
           "numDimensions": 1536,
           "similarity": "cosine"
         }
       ]
     }
   }
   ```

4. **Ingest sample data (optional)**
   ```bash
   python scripts/ingest_movies.py
   ```

5. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```

## API Surface

- `POST /documents` – create
- `GET /documents/{id}` – read
- `PUT /documents/{id}` – update (partial)
- `DELETE /documents/{id}` – delete
- `POST /search` – semantic search, returns top-N docs with metadata + score and supports optional `session_id`
- `GET /health` – readiness probe

Example search payload:
```json
{
  "query": "Vector search tutorial",
  "top_k": 5,
  "session_id": "demo-session-123"
}
```

## Monitoring & Evaluations

- **LangSmith tracing** is wired via `app/monitoring.py`. Set the following env vars to activate:
  ```bash
  export LANGSMITH_API_KEY=...
  export LANGSMITH_PROJECT=doc-search-observability
  export LANGSMITH_RUN_NAME=doc-search
  ```
  Every API call is logged as a LangSmith run; offline evaluations (below) are recorded under the `evaluation::dataset` namespace.
- **Offline eval harness** (`scripts/run_evals.py`) replays curated search cases and reports accuracy:
  ```bash
  python scripts/run_evals.py
  ```
  Results print locally and are forwarded to LangSmith if enabled.

## Testing

Pytest uses the mock DB + fake embeddings (controlled by env defaults in `tests/test_api.py`).
```bash
python -m pytest
```

## Design Notes & Extensions

- **Sessions & memory** – `session_memory_agent` + `memory_write_agent` persist interactions in `<collection>_sessions`. Multi-turn flows can re-use the same `session_id` to provide conversational context, filters, or personalization.
- **Context engineering** – embeddings combine title + body. Downstream LLM answer generation can be added after `vector_search_agent` by introducing another node that builds a retrieval-augmented response.
- **MCP / toolability** – the LangGraph workflow is self-contained; exposing it through MCP simply requires wrapping the `GRAPH.invoke()` calls inside MCP tool definitions (mirroring how the predictive-maintenance repos expose agent tools).
- **Provider-agnostic embeddings** – swap OpenAI for HuggingFace/Cohere by editing `app/embeddings.py`. Mock embeddings keep local tests deterministic.
- **Vector store alternatives** – the data-access layer only touches `get_collection()`. Swap `pymongo` for Pinecone/Chroma adapters if required.

## Repository Layout

```
app/
  agents.py         # LangGraph multi-agent workflow
  config.py         # Settings / env plumbing
  db.py             # MongoDB helpers (+ session collection)
  embeddings.py     # Embedding provider abstraction
  main.py           # FastAPI entrypoints
  models.py         # Pydantic schemas
  monitoring.py     # LangSmith tracing
scripts/
  ingest_movies.py  # Sample dataset ingestion
  run_evals.py      # Offline evaluation harness
tests/
  test_api.py       # CRUD + semantic search regression test
```

## Next Steps

- Add a summariser/answer agent that cites retrieved docs.
- Layer hybrid retrieval (keyword + vector) by extending `vector_search_agent` with `$match` filters.
- Promote session history into a standalone analytics dashboard using Atlas Charts or MongoDB Stream Processing.
