"""LangSmith-powered monitoring helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from bson import json_util

from .config import get_settings

try:  # pragma: no cover - depends on optional langsmith extra
    from langsmith import Client
    from langsmith.run_trees import RunTree
except ImportError:  # pragma: no cover
    Client = None  # type: ignore
    RunTree = None  # type: ignore

settings = get_settings()
logger = logging.getLogger(__name__)


def _serialize(obj: Any) -> Any:
    try:
        return json_util.loads(json_util.dumps(obj))
    except Exception:  # pragma: no cover
        return str(obj)


class Observability:
    """Thin layer around LangSmith traces so we can re-use it everywhere."""

    def __init__(self) -> None:
        self._client: Optional[Client] = None
        if Client is None or RunTree is None:
            logger.debug("LangSmith not installed, monitoring disabled")
            return
        if not settings.langsmith_project:
            logger.debug("LangSmith project not configured, monitoring disabled")
            return
        try:
            self._client = Client(api_key=settings.langsmith_api_key or None)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to init LangSmith client: %s", exc)
            self._client = None

    def log_workflow_run(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._client or RunTree is None:
            return
        run = RunTree(
            name=operation or settings.langsmith_run_name,
            run_type="chain",
            inputs=_serialize(inputs),
            extra={**(metadata or {}), "success": success},
            tags=tags or [operation],
            project_name=settings.langsmith_project,
        )
        run.end(outputs=_serialize(outputs), error=error)
        run.post(client=self._client)

    def log_evaluation(
        self,
        dataset_name: str,
        metrics: Dict[str, Any],
        examples: list[Dict[str, Any]],
    ) -> None:
        if not self._client:
            return
        payload = {
            "dataset": dataset_name,
            "metrics": metrics,
            "examples": [_serialize(ex) for ex in examples],
        }
        try:
            self._client.log_event(
                run={
                    "name": f"evaluation::{dataset_name}",
                    "run_type": "evaluation",
                    "project_name": settings.langsmith_project,
                    "inputs": {"dataset": dataset_name},
                    "outputs": payload,
                }
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to log evaluation to LangSmith: %s", exc)


monitor = Observability()
