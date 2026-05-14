"""
Natural-language-to-SQL client for SynapCores Python SDK (v1.5.0-ce).
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SynapCores


class NL2SqlClient:
    """Wrap /v1/nl2sql/*."""

    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def ask(
        self,
        question: str,
        execute: Optional[bool] = None,
        database: Optional[str] = None,
        dialect: Optional[str] = None,
        tables: Optional[List[str]] = None,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"question": question}
        if execute is not None:
            payload["execute"] = execute
        if database:
            payload["database"] = database
        if dialect:
            payload["dialect"] = dialect
        if tables is not None:
            payload["tables"] = tables
        if context:
            payload["context"] = context
        if session_id:
            payload["session_id"] = session_id
        response = self.client._client.post("/nl2sql/query", json=payload)
        return self.client._handle_response(response)

    def update_schema_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client._client.post("/nl2sql/schema/context", json=payload)
        return self.client._handle_response(response)

    def history(
        self,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if session_id:
            params["session_id"] = session_id
        response = self.client._client.get("/nl2sql/history", params=params)
        data = self.client._handle_response(response)
        return data.get("history") or data.get("entries") or data or []

    def validate(self, sql: str) -> Dict[str, Any]:
        response = self.client._client.post("/nl2sql/validate", json={"sql": sql})
        return self.client._handle_response(response)
