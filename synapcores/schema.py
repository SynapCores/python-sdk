"""
Schema introspection client for SynapCores Python SDK (v1.5.0-ce).
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SynapCores


class SchemaClient:
    """Wrap /v1/schema/*."""

    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def list_databases(self) -> List[Dict[str, Any]]:
        response = self.client._client.get("/schema/databases")
        data = self.client._handle_response(response)
        return data.get("databases") or data or []

    def list_tables(self, include_system: bool = False) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {}
        if include_system:
            params["include_system"] = "true"
        response = self.client._client.get("/schema/tables", params=params)
        data = self.client._handle_response(response)
        return data.get("tables") or data or []

    def get_table(self, name: str) -> Dict[str, Any]:
        response = self.client._client.get(f"/schema/tables/{name}")
        return self.client._handle_response(response)

    def get_columns(self, name: str) -> List[Dict[str, Any]]:
        response = self.client._client.get(f"/schema/tables/{name}/columns")
        data = self.client._handle_response(response)
        return data.get("columns") or data or []

    def get_indexes(self, name: str) -> List[Dict[str, Any]]:
        response = self.client._client.get(f"/schema/tables/{name}/indexes")
        data = self.client._handle_response(response)
        return data.get("indexes") or data or []

    def preview_table(
        self,
        name: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        response = self.client._client.get(
            f"/schema/tables/{name}/data", params=params
        )
        return self.client._handle_response(response)
