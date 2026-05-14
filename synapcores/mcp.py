"""
Model Context Protocol (MCP) client for SynapCores Python SDK (v1.5.0-ce).
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SynapCores


_request_counter = 0


def _with_id(req: Dict[str, Any]) -> Dict[str, Any]:
    global _request_counter
    if req.get("id") is not None:
        return req
    _request_counter += 1
    req = dict(req)
    req["id"] = f"mcp-{int(time.time() * 1000)}-{_request_counter}"
    return req


class McpClient:
    """Wraps /v1/mcp."""

    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def invoke(self, req: Dict[str, Any]) -> Dict[str, Any]:
        body = _with_id(req)
        response = self.client._client.post("/mcp", json=body)
        return self.client._handle_response(response)

    def batch(self, reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        body = [_with_id(r) for r in reqs]
        response = self.client._client.post("/mcp/batch", json=body)
        data = self.client._handle_response(response)
        if isinstance(data, list):
            return data
        return data.get("responses") or data.get("results") or []

    def info(self) -> Dict[str, Any]:
        response = self.client._client.get("/mcp/info")
        return self.client._handle_response(response)
