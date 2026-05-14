"""
System / admin client for SynapCores Python SDK (v1.5.0-ce).

Currently only the vision provider config surface is exposed:
  GET/PUT/DELETE /v1/system/vision
  POST /v1/system/vision/test
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SynapCores


class _Vision:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def get(self) -> Optional[Dict[str, Any]]:
        try:
            response = self.client._client.get("/system/vision")
            data = self.client._handle_response(response)
            return data or None
        except Exception:
            return None

    def set(self, config: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client._client.put("/system/vision", json=config)
        return self.client._handle_response(response)

    def delete(self) -> None:
        response = self.client._client.delete("/system/vision")
        self.client._handle_response(response)

    def test(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self.client._client.post(
            "/system/vision/test", json=payload or {}
        )
        return self.client._handle_response(response)


class SystemClient:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client
        self.vision = _Vision(client)
