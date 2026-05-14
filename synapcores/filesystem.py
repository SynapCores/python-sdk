"""
Filesystem-backed collections client for SynapCores Python SDK (v1.5.0-ce).
"""

from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from .client import SynapCores


class _FsCollections:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def create(
        self,
        name: str,
        path: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        watch: bool = False,
        include_extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "watch": watch}
        if path:
            body["path"] = path
        if description:
            body["description"] = description
        if config is not None:
            body["config"] = config
        if include_extensions is not None:
            body["include_extensions"] = include_extensions
        response = self.client._client.post("/filesystem-collections", json=body)
        return self.client._handle_response(response)

    def list(self) -> List[Dict[str, Any]]:
        response = self.client._client.get("/filesystem-collections")
        data = self.client._handle_response(response)
        return data.get("collections") or data or []

    def get(self, id: str) -> Dict[str, Any]:
        response = self.client._client.get(f"/filesystem-collections/{id}")
        return self.client._handle_response(response)

    def patch(self, id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client._client.patch(
            f"/filesystem-collections/{id}", json=updates
        )
        return self.client._handle_response(response)

    def delete(self, id: str) -> None:
        response = self.client._client.delete(f"/filesystem-collections/{id}")
        self.client._handle_response(response)

    def documents(
        self,
        id: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {}
        if page is not None:
            params["page"] = str(page)
        if page_size is not None:
            params["page_size"] = str(page_size)
        response = self.client._client.get(
            f"/filesystem-collections/{id}/documents", params=params
        )
        data = self.client._handle_response(response)
        return data.get("documents") or data or []

    def reprocess(self, id: str, file_id: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if file_id:
            body["document_id"] = file_id
        response = self.client._client.post(
            f"/filesystem-collections/{id}/reprocess", json=body
        )
        return self.client._handle_response(response)

    def subscribe_progress(self, id: str) -> Iterator[Dict[str, Any]]:
        """Synchronous generator over progress events.

        Uses ``websockets.sync`` (>=12) for blocking reads. The caller can
        iterate naturally:

            for evt in client.filesystem.collections.subscribe_progress(coll_id):
                print(evt)

        Each event is the parsed JSON payload sent by the gateway.
        """
        try:
            from websockets.sync.client import connect as ws_connect  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "subscribe_progress requires the synchronous websockets API "
                "(websockets >= 12). Install with `pip install 'websockets>=12'`."
            ) from e

        import json as _json

        ticket = self.client.create_ws_ticket()
        token = ticket.get("token") or ticket.get("ticket") or ""
        ws_base = self.client._ws_base_url()
        url = f"{ws_base}/ws/filesystem-collections/{id}/progress?token={quote(token)}"

        with ws_connect(url) as ws:
            try:
                while True:
                    message = ws.recv()
                    if isinstance(message, bytes):
                        message = message.decode("utf-8", errors="ignore")
                    try:
                        yield _json.loads(message)
                    except Exception:
                        yield {"raw": message}
            except Exception:
                return


class FilesystemCollectionsClient:
    """High-level wrapper around the v1.5.0-ce filesystem collections API."""

    def __init__(self, client: "SynapCores") -> None:
        self.client = client
        self.collections = _FsCollections(client)
