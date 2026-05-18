"""
Graph database client for SynapCores Python SDK (v1.5.0-ce).
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SynapCores


class _GraphNodes:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def create(self, label: str, props: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Gateway contract: {"labels": [...], "properties": {...}}.
        response = self.client._client.post(
            "/graph/nodes",
            json={"labels": [label], "properties": props or {}},
        )
        return self.client._handle_response(response)

    def get(self, id: str) -> Dict[str, Any]:
        response = self.client._client.get(f"/graph/nodes/{id}")
        return self.client._handle_response(response)

    def update(self, id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client._client.patch(
            f"/graph/nodes/{id}", json={"properties": patch}
        )
        return self.client._handle_response(response)

    def delete(self, id: str) -> None:
        response = self.client._client.delete(f"/graph/nodes/{id}")
        self.client._handle_response(response)

    def neighbors(
        self,
        id: str,
        direction: Optional[str] = None,
        limit: Optional[int] = None,
        type: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, str] = {}
        if direction:
            params["direction"] = direction
        if limit is not None:
            params["limit"] = str(limit)
        if type:
            params["type"] = type
        response = self.client._client.get(
            f"/graph/nodes/{id}/neighbors", params=params
        )
        return self.client._handle_response(response)


class _GraphEdges:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def create(
        self,
        from_id: str,
        to_id: str,
        type: str,
        props: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Gateway contract: {"src": ..., "dst": ..., "type": ..., "properties": {...}}.
        response = self.client._client.post(
            "/graph/edges",
            json={"src": from_id, "dst": to_id, "type": type, "properties": props or {}},
        )
        return self.client._handle_response(response)

    def delete(self, id: str) -> None:
        response = self.client._client.delete(f"/graph/edges/{id}")
        self.client._handle_response(response)


class _GraphIndexes:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def create(self, statement: str) -> Dict[str, Any]:
        response = self.client._client.post(
            "/graph/indexes", json={"statement": statement}
        )
        return self.client._handle_response(response)


class _GraphAlgorithms:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def run(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        graph: Optional[str] = None,
        node_filter: Optional[str] = None,
        edge_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"algorithm": name}
        if params is not None:
            body["params"] = params
        if graph:
            body["graph"] = graph
        if node_filter:
            body["node_filter"] = node_filter
        if edge_filter:
            body["edge_filter"] = edge_filter
        response = self.client._client.post("/graph/algorithms", json=body)
        return self.client._handle_response(response)


class _GraphsApi:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def list(self) -> List[Dict[str, Any]]:
        # v0.3.0: gateway mounts graphs-metadata API at /v1/graph/graphs.
        response = self.client._client.get("/graph/graphs")
        data = self.client._handle_response(response)
        if isinstance(data, dict):
            return data.get("graphs") or data.get("items") or []
        return data or []

    def create(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        response = self.client._client.post("/graph/graphs", json=body)
        return self.client._handle_response(response)

    def get(self, name: str) -> Dict[str, Any]:
        response = self.client._client.get(f"/graph/graphs/{name}")
        return self.client._handle_response(response)

    def delete(self, name: str) -> None:
        response = self.client._client.delete(f"/graph/graphs/{name}")
        self.client._handle_response(response)


class GraphClient:
    """High-level wrapper around the v1.5.0-ce graph API."""

    def __init__(self, client: "SynapCores") -> None:
        self.client = client
        self.nodes = _GraphNodes(client)
        self.edges = _GraphEdges(client)
        self.indexes = _GraphIndexes(client)
        self.algorithms = _GraphAlgorithms(client)
        self.graphs = _GraphsApi(client)

    def cypher(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        graph: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Gateway's MatchRequest contract is {"sql": <cypher>}.
        body: Dict[str, Any] = {"sql": query}
        if graph:
            body["graph"] = graph
        response = self.client._client.post("/graph/match", json=body)
        return self.client._handle_response(response)

    def cypher_profile(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        graph: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"sql": query}
        if graph:
            body["graph"] = graph
        response = self.client._client.post("/graph/match/profile", json=body)
        return self.client._handle_response(response)

    def extract(
        self,
        text: Union[str, Dict[str, Any]],
        graph: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = (
            {"text": text, "graph": graph}
            if isinstance(text, str)
            else dict(text)
        )
        response = self.client._client.post("/graph/extract", json=body)
        return self.client._handle_response(response)
