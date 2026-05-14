"""
Multimodal client for SynapCores Python SDK (v1.5.0-ce).
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SynapCores


def _normalize_input(value: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(value, str):
        return {"type": "text", "text": value}
    return value


class MultimodalClient:
    """Wraps /v1/multimodal/*."""

    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def similarity(
        self,
        a: Union[str, Dict[str, Any]],
        b: Union[str, Dict[str, Any]],
        metric: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"a": _normalize_input(a), "b": _normalize_input(b)}
        if metric:
            body["metric"] = metric
        if model:
            body["model"] = model
        response = self.client._client.post("/multimodal/similarity", json=body)
        return self.client._handle_response(response)

    def search(
        self,
        query: Union[str, Dict[str, Any]],
        collection: Optional[str] = None,
        limit: Optional[int] = None,
        modality: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {"query": _normalize_input(query)}
        if collection:
            body["collection"] = collection
        if limit is not None:
            body["limit"] = limit
        if modality:
            body["modality"] = modality
        if filter is not None:
            body["filter"] = filter
        if model:
            body["model"] = model
        response = self.client._client.post("/multimodal/search", json=body)
        data = self.client._handle_response(response)
        return data.get("results") or data.get("hits") or data or []

    def join(
        self,
        left: Union[List[Union[str, Dict[str, Any]]], Dict[str, Any]],
        right: Union[List[Union[str, Dict[str, Any]]], Dict[str, Any]],
        threshold: Optional[float] = None,
        limit: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        def _maybe_normalize_list(side: Any) -> Any:
            if isinstance(side, list):
                return [_normalize_input(s) for s in side]
            return side

        body: Dict[str, Any] = {
            "left": _maybe_normalize_list(left),
            "right": _maybe_normalize_list(right),
        }
        if threshold is not None:
            body["threshold"] = threshold
        if limit is not None:
            body["limit"] = limit
        if model:
            body["model"] = model
        response = self.client._client.post("/multimodal/join", json=body)
        return self.client._handle_response(response)

    def embed(
        self,
        input: Union[str, Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"input": _normalize_input(input)}
        if model:
            body["model"] = model
        response = self.client._client.post("/multimodal/embed", json=body)
        return self.client._handle_response(response)
