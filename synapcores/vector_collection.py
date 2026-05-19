"""
Thin wrapper around the gateway's `/v1/vectors/collections/{name}/...` API.

v0.3.0: exposed via ``client.collection(name)``. Distinct from the
document-store :class:`~synapcores.collection.Collection` because the
underlying gateway routes are different (and use ``k`` rather than
``top_k``).
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .client import SynapCores


class VectorCollection:
    """Vector-collection accessor returned by ``client.collection(name)``."""

    def __init__(self, client: "SynapCores", name: str) -> None:
        self.client = client
        self.name = name
        self._base = f"/vectors/collections/{name}"

    # ----- read ------------------------------------------------------------
    def vector_search(
        self,
        vector: Union[List[float], "np.ndarray"],
        top_k: int = 10,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """POST /v1/vectors/collections/{name}/search.

        Args:
            vector: Query vector (list of floats or numpy array).
            top_k: Number of nearest neighbours. Accepted for parity with
                the document-store API; forwarded as ``k`` to the gateway.
            k: Synonym for ``top_k``. If both are provided, ``k`` wins.
            filter: Optional metadata filter.
            include_metadata: Whether to include per-result metadata.

        Returns the raw payload from the gateway (list of hits).
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        body: Dict[str, Any] = {
            "vector": list(vector),
            "k": int(k if k is not None else top_k),
            "include_metadata": include_metadata,
        }
        if filter is not None:
            body["filter"] = filter

        response = self.client._client.post(f"{self._base}/search", json=body)
        return self.client._handle_response(response)

    # ----- write -----------------------------------------------------------
    def insert(
        self,
        vectors: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Insert one or more vectors.

        Wire: ``POST /v1/vectors/collections/{name}/vectors`` with
        ``{vectors: [{id, values, metadata}]}``.

        Accepts either a single record or a list — the SDK always sends
        the wrapped ``{vectors: [...]}`` envelope the gateway expects.

        v0.4.0: introduced as part of the vector-subsystem split so
        users don't have to drop down to ``client._client`` to insert
        vectors. Mirrors the Node SDK's ``VectorCollection.insert``.
        """
        if isinstance(vectors, dict):
            vectors = [vectors]
        response = self.client._client.post(
            f"{self._base}/vectors",
            json={"vectors": list(vectors)},
        )
        return self.client._handle_response(response)

    def delete(self, ids: Union[str, List[str]]) -> Dict[str, Any]:
        """Delete vectors by id.

        Single-id wire: ``DELETE /v1/vectors/collections/{name}/vectors/{id}``.
        Bulk wire: ``DELETE /v1/vectors/collections/{name}/vectors`` with
        ``{ids: [...]}``.
        """
        if isinstance(ids, str):
            response = self.client._client.delete(
                f"{self._base}/vectors/{ids}",
            )
        else:
            response = self.client._client.request(
                "DELETE",
                f"{self._base}/vectors",
                json={"ids": list(ids)},
            )
        return self.client._handle_response(response)

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single vector by id.

        Wire: ``GET /v1/vectors/collections/{name}/vectors/{id}``.
        Returns the gateway payload ``{id, values, metadata}`` or
        ``None`` on 404.
        """
        from .exceptions import NotFoundError

        try:
            response = self.client._client.get(f"{self._base}/vectors/{id}")
            return self.client._handle_response(response)
        except NotFoundError:
            return None

    def count(self) -> int:
        """Vector count.

        Wire: ``GET /v1/vectors/collections/{name}/count`` if the
        gateway exposes it, otherwise falls back to ``info().vector_count``.
        """
        try:
            response = self.client._client.get(f"{self._base}/count")
            data = self.client._handle_response(response)
            if isinstance(data, int):
                return data
            if isinstance(data, dict):
                n = data.get("count") or data.get("vector_count")
                if isinstance(n, int):
                    return n
        except Exception:
            pass
        info = self.info()
        n = info.get("vector_count") if isinstance(info, dict) else None
        return int(n) if isinstance(n, int) else 0

    # ----- metadata --------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        """GET /v1/vectors/collections/{name}."""
        response = self.client._client.get(self._base)
        return self.client._handle_response(response)

    # ----- search ----------------------------------------------------------
    def search(
        self,
        vector: Union[List[float], "np.ndarray"],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """k-NN search.

        Wire: ``POST /v1/vectors/collections/{name}/search`` with
        ``{vector, k, include_metadata, filter?}``. Returns the bare
        list of hits (gateway envelope unwrapped).

        v0.4.0: cleaner alias for :meth:`vector_search` that returns
        the bare list of hits instead of a dict envelope. Use whichever
        feels more natural for the calling code.
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        body: Dict[str, Any] = {
            "vector": list(vector),
            "k": int(k),
            "include_metadata": include_metadata,
        }
        if filter is not None:
            body["filter"] = filter
        response = self.client._client.post(f"{self._base}/search", json=body)
        data = self.client._handle_response(response)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("matches") or data.get("results") or data.get("hits") or []
        return []

    def __repr__(self) -> str:
        return f"VectorCollection(name={self.name!r})"
