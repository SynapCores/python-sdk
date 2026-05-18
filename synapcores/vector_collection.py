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
        vectors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """POST /v1/vectors/collections/{name}/vectors (bulk insert)."""
        response = self.client._client.post(
            f"{self._base}/vectors",
            json={"vectors": vectors},
        )
        return self.client._handle_response(response)

    def delete(self, ids: Union[str, List[str]]) -> Dict[str, Any]:
        """DELETE /v1/vectors/collections/{name}/vectors."""
        ids_list = [ids] if isinstance(ids, str) else list(ids)
        response = self.client._client.request(
            "DELETE",
            f"{self._base}/vectors",
            json={"ids": ids_list},
        )
        return self.client._handle_response(response)

    # ----- metadata --------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        """GET /v1/vectors/collections/{name}."""
        response = self.client._client.get(self._base)
        return self.client._handle_response(response)

    def __repr__(self) -> str:
        return f"VectorCollection(name={self.name!r})"
