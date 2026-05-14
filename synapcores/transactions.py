"""
Server-side transactions client for SynapCores Python SDK (v1.5.0-ce).

Wraps:
  POST /v1/transactions
  GET  /v1/transactions/:id
  POST /v1/transactions/:id/{execute,commit,rollback,savepoint}
  POST /v1/transactions/:id/savepoint/:name/rollback
  GET  /v1/transactions/history
  GET  /v1/transactions/history/:id
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SynapCores


class Tx:
    """Active server-side transaction handle."""

    def __init__(self, client: "SynapCores", id: str) -> None:
        self._client = client
        self.id = id
        self._active = True

    def is_active(self) -> bool:
        return self._active

    def execute(
        self, sql: str, parameters: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        self._assert_active()
        response = self._client._client.post(
            f"/transactions/{self.id}/execute",
            json={"sql": sql, "parameters": parameters or []},
        )
        return self._client._handle_response(response)

    def commit(self) -> None:
        self._assert_active()
        response = self._client._client.post(
            f"/transactions/{self.id}/commit", json={}
        )
        self._client._handle_response(response)
        self._active = False

    def rollback(self) -> None:
        self._assert_active()
        response = self._client._client.post(
            f"/transactions/{self.id}/rollback", json={}
        )
        self._client._handle_response(response)
        self._active = False

    def savepoint(self, name: str) -> None:
        self._assert_active()
        response = self._client._client.post(
            f"/transactions/{self.id}/savepoint", json={"name": name}
        )
        self._client._handle_response(response)

    def rollback_to(self, name: str) -> None:
        self._assert_active()
        response = self._client._client.post(
            f"/transactions/{self.id}/savepoint/{name}/rollback", json={}
        )
        self._client._handle_response(response)

    def __enter__(self) -> "Tx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._active:
            try:
                if exc_type is not None:
                    self.rollback()
                else:
                    self.commit()
            except Exception:
                pass

    def _assert_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                f"Transaction {self.id} has already been committed/rolled back"
            )


class _TxHistory:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def list(
        self,
        limit: Optional[int] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if status:
            params["status"] = status
        response = self.client._client.get(
            "/transactions/history", params=params
        )
        data = self.client._handle_response(response)
        return data.get("transactions") or data.get("history") or data or []

    def get(self, id: str) -> Dict[str, Any]:
        response = self.client._client.get(f"/transactions/history/{id}")
        return self.client._handle_response(response)


class TransactionsClient:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client
        self.history = _TxHistory(client)

    def begin(
        self,
        isolation_level: Optional[str] = None,
        read_only: Optional[bool] = None,
        timeout_secs: Optional[int] = None,
        database: Optional[str] = None,
    ) -> Tx:
        body: Dict[str, Any] = {}
        if isolation_level:
            body["isolation_level"] = isolation_level
        if read_only is not None:
            body["read_only"] = read_only
        if timeout_secs is not None:
            body["timeout_secs"] = timeout_secs
        if database:
            body["database"] = database
        response = self.client._client.post("/transactions", json=body)
        data = self.client._handle_response(response)
        tx_id = data.get("id") or data.get("transaction_id")
        if not tx_id:
            raise RuntimeError("begin(): server did not return a transaction id")
        return Tx(self.client, str(tx_id))
