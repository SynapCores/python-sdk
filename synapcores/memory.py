"""
Agent memory client for SynapCores Python SDK.

Wraps the engine-side ``MEMORY_STORE`` / ``MEMORY_RECALL`` /
``MEMORY_FORGET`` SQL functions (shipped in v1.8.5-prep).

Memories are scoped by a ``namespace`` identifier; the engine
auto-creates the backing table on first store and embeds content with
the configured embedding model. This module exposes a small typed
surface --- :meth:`MemoryClient.store`, :meth:`MemoryClient.recall`,
:meth:`MemoryClient.forget` --- that the OSS aerospace-rca demo, the
OpenClaw plugin, and the planned Hermes plugin all consume.
"""

from datetime import datetime
import json as _json
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from pydantic import BaseModel, ConfigDict

from .exceptions import SynapCoresError

if TYPE_CHECKING:
    from .client import SynapCores


# Namespace must be a valid SQL identifier; the engine enforces the
# same regex on the server side, but validating client-side avoids a
# round-trip when callers pass obviously bad input.
_NAMESPACE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MemoryError(SynapCoresError):
    """Raised when an agent-memory operation fails.

    Covers client-side validation failures (invalid namespace, empty
    id) and server-side failures surfaced through the SQL layer
    (embedding model not configured, namespace storage error, etc.).
    """

    pass


class MemoryRecord(BaseModel):
    """A single memory row returned by ``MEMORY_RECALL``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    content: str
    similarity: float  # in [0, 1]
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime


class MemoryClient:
    """Agent-memory primitives --- store / recall / forget over the
    engine's ``MEMORY_STORE`` / ``MEMORY_RECALL`` / ``MEMORY_FORGET`` SQL
    functions.

    Memories are scoped by ``namespace``; the engine auto-creates the
    backing table on first store. Content is embedded via the
    configured embedding model.
    """

    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        namespace: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store ``content`` under ``namespace``. Returns the generated memory id.

        Args:
            namespace: Identifier scoping the memory (matches
                ``^[A-Za-z_][A-Za-z0-9_]*$``).
            content: Free-form text to embed and persist.
            metadata: Optional JSON-serialisable dict attached to the row.

        Raises:
            MemoryError: If ``namespace`` is invalid, the engine errors,
                or the response is missing the generated id.
        """
        self._validate_namespace(namespace)
        if not isinstance(content, str):
            raise MemoryError(
                f"content must be a string (got {type(content).__name__})",
                code="MEMORY_INVALID_CONTENT",
            )

        if metadata is None:
            sql = "SELECT MEMORY_STORE($1, $2) AS id"
            params: List[Any] = [namespace, content]
        else:
            if not isinstance(metadata, dict):
                raise MemoryError(
                    "metadata must be a dict",
                    code="MEMORY_INVALID_METADATA",
                )
            try:
                metadata_json = _json.dumps(metadata)
            except (TypeError, ValueError) as e:
                raise MemoryError(
                    f"metadata is not JSON-serialisable: {e}",
                    code="MEMORY_INVALID_METADATA",
                )
            sql = "SELECT MEMORY_STORE($1, $2, $3) AS id"
            params = [namespace, content, metadata_json]

        try:
            result = self.client.execute_query(sql, parameters=params)
        except SynapCoresError:
            raise
        except Exception as e:
            raise MemoryError(
                f"MEMORY_STORE failed: {e}",
                code="MEMORY_STORE_FAILED",
            )

        memory_id = self._first_cell(result)
        if not isinstance(memory_id, str) or not memory_id:
            raise MemoryError(
                "MEMORY_STORE did not return a memory id",
                code="MEMORY_STORE_EMPTY",
            )
        return memory_id

    def recall(
        self,
        namespace: str,
        query: str,
        *,
        top_k: int = 10,
    ) -> List[MemoryRecord]:
        """Return the top-K semantically similar memories in ``namespace``.

        Args:
            namespace: Identifier scoping the memory.
            query: Free-form text whose embedding is compared against
                stored rows.
            top_k: Maximum number of rows to return.

        Returns:
            A list of :class:`MemoryRecord`, ordered by descending
            ``similarity``. Empty if the namespace is empty or has not
            been initialised yet.
        """
        self._validate_namespace(namespace)
        if not isinstance(query, str):
            raise MemoryError(
                f"query must be a string (got {type(query).__name__})",
                code="MEMORY_INVALID_QUERY",
            )
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise MemoryError(
                f"top_k must be a positive integer (got {top_k!r})",
                code="MEMORY_INVALID_TOP_K",
            )

        sql = (
            "SELECT id, content, similarity, metadata, created_at "
            "FROM MEMORY_RECALL($1, $2, $3)"
        )
        params: List[Any] = [namespace, query, top_k]

        try:
            result = self.client.execute_query(sql, parameters=params)
        except SynapCoresError:
            raise
        except Exception as e:
            raise MemoryError(
                f"MEMORY_RECALL failed: {e}",
                code="MEMORY_RECALL_FAILED",
            )

        col_names = self._column_names(result.get("columns"))
        rows = result.get("rows") or []
        return [self._row_to_record(row, col_names) for row in rows]

    def forget(self, namespace: str, id: str) -> bool:
        """Delete a memory by id. Returns ``True`` if a row was removed.

        Args:
            namespace: Identifier scoping the memory.
            id: The memory id returned by :meth:`store`.

        Returns:
            ``True`` if the row existed and was deleted; ``False`` if no
            row matched (the engine returns ``BOOLEAN`` for this case).
        """
        self._validate_namespace(namespace)
        if not isinstance(id, str) or not id:
            raise MemoryError(
                f"id must be a non-empty string (got {id!r})",
                code="MEMORY_INVALID_ID",
            )

        sql = "SELECT MEMORY_FORGET($1, $2) AS deleted"
        params: List[Any] = [namespace, id]

        try:
            result = self.client.execute_query(sql, parameters=params)
        except SynapCoresError:
            raise
        except Exception as e:
            raise MemoryError(
                f"MEMORY_FORGET failed: {e}",
                code="MEMORY_FORGET_FAILED",
            )

        deleted = self._first_cell(result)
        return bool(deleted)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        if not isinstance(namespace, str) or not _NAMESPACE_RE.match(namespace):
            raise MemoryError(
                f"Invalid namespace {namespace!r}: must match {_NAMESPACE_RE.pattern}",
                code="MEMORY_INVALID_NAMESPACE",
            )

    @staticmethod
    def _column_names(columns: Any) -> List[str]:
        names: List[str] = []
        for c in columns or []:
            if isinstance(c, dict):
                names.append(str(c.get("name") or c.get("column") or ""))
            else:
                names.append(str(c))
        return names

    @staticmethod
    def _first_cell(result: Dict[str, Any]) -> Any:
        """Return the first column of the first row, or ``None``.

        Tolerates both positional row shapes (``[[val]]``) and dict-row
        shapes (``[{"id": val}]``) that the gateway may emit.
        """
        rows = result.get("rows") or []
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, list):
            return row[0] if row else None
        if isinstance(row, dict):
            return next(iter(row.values()), None)
        return row

    def _row_to_record(
        self,
        row: Union[List[Any], Dict[str, Any]],
        col_names: List[str],
    ) -> MemoryRecord:
        """Map a single gateway-shaped row to a :class:`MemoryRecord`."""
        if isinstance(row, dict):
            data: Dict[str, Any] = row
        elif isinstance(row, list):
            if col_names and len(col_names) >= len(row):
                data = {col_names[i]: row[i] for i in range(len(row))}
            else:
                # Fall back to the documented column order from MEMORY_RECALL.
                fallback = ["id", "content", "similarity", "metadata", "created_at"]
                data = {
                    fallback[i]: row[i]
                    for i in range(min(len(row), len(fallback)))
                }
        else:
            raise MemoryError(
                f"Unexpected MEMORY_RECALL row shape: {type(row).__name__}",
                code="MEMORY_RECALL_BAD_ROW",
            )

        return MemoryRecord(
            id=str(data.get("id") or ""),
            content=str(data.get("content") or ""),
            similarity=float(data.get("similarity") or 0.0),
            metadata=self._parse_metadata(data.get("metadata")),
            created_at=self._parse_created_at(data.get("created_at")),
        )

    @staticmethod
    def _parse_metadata(raw: Any) -> Optional[Dict[str, Any]]:
        """Parse the ``metadata`` TEXT column into a dict (or None)."""
        if raw is None or raw == "":
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = _json.loads(raw)
            except (ValueError, TypeError):
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    @staticmethod
    def _parse_created_at(raw: Any) -> datetime:
        """Parse the ``created_at`` TIMESTAMP column into a datetime."""
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            s = raw.strip()
            # Tolerate both "...Z" and "...+00:00" forms.
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                pass
            for fmt in (
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
            ):
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
        # Last-resort fallback so the model still parses; the value is
        # clearly bogus but better than crashing the recall loop.
        return datetime.utcnow()
