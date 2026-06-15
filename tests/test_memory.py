"""
Tests for the agent-memory client (MemoryClient).

This file mixes two flavours:

1. **Unit tests** (always run) cover client-side validation, SQL
   construction, metadata parsing, and response shape handling. They
   use ``unittest.mock`` exactly like ``tests/test_client.py``.

2. **Live integration tests** (skipped unless an engine is reachable)
   cover the seven scenarios listed in the implementation brief. Set
   ``SYNAPCORES_TEST_LIVE=1`` --- and optionally
   ``SYNAPCORES_TEST_HOST`` / ``SYNAPCORES_TEST_PORT`` /
   ``SYNAPCORES_TEST_API_KEY`` --- to run them against a real v1.8.5+
   gateway. When the env var is unset the tests are skipped cleanly
   (``pytest -q`` shows them as ``s``).
"""

from __future__ import annotations

import json
import os
import secrets
import time
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

from synapcores import SynapCores
from synapcores.memory import MemoryClient, MemoryError, MemoryRecord


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _envelope(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a payload in the gateway's ``{"data": ..., "meta": ...}`` envelope.

    The client's ``_handle_response`` strips this envelope before
    handing the body off to the sub-client, so simulating it keeps the
    mocks aligned with production wire shape.
    """
    return {"data": payload, "meta": {}}


def _mock_engine_response(
    mock_httpx_client: Any,
    payload: Dict[str, Any],
    status: int = 200,
) -> Mock:
    """Wire ``mock_httpx_client`` to return ``payload`` on ``post``.

    Returns the inner mock client so callers can assert on the call.
    """
    response = Mock()
    response.status_code = status
    response.content = json.dumps(_envelope(payload)).encode()
    response.json.return_value = _envelope(payload)

    inner = Mock()
    inner.post.return_value = response
    mock_httpx_client.return_value = inner
    return inner


@pytest.fixture
def mock_httpx_client() -> Any:
    with patch("synapcores.client.httpx.Client") as mock:
        yield mock


# ---------------------------------------------------------------------------
# Unit tests (always run)
# ---------------------------------------------------------------------------


def test_memory_client_wired_into_synapcores(mock_httpx_client: Any) -> None:
    """``client.memory`` is a ready MemoryClient bound to the parent."""
    mock_httpx_client.return_value = Mock()
    client = SynapCores()
    assert isinstance(client.memory, MemoryClient)
    assert client.memory.client is client


def test_invalid_namespace_raises_without_hitting_engine(
    mock_httpx_client: Any,
) -> None:
    """Spec test #7: invalid namespace MUST raise MemoryError locally."""
    inner = Mock()
    mock_httpx_client.return_value = inner
    client = SynapCores()

    bad_namespaces = ["", "1bad", "has space", "has-dash", "drop;table", None]
    for ns in bad_namespaces:
        with pytest.raises(MemoryError):
            client.memory.store(ns, "anything")  # type: ignore[arg-type]
        with pytest.raises(MemoryError):
            client.memory.recall(ns, "anything")  # type: ignore[arg-type]
        with pytest.raises(MemoryError):
            client.memory.forget(ns, "mem_xxx_yyy")  # type: ignore[arg-type]

    # No HTTP call should have been issued for client-side rejections.
    inner.post.assert_not_called()
    inner.get.assert_not_called()


def test_store_builds_two_arg_sql_without_metadata(
    mock_httpx_client: Any,
) -> None:
    inner = _mock_engine_response(
        mock_httpx_client,
        {
            "rows": [["mem_abc123_XYZ"]],
            "columns": [{"name": "id", "data_type": "TEXT"}],
            "rows_affected": 1,
            "execution_time_ms": 4,
        },
    )

    client = SynapCores()
    memory_id = client.memory.store("default", "I prefer Python")

    assert memory_id == "mem_abc123_XYZ"
    inner.post.assert_called_once()
    path, kwargs = inner.post.call_args.args[0], inner.post.call_args.kwargs
    body = kwargs["json"]
    assert path == "/query/execute"
    assert body["sql"] == "SELECT MEMORY_STORE($1, $2) AS id"
    assert body["parameters"] == ["default", "I prefer Python"]


def test_store_builds_three_arg_sql_with_metadata(
    mock_httpx_client: Any,
) -> None:
    inner = _mock_engine_response(
        mock_httpx_client,
        {
            "rows": [["mem_abc123_XYZ"]],
            "columns": [{"name": "id"}],
        },
    )

    client = SynapCores()
    metadata = {"importance": 0.9, "source": "test"}
    memory_id = client.memory.store(
        "default", "Customer renewed", metadata=metadata
    )

    assert memory_id == "mem_abc123_XYZ"
    body = inner.post.call_args.kwargs["json"]
    assert body["sql"] == "SELECT MEMORY_STORE($1, $2, $3) AS id"
    assert body["parameters"][:2] == ["default", "Customer renewed"]
    # The metadata is sent as a JSON string the engine will parse.
    assert json.loads(body["parameters"][2]) == metadata


def test_store_raises_when_engine_returns_empty(
    mock_httpx_client: Any,
) -> None:
    _mock_engine_response(
        mock_httpx_client,
        {"rows": [], "columns": [{"name": "id"}]},
    )

    client = SynapCores()
    with pytest.raises(MemoryError) as exc:
        client.memory.store("default", "nothing")
    assert "did not return a memory id" in str(exc.value)


def test_store_rejects_non_serialisable_metadata(
    mock_httpx_client: Any,
) -> None:
    inner = Mock()
    mock_httpx_client.return_value = inner
    client = SynapCores()

    class _NotJson:
        pass

    with pytest.raises(MemoryError):
        client.memory.store(
            "default", "content", metadata={"bad": _NotJson()}
        )
    inner.post.assert_not_called()


def test_recall_builds_table_valued_sql(mock_httpx_client: Any) -> None:
    inner = _mock_engine_response(
        mock_httpx_client,
        {
            "rows": [
                [
                    "mem_1_a",
                    "I prefer Python over Java",
                    0.92,
                    '{"importance": 0.9}',
                    "2025-05-15T12:34:56.789012",
                ]
            ],
            "columns": [
                {"name": "id"},
                {"name": "content"},
                {"name": "similarity"},
                {"name": "metadata"},
                {"name": "created_at"},
            ],
        },
    )

    client = SynapCores()
    hits = client.memory.recall("default", "what language do I like", top_k=5)

    body = inner.post.call_args.kwargs["json"]
    assert (
        body["sql"]
        == "SELECT id, content, similarity, metadata, created_at FROM MEMORY_RECALL($1, $2, $3)"
    )
    assert body["parameters"] == ["default", "what language do I like", 5]

    assert len(hits) == 1
    record = hits[0]
    assert isinstance(record, MemoryRecord)
    assert record.id == "mem_1_a"
    assert record.content == "I prefer Python over Java"
    assert record.similarity == pytest.approx(0.92)
    # Spec test #3: metadata parsed back to a dict.
    assert record.metadata == {"importance": 0.9}
    assert record.created_at.year == 2025


def test_recall_default_top_k(mock_httpx_client: Any) -> None:
    inner = _mock_engine_response(
        mock_httpx_client,
        {"rows": [], "columns": []},
    )
    client = SynapCores()
    client.memory.recall("default", "nothing here")
    body = inner.post.call_args.kwargs["json"]
    assert body["parameters"][2] == 10


def test_recall_empty_namespace_returns_empty_list(
    mock_httpx_client: Any,
) -> None:
    """Spec test #4: empty namespace returns ``[]``, not an error."""
    _mock_engine_response(
        mock_httpx_client,
        {"rows": [], "columns": []},
    )
    client = SynapCores()
    assert client.memory.recall("default", "anything") == []


def test_recall_handles_null_metadata(mock_httpx_client: Any) -> None:
    _mock_engine_response(
        mock_httpx_client,
        {
            "rows": [
                [
                    "mem_x",
                    "content",
                    0.5,
                    None,
                    "2025-05-15T00:00:00",
                ]
            ],
            "columns": [
                {"name": "id"},
                {"name": "content"},
                {"name": "similarity"},
                {"name": "metadata"},
                {"name": "created_at"},
            ],
        },
    )
    client = SynapCores()
    hits = client.memory.recall("default", "q")
    assert hits[0].metadata is None


def test_recall_rejects_invalid_top_k(mock_httpx_client: Any) -> None:
    inner = Mock()
    mock_httpx_client.return_value = inner
    client = SynapCores()

    for bad in (0, -1, "5", 1.5, True):
        with pytest.raises(MemoryError):
            client.memory.recall("default", "q", top_k=bad)  # type: ignore[arg-type]
    inner.post.assert_not_called()


def test_forget_returns_true_on_deletion(mock_httpx_client: Any) -> None:
    inner = _mock_engine_response(
        mock_httpx_client,
        {
            "rows": [[True]],
            "columns": [{"name": "deleted"}],
        },
    )

    client = SynapCores()
    assert client.memory.forget("default", "mem_xxx_yyy") is True
    body = inner.post.call_args.kwargs["json"]
    assert body["sql"] == "SELECT MEMORY_FORGET($1, $2) AS deleted"
    assert body["parameters"] == ["default", "mem_xxx_yyy"]


def test_forget_returns_false_when_no_row_matched(
    mock_httpx_client: Any,
) -> None:
    _mock_engine_response(
        mock_httpx_client,
        {
            "rows": [[False]],
            "columns": [{"name": "deleted"}],
        },
    )
    client = SynapCores()
    assert client.memory.forget("default", "mem_does_not_exist") is False


def test_forget_rejects_empty_id(mock_httpx_client: Any) -> None:
    inner = Mock()
    mock_httpx_client.return_value = inner
    client = SynapCores()
    with pytest.raises(MemoryError):
        client.memory.forget("default", "")
    inner.post.assert_not_called()


# ---------------------------------------------------------------------------
# Live integration tests (require a real gateway)
# ---------------------------------------------------------------------------


_LIVE_OPT_IN = os.environ.get("SYNAPCORES_TEST_LIVE", "").lower() in {
    "1",
    "true",
    "yes",
}
_LIVE_HOST = os.environ.get("SYNAPCORES_TEST_HOST", "localhost")
_LIVE_PORT = int(os.environ.get("SYNAPCORES_TEST_PORT", "8080"))
_LIVE_API_KEY = os.environ.get("SYNAPCORES_TEST_API_KEY") or os.environ.get(
    "AIDB_API_KEY"
)


def _live_client_or_skip() -> SynapCores:
    if not _LIVE_OPT_IN:
        pytest.skip(
            "Live engine tests skipped (set SYNAPCORES_TEST_LIVE=1 to enable)"
        )
    try:
        client = SynapCores(
            host=_LIVE_HOST,
            port=_LIVE_PORT,
            api_key=_LIVE_API_KEY,
            timeout=30.0,
        )
        # Touch a cheap endpoint so we fail fast if nothing's listening.
        client.execute_query("SELECT 1", parameters=[])
    except Exception as e:
        pytest.skip(f"Live engine unreachable at {_LIVE_HOST}:{_LIVE_PORT}: {e}")
    return client


@pytest.fixture
def live_namespace() -> str:
    """Unique per-test namespace so concurrent runs don't collide."""
    return f"sdk_test_{secrets.token_hex(4)}"


@pytest.mark.live
def test_live_store_returns_memory_id(live_namespace: str) -> None:
    """Spec test #1: ``store`` returns a string matching the id regex."""
    client = _live_client_or_skip()
    import re

    memory_id = client.memory.store(live_namespace, "I prefer Python over Java")
    assert isinstance(memory_id, str)
    assert re.match(r"^mem_[a-z0-9]+_[a-zA-Z0-9]+$", memory_id), memory_id


@pytest.mark.live
def test_live_store_then_recall_similar(live_namespace: str) -> None:
    """Spec test #2: store, then recall similar text, sim > 0.5."""
    client = _live_client_or_skip()
    client.memory.store(live_namespace, "I prefer Python over Java")
    # Embedding pipeline can be eventually-consistent; allow a brief wait.
    time.sleep(0.5)
    hits = client.memory.recall(
        live_namespace, "what language do I like", top_k=5
    )
    assert hits, "expected at least one recall hit"
    assert hits[0].similarity > 0.5


@pytest.mark.live
def test_live_store_with_metadata_then_recall(live_namespace: str) -> None:
    """Spec test #3: metadata round-trips as a parsed dict."""
    client = _live_client_or_skip()
    client.memory.store(
        live_namespace,
        "Customer renewed annual plan",
        metadata={"importance": 0.9, "source": "test"},
    )
    time.sleep(0.5)
    hits = client.memory.recall(live_namespace, "customer renewal")
    assert hits
    assert hits[0].metadata is not None
    assert hits[0].metadata.get("importance") == 0.9
    assert hits[0].metadata.get("source") == "test"


@pytest.mark.live
def test_live_recall_on_empty_namespace(live_namespace: str) -> None:
    """Spec test #4: recall on an empty namespace returns ``[]``."""
    client = _live_client_or_skip()
    hits = client.memory.recall(live_namespace, "anything")
    assert hits == []


@pytest.mark.live
def test_live_forget_round_trip(live_namespace: str) -> None:
    """Spec tests #5 + #6: forget returns True/False; recall no longer sees it."""
    client = _live_client_or_skip()
    memory_id = client.memory.store(live_namespace, "ephemeral fact")
    time.sleep(0.5)
    assert client.memory.forget(live_namespace, memory_id) is True
    # Second delete on the same id is a no-op.
    assert client.memory.forget(live_namespace, memory_id) is False

    # Spec #6: the deleted row should no longer surface.
    time.sleep(0.5)
    hits = client.memory.recall(live_namespace, "ephemeral fact", top_k=10)
    assert all(record.id != memory_id for record in hits)


if __name__ == "__main__":
    pytest.main([__file__])
