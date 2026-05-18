"""
Live smoke tests against a running SynapCores gateway.

Mirrors ``validate-python-sdk.py`` so CI can exercise the same wire
contracts after a release. Gated behind ``AIDB_LIVE_TEST=1`` so it is a
no-op for ordinary unit-test runs.

Usage::

    AIDB_LIVE_TEST=1 AIDB_JWT=<token> \\
        AIDB_HOST=127.0.0.1 AIDB_PORT=28095 \\
        pytest tests/test_live_smoke.py -v
"""

from __future__ import annotations

import os

import pytest

from synapcores import SynapCores

LIVE = os.environ.get("AIDB_LIVE_TEST") == "1"
JWT = os.environ.get("AIDB_JWT")
HOST = os.environ.get("AIDB_HOST", "127.0.0.1")
PORT = int(os.environ.get("AIDB_PORT", "28095"))

pytestmark = pytest.mark.skipif(
    not LIVE or not JWT,
    reason="AIDB_LIVE_TEST=1 + AIDB_JWT required for live smoke",
)


@pytest.fixture(scope="module")
def client() -> SynapCores:
    return SynapCores(
        host=HOST,
        port=PORT,
        jwt_token=JWT,
        use_https=False,
    )


# ── Tier 1: foundational ──
def test_sql_select_one(client: SynapCores) -> None:
    r = client.sql("SELECT 1 as one")
    assert r is not None


def test_execute_query(client: SynapCores) -> None:
    r = client.execute_query(sql="SELECT 1 as one", parameters=[])
    assert r is not None
    # ``rows`` is the standard QueryResult shape; tolerate either an
    # attribute or a key for forwards-compatibility.
    rows = getattr(r, "rows", None) if not isinstance(r, dict) else r.get("rows")
    assert rows is not None


# ── Tier 2: collections ──
def test_list_collections(client: SynapCores) -> None:
    collections = client.list_collections()
    assert isinstance(collections, list)


# ── Tier 3: graph ──
def test_cypher_no_params(client: SynapCores) -> None:
    r = client.graph.cypher("MATCH (n) RETURN n LIMIT 1")
    assert r is not None


def test_graphs_list(client: SynapCores) -> None:
    # v0.3.0: URL is /graph/graphs, not /graphs.
    graphs = client.graph.graphs.list()
    assert isinstance(graphs, list)


# ── Tier 4: AutoML ──
def test_automl_list_models(client: SynapCores) -> None:
    # v0.3.0: gateway returns the list directly under "data"; SDK must
    # not assume a dict envelope.
    models = client.automl.list_models()
    assert isinstance(models, list)


def test_automl_predict_mcp_churn(client: SynapCores) -> None:
    # v0.3.0: get_model tolerates 404 (e.g. built-in MCP models) so the
    # subsequent predict() call can still hit /predict.
    model = client.automl.get_model("mcp_churn")
    result = model.predict(
        [{"tenure_months": 41, "visits_30d": 3, "spend_30d": 12.51}]
    )
    assert result is not None


# ── Tier 5: vector collections ──
def test_collection_vector_search(client: SynapCores) -> None:
    # v0.3.0: ``client.collection(name)`` returns a VectorCollection
    # that hits /v1/vectors/collections/{name}/search with the body
    # field named ``k`` (not ``top_k``).
    coll = client.collection("verify_members")
    result = coll.vector_search(
        vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0],
        top_k=3,
    )
    # Empty list is acceptable (no seeded vectors); the test asserts the
    # call succeeded end-to-end and returned a list-like payload.
    assert result is not None
