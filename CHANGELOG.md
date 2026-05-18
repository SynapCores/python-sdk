# Changelog

All notable changes to the SynapCores Python SDK.

## [0.3.0] — 2026-05-18 — wire-format alignment with v1.6.5.1-ce gateway

This release re-aligns the SDK with the live `:28095` gateway after
the v1.6.5.1-ce wire changes. Validation against the canonical script
(`validate-python-sdk.py`) goes from **4/9 PASS** under v0.2.1 to
**8/9 PASS** under v0.3.0 (the single remaining failure is a
gateway-side Cypher `$param` bug tracked separately).

### Fixed

- **`client.graph.graphs.list()` / `create()` / `get()` / `delete()` URL**
  (`synapcores/graph.py`). The graphs-metadata API moved to
  `/v1/graph/graphs`; the SDK was still hitting `/v1/graphs`, so every
  call returned `404 NotFoundError`. The `_GraphsApi` class now uses
  the correct mount and tolerates either a `{"graphs": [...]}` or
  `{"items": [...]}` response envelope.
- **`client.automl.list_models()` response parsing**
  (`synapcores/automl.py`). The gateway returns the model list
  directly under the envelope (`{"data": [...], "meta": {...}}`),
  which after `_unwrap` is a bare `list`. v0.2.1 called `.get(...)`
  on it and crashed with `AttributeError`. v0.3.0 handles both list
  and dict shapes. The same defensive parsing is applied to
  `list_training_jobs()`.
- **`client.automl.get_model(id).predict(...)` against built-in models**
  (`synapcores/automl.py`). Built-in / MCP-registered models such as
  `mcp_churn` expose `POST /predict` but have no `GET /models/{id}`
  record, so `get_model` would raise `NotFoundError` before `predict`
  could run. `get_model` now treats 404 as a stub: it returns a
  minimal `ModelInfo` keyed on the supplied id so `predict()` and
  `evaluate()` still reach the gateway.

### Added

- **`client.collection(name)` accessor** + new
  `synapcores/vector_collection.py` module. Returns a
  `VectorCollection` wired to the gateway's
  `/v1/vectors/collections/{name}/...` routes. Exposes:
  - `vector_search(vector, top_k=10, filter=None, include_metadata=True)`
    — forwards `top_k` as the wire field `k` (the gateway uses `k`,
    not `top_k`).
  - `insert(vectors)`, `delete(ids)`, `info()`.
  Mirrors the `client.collection(name)` accessor already shipped in
  the Node SDK. The legacy `Collection` class (`/v1/collections/...`)
  is unchanged.
- `tests/test_live_smoke.py` — pytest-runnable equivalent of the
  release validation script, gated on `AIDB_LIVE_TEST=1` so it is a
  no-op for ordinary unit-test runs.
- `[tool.setuptools.packages.find]` block in `pyproject.toml` so
  modern setuptools picks up the package automatically.

### Wire-shape findings (for the gateway team)

While verifying these fixes against `:28095`:

- `GET /v1/graph/graphs` returns `{"graphs": [...], "count": N}` —
  no `data`/`meta` envelope.
- `GET /v1/automl/models` returns `{"data": [...], "meta": {...}}`
  with the model array directly under `data` (i.e. no inner
  `{"models": [...]}` wrapper).
- `POST /v1/automl/models/mcp_churn/predict` returns
  `{"predictions": [...], "model_id": "..."}` — also no envelope.
- `POST /v1/vectors/collections/{name}/search` body uses `k`, not
  `top_k`, and the route is under `/vectors/collections/`, not
  `/collections/`.
- `GET /v1/automl/models/mcp_churn` returns `404 Model not found`
  even though `/predict` works — built-in MCP models are not
  enumerable through the standard model registry.

## [0.2.1] — Previous

Prior baseline. See git history for full notes.
