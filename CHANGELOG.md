# Changelog

All notable changes to the SynapCores Python SDK.

## [0.4.0] — 2026-05-18 — vector-subsystem + auth alignment with v1.6.5.2-ce gateway

Closes three wire-format gaps surfaced during the OpenClaw v0.1.0 integration.
Validation against `validate-python-sdk-v04.py` goes from **8/9** under 0.3.0
to **11/12** under 0.4.0 (#5 — Cypher `$param` — remains the documented
gateway-side blocker).

### Fixed

- **Auth: `api_key` now sends `Authorization: Bearer <key>` instead of
  `X-API-Key`** (`synapcores/client.py::_build_headers`). Gateway
  v1.6.5.2-ce only honours the `Authorization` header for both JWTs
  and AIDB-issued API keys (`aidb_*` / `ak_*`); the legacy
  `X-API-Key` shim was rejected with HTTP 401 `missing_authorization`,
  forcing callers to manually promote API keys into `jwt_token`. The
  SDK now sends `Bearer` for every credential type — `api_key='aidb_...'`
  Just Works.
- **`Collection.vector_search` now hits the v1.6.x vector subsystem**
  (`synapcores/collection.py`). The legacy `/v1/collections/{name}/vector_search`
  route was removed in gateway v1.6.0; v0.3.0 still pointed there from
  the `Collection` class and would 404 / 500. v0.4.0 routes through
  `POST /v1/vectors/collections/{name}/search` with `{vector, k,
  include_metadata, filter?}` and normalises the response envelope.
- **`Document` model accepts vector-collection hit shapes**
  (`synapcores/models.py`). `data` is now optional and the model
  carries `values`, `metadata`, `score`, `distance` so the same class
  can hold either a document-store row or a vector-collection hit.

### Added

- **`client.create_vector_collection(name, dimensions, distance_metric)`** —
  posts to `POST /v1/vectors/collections` with the gateway's expected
  body shape. The legacy `create_collection(name, vector_size=...)`
  silently dropped `vector_size` and posted to the document-store
  subsystem, leaving the collection invisible to vector search.
- **`client.vector_collection(name)`** — synchronous typed handle for an
  existing vector collection. Returns a `VectorCollection` wired to
  `/v1/vectors/collections/{name}/...`, distinct from the document-store
  `client.collection(name)` accessor. (Behavior change from 0.3.0:
  `client.collection(name)` now returns a `Collection`, matching the
  Node SDK; in 0.3.0 it returned `VectorCollection`.)
- **`VectorCollection` expanded surface** — adds `search`, `get`,
  `count` to the existing `insert`, `delete`, `info`. `insert` now
  accepts either a single vector dict or a list (matching the Node
  SDK's `VectorCollection.insert`).
- **`client.list_vector_collections()`** and
  **`client.delete_vector_collection(name)`** — sibling helpers for the
  vector subsystem.

### Documentation

- README now lists v1.6.5.2-ce as the verified-against gateway version.

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
