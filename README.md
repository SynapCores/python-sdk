# SynapCores Python SDK

> **Questions or feedback?** Join the [SynapCores community discussions](https://github.com/SynapCores/synapcores-docs/discussions) — single hub for the engine, both SDKs, OpenClaw plugin, and demos.


Official Python SDK for SynapCores - The AI-Native Database Management System.

> **0.3.0 — agent memory.** New `client.memory` sub-client wraps the
> v1.8.5+ engine's `MEMORY_STORE` / `MEMORY_RECALL` / `MEMORY_FORGET`
> SQL functions. Namespaced store, semantic recall with similarity,
> and id-based forget — the surface the OSS aerospace-rca demo, the
> OpenClaw plugin, and the planned Hermes plugin all share.
>
> ```python
> # Agent memory
> mem_id = client.memory.store("default", "User prefers Python")
> hits = client.memory.recall(
>     "default", "preferred programming language", top_k=3
> )
> client.memory.forget("default", mem_id)
> ```

> **0.2.1 — gateway response + graph fixes.** Bug-fix release: `_handle_response`
> now unwraps the gateway's `{"data": …, "meta": …}` success envelope (so
> `sql()` returns rows instead of an empty result) and tolerates empty `200`
> bodies from `DELETE`; `QueryResult.columns` is populated with column names;
> and the graph client matches the gateway contract — `nodes.create` sends
> `labels`, `edges.create` sends `src`/`dst`, `graph.cypher()` sends `{"sql": …}`,
> and `edges.delete()` was added.

> **0.2.0 — gateway v1.5.0-ce alignment.** This release rewires every
> module against the v1.5.0-ce route surface. AutoML moved from `/ai/*`
> to `/automl/*`, vector ops collapsed onto `/vector-algebra/operation`,
> WebSocket auth now uses ticket exchange, API keys are sent via the
> `X-API-Key` header (and accept the new `ak_*` prefix in addition to
> the legacy `aidb_*`), and we ship new top-level modules: `graph`,
> `nl2sql`, `filesystem`, `chat`, `multimodal`, `system`,
> `transactions`, `mcp`, `recipes`, and `schema`.

## What's new in 0.2.0

```python
from synapcores import SynapCores

client = SynapCores(host="localhost", port=8080, api_key="ak_prod_xxx...")

# 1. Cypher-style graph traversal
friends = client.graph.cypher(
    "MATCH (u:User {id:$id})-[:FRIEND]->(f) RETURN f",
    {"id": "u-123"},
)

# 2. Natural language → SQL (with optional execution)
ans = client.nl2sql.ask(
    "top 10 customers by revenue this quarter",
    execute=True,
)
print(ans["sql"], ans.get("rows"))

# 3. AI chat sessions with streaming
session = client.chat.sessions.create(model="gpt-4o")
for chunk in client.chat.stream(session["id"], "Summarize today's alerts"):
    if chunk.get("delta"):
        print(chunk["delta"], end="", flush=True)

# 4. Filesystem-backed RAG collections with progress
fs = client.filesystem.collections.create(name="docs", path="/data/docs", watch=True)
for evt in client.filesystem.collections.subscribe_progress(fs["id"]):
    print(evt.get("status"), evt.get("progress"), evt.get("filename"))

# 5. Server-side transactions with savepoints
with client.transactions.begin(isolation_level="SERIALIZABLE") as tx:
    tx.execute("UPDATE accounts SET balance = balance - $1 WHERE id = $2", [100, "a"])
    tx.savepoint("mid")
    tx.execute("UPDATE accounts SET balance = balance + $1 WHERE id = $2", [100, "b"])
    # commit() runs implicitly on clean exit; rollback() on exception.
```

Other new surfaces:

- `client.multimodal.{similarity,search,join,embed}` for cross-modal retrieval.
- `client.system.vision.{get,set,delete,test}` for the admin vision config.
- `client.mcp.{invoke,batch,info}` for the Model Context Protocol gateway.
- `client.recipes.list_categories()/list_templates()/execute_template()/list_executions()`.
- `client.schema.list_databases()/preview_table()`.

### Breaking changes vs 0.1.0

- `embed()` routes to `/ai/embeddings` (single) or `/ai/embeddings/batch`.
- `automl.*` paths moved from `/ai/*` to `/automl/*`.
- All vector math goes through `POST /vector-algebra/operation` with an `op` discriminator.
- KNN/range/hybrid search routes to `/vectors/collections/:name/search` with a `mode` field.
- API keys are sent as the `X-API-Key` header (was previously `Authorization: Bearer`).
- WebSocket subscriptions now use ticket exchange via `POST /v1/ws/ticket` and connect to `/ws?token=...`.
- `nlp.analyze()` is implemented client-side as parallel calls to `/ai/sentiment`, `/ai/entities`, and `/ai/summarize` since `/ai/analyze` was removed.
- The `client.sql(query, params=...)` payload now matches `POST /v1/query/execute` (positional `parameters`); `params` is still accepted as a dict for backwards compatibility.

## Features

- **AI-Native Operations**: Built-in support for embeddings, vector search, and semantic analysis
- **Document & Vector Storage**: Seamlessly work with both structured and unstructured data
- **SQL with AI Extensions**: Use familiar SQL syntax enhanced with AI operations
- **Real-time Subscriptions**: WebSocket support for live data updates
- **AutoML Integration**: Automated machine learning model training and deployment
- **Type-Safe**: Full type hints and Pydantic models for better IDE support

## Installation

```bash
pip install synapcores
```

## Quick Start

```python
from synapcores import SynapCores

# Initialize client with API key authentication
# API keys can be created from your AIDB dashboard at Settings > API Keys
# Note: API keys must start with 'aidb_' prefix
client = SynapCores(
    host="localhost",
    port=8080,
    api_key="aidb_sk_your_api_key_here"  # Replace with your actual API key
)

# Create a collection
collection = client.create_collection(
    name="products",
    schema={
        "name": "string",
        "description": "string",
        "price": "float",
        "embedding": "vector[384]"
    }
)

# Insert documents
collection.insert([
    {
        "name": "Laptop",
        "description": "High-performance laptop for developers",
        "price": 1299.99
    },
    {
        "name": "Mouse",
        "description": "Ergonomic wireless mouse",
        "price": 49.99
    }
])

# Semantic search
results = collection.search(
    query="computer accessories",
    top_k=5
)

# SQL with AI extensions
df = client.sql("""
    SELECT name, price, 
           similarity(embedding, embed('perfect for coding')) as relevance
    FROM products
    WHERE price < 1500
    ORDER BY relevance DESC
    LIMIT 10
""")

# Real-time subscriptions
async def handle_update(change):
    print(f"Document {change.op}: {change.document}")

subscription = await collection.subscribe(
    filter={"price": {"$lt": 100}},
    on_change=handle_update
)
```

## Advanced Features

### Vector Operations

```python
# Generate embeddings
embedding = client.embed("High-quality mechanical keyboard")

# Vector similarity search
similar_products = collection.vector_search(
    vector=embedding,
    top_k=10,
    filter={"price": {"$between": [50, 200]}}
)
```

### AutoML

```python
# Train a model
model = client.automl.train(
    collection="sales_data",
    target="revenue",
    features=["product_category", "season", "price"],
    task="regression"
)

# Make predictions
predictions = model.predict({
    "product_category": "electronics",
    "season": "holiday",
    "price": 299.99
})
```

### NLP Analysis

```python
# Analyze text
analysis = client.nlp.analyze(
    text="This product exceeded my expectations. Highly recommend!",
    tasks=["sentiment", "entities", "summary"]
)

print(f"Sentiment: {analysis.sentiment.label} ({analysis.sentiment.score})")
print(f"Entities: {[e.text for e in analysis.entities]}")
```

## Documentation

For detailed documentation, visit [https://synapcores.com/developers](https://synapcores.com/developers)

## License

MIT License - see LICENSE file for details.