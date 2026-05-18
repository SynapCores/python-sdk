"""
Main client for SynapCores Python SDK.
"""

from typing import Optional, Dict, Any, List, Union, Literal
import httpx
from datetime import datetime
import json
import pandas as pd
import numpy as np

from .collection import Collection
from .exceptions import (
    ConnectionError,
    AuthenticationError,
    ValidationError,
    NotFoundError,
    ServerError,
    RateLimitError,
    SynapCoresError,
    SQLError,
    VectorError,
    TransactionError,
    BatchOperationError,
    IndexError,
    ConstraintError,
    PreparedStatementError,
)
from .models import (
    QueryResult,
    ModelInfo,
    NLPAnalysis,
    ColumnDefinition,
    CreateTableOptions,
    AlterTableOptions,
    IndexDefinition,
    TableInfo,
    IndexInfo,
    TransactionOptions,
    TransactionContext,
    SavepointInfo,
    BatchInsertOptions,
    BatchUpdateOptions,
    BatchDeleteOptions,
    BatchResult,
    PreparedStatement,
    PreparedStatementOptions,
    CTEDefinition,
    WindowFunction,
    JSONOperation,
    Vector,
    VectorArithmeticResult,
    VectorSimilarityResult,
    VectorMagnitudeResult,
    VectorSearchResult,
    KNNSearchOptions,
    RangeSearchOptions,
    HybridSearchOptions,
)
from .automl import AutoMLClient
from .nlp import NLPClient
from .vector_ops import VectorOperationsMixin


class SynapCores(VectorOperationsMixin):
    """
    Main client for interacting with SynapCores database.
    
    Args:
        host: Server hostname
        port: Server port
        api_key: API key for authentication
        use_https: Whether to use HTTPS
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        use_https: bool = False,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.use_https = use_https
        self.timeout = timeout
        self.max_retries = max_retries

        # Validate API key format if provided. v0.2.0: gateway accepts both
        # `ak_*` (production / dashboard) and the legacy `aidb_*` keys.
        if api_key and not (api_key.startswith("ak_") or api_key.startswith("aidb_")):
            raise ValueError(
                "Invalid API key format. API keys should start with 'ak_' or 'aidb_' prefix. "
                "Please create a valid API key from your AIDB dashboard."
            )

        # Build base URL
        protocol = "https" if use_https else "http"
        self.base_url = f"{protocol}://{host}:{port}/v1"

        # Initialize HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._build_headers(),
        )

        # Initialize sub-clients
        self.automl = AutoMLClient(self)
        self.nlp = NLPClient(self)

        # v0.2.0 modules — local imports to avoid circular references at
        # module import time.
        from .graph import GraphClient
        from .nl2sql import NL2SqlClient
        from .filesystem import FilesystemCollectionsClient
        from .chat import ChatClient
        from .multimodal import MultimodalClient
        from .system import SystemClient
        from .transactions import TransactionsClient
        from .mcp import McpClient
        from .recipes import RecipeClient
        from .schema import SchemaClient

        self.graph = GraphClient(self)
        self.nl2sql = NL2SqlClient(self)
        self.filesystem = FilesystemCollectionsClient(self)
        self.chat = ChatClient(self)
        self.multimodal = MultimodalClient(self)
        self.system = SystemClient(self)
        self.transactions = TransactionsClient(self)
        self.mcp = McpClient(self)
        self.recipes = RecipeClient(self)
        self.schema = SchemaClient(self)

        # Cache for collections
        self._collections_cache: Dict[str, Collection] = {}

        # Transaction management
        self._current_transaction: Optional[TransactionContext] = None
        self._prepared_statements: Dict[str, PreparedStatement] = {}
        self._savepoints: Dict[str, SavepointInfo] = {}
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers.

        v0.2.0: gateway expects API keys in the X-API-Key header (not
        the Authorization header). JWT tokens still go through the
        Bearer scheme.
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "synapcores-python/0.3.0",
        }
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        elif self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    # -----------------------------------------------------------------
    # v0.2.0 helpers reused by the new sub-modules.
    # -----------------------------------------------------------------
    def _ws_base_url(self) -> str:
        """Return the WebSocket base URL (ws:// or wss://)."""
        protocol = "wss" if self.use_https else "ws"
        return f"{protocol}://{self.host}:{self.port}"

    def create_ws_ticket(self) -> Dict[str, Any]:
        """Exchange the current credentials for a short-lived WS ticket.

        Returns the raw payload, including ``token`` and ``expiresAt``.
        Use the token as the ``?token=`` query parameter on any /ws URL.
        """
        response = self._client.post("/ws/ticket", json={})
        return self._handle_response(response)

    def revoke_ws_ticket(self, token: str) -> None:
        """Revoke a previously-issued ticket (best-effort)."""
        response = self._client.post("/ws/ticket/revoke", json={"token": token})
        self._handle_response(response)

    def set_jwt_token(self, token: str) -> None:
        """Switch the client over to JWT auth at runtime."""
        self.jwt_token = token
        # Refresh headers so subsequent requests pick up the new auth.
        self._client.headers.update(self._build_headers())
        # Drop the X-API-Key header if it was set previously.
        if "X-API-Key" in self._client.headers:
            del self._client.headers["X-API-Key"]

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login with username/password and store the returned JWT."""
        response = self._client.post(
            "/auth/login",
            json={"username": username, "password": password},
        )
        data = self._handle_response(response)
        token = data.get("access_token")
        if token:
            self.set_jwt_token(token)
        return data

    def logout(self) -> None:
        """Forget the current auth state."""
        self.jwt_token = None
        for header in ("Authorization", "X-API-Key"):
            if header in self._client.headers:
                del self._client.headers[header]
    
    @staticmethod
    def _unwrap(payload: Any) -> Any:
        """Unwrap the gateway's ``{"data": ..., "meta": ...}`` success envelope.

        The CE gateway wraps every 2xx body as ``{"data": <payload>,
        "meta": {...}}``. Callers expect the payload directly, so peel
        the envelope when its signature is present and leave any other
        shape untouched.
        """
        if isinstance(payload, dict) and "data" in payload and "meta" in payload:
            return payload["data"]
        return payload

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and errors."""
        if response.status_code in (200, 201):
            # DELETE and some mutations return 200 with an empty body.
            if not response.content:
                return {}
            return self._unwrap(response.json())
        elif response.status_code == 204:
            return {}
        elif response.status_code == 401:
            raise AuthenticationError("Authentication failed", code="AUTH_FAILED")
        elif response.status_code == 403:
            raise AuthenticationError("Access forbidden", code="FORBIDDEN")
        elif response.status_code == 404:
            raise NotFoundError("Resource not found", code="NOT_FOUND")
        elif response.status_code == 422:
            error_data = response.json()
            raise ValidationError(
                error_data.get("message", "Validation failed"),
                code="VALIDATION_ERROR",
                details=error_data.get("errors", {}),
            )
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                "Rate limit exceeded",
                retry_after=int(retry_after) if retry_after else None,
            )
        elif response.status_code >= 500:
            raise ServerError(
                f"Server error: {response.status_code}",
                code="SERVER_ERROR",
            )
        else:
            raise SynapCoresError(
                f"Unexpected status code: {response.status_code}",
                code="UNEXPECTED_ERROR",
            )
    
    def create_collection(
        self,
        name: str,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Collection:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            schema: Collection schema definition
            **kwargs: Additional collection options
            
        Returns:
            Collection instance
        """
        payload = {
            "name": name,
            "schema": schema or {},
            **kwargs,
        }
        
        response = self._client.post("/collections", json=payload)
        data = self._handle_response(response)
        
        collection = Collection(self, name, data.get("schema"))
        self._collections_cache[name] = collection
        return collection
    
    def get_collection(self, name: str) -> Collection:
        """
        Get existing collection.
        
        Args:
            name: Collection name
            
        Returns:
            Collection instance
        """
        if name in self._collections_cache:
            return self._collections_cache[name]
        
        response = self._client.get(f"/collections/{name}")
        data = self._handle_response(response)
        
        collection = Collection(self, name, data.get("schema"))
        self._collections_cache[name] = collection
        return collection
    
    def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List of collection names
        """
        response = self._client.get("/collections")
        data = self._handle_response(response)
        return data.get("collections", [])
    
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.

        Args:
            name: Collection name
        """
        response = self._client.delete(f"/collections/{name}")
        self._handle_response(response)

        if name in self._collections_cache:
            del self._collections_cache[name]

    def collection(self, name: str) -> "VectorCollection":
        """Return a thin wrapper bound to a named vector collection.

        v0.3.0: mirrors the Node SDK's ``client.collection(name)`` accessor.
        The returned object exposes ``vector_search`` (and a few sibling
        helpers) wired to the gateway's
        ``/v1/vectors/collections/{name}/...`` routes, which is distinct
        from the document-store endpoints under ``/v1/collections/{name}``.
        """
        from .vector_collection import VectorCollection
        return VectorCollection(self, name)

    def sql(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
        as_dataframe: bool = True,
    ) -> Union[QueryResult, pd.DataFrame]:
        """
        Execute SQL query with AI extensions.

        v0.2.0: gateway endpoint moved to ``POST /v1/query/execute`` and
        expects parameters as a positional list. ``params`` may still be
        a dict for backwards compatibility — values are ordered by their
        keys' insertion order.

        Args:
            query: SQL query string
            params: Query parameters (list or dict)
            as_dataframe: Return result as pandas DataFrame

        Returns:
            QueryResult or pandas DataFrame
        """
        if isinstance(params, dict):
            param_list = list(params.values())
        elif params is None:
            param_list = []
        else:
            param_list = list(params)

        payload = {
            "sql": query,
            "parameters": param_list,
        }

        response = self._client.post("/query/execute", json=payload)
        data = self._handle_response(response)

        # Normalize column shape to a list of dicts; gateway returns
        # objects like {"name": ..., "data_type": ...}.
        cols_raw = data.get("columns", []) or []
        rows_raw = data.get("rows", []) or []

        # Convert positional rows -> dict-rows when columns metadata is
        # rich enough so existing callers still work.
        col_names: List[str] = []
        for c in cols_raw:
            if isinstance(c, dict):
                col_names.append(c.get("name") or c.get("column") or "")
            else:
                col_names.append(str(c))

        if rows_raw and isinstance(rows_raw[0], list) and col_names:
            dict_rows = [
                {col_names[i] if i < len(col_names) else str(i): val for i, val in enumerate(row)}
                for row in rows_raw
            ]
        else:
            dict_rows = rows_raw

        result = QueryResult(
            rows=dict_rows,
            columns=col_names,
            row_count=data.get("rows_affected") if data.get("rows_affected") is not None else len(dict_rows),
            took_ms=data.get("execution_time_ms") or data.get("took_ms") or 0,
            query_plan=data.get("query_plan"),
        )

        if as_dataframe and result.rows:
            return pd.DataFrame(result.rows)
        return result

    def execute_query(
        self,
        sql: str,
        parameters: Optional[List[Any]] = None,
        max_rows: int = 1000,
        timeout_secs: int = 300,
    ) -> Dict[str, Any]:
        """Direct, gateway-shaped SQL exec returning the raw JSON payload."""
        response = self._client.post(
            "/query/execute",
            json={
                "sql": sql,
                "parameters": parameters or [],
                "max_rows": max_rows,
                "timeout_secs": timeout_secs,
            },
        )
        return self._handle_response(response)

    def execute_batch_queries(
        self,
        queries: List[Dict[str, Any]],
        transactional: bool = False,
    ) -> Dict[str, Any]:
        """POST /v1/query/execute/batch."""
        response = self._client.post(
            "/query/execute/batch",
            json={"queries": queries, "transactional": transactional},
        )
        return self._handle_response(response)

    def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text.

        v0.2.0: routes through ``POST /v1/ai/embeddings`` for a single
        string and ``POST /v1/ai/embeddings/batch`` for a list.
        """
        is_batch = isinstance(text, list)
        if is_batch:
            payload: Dict[str, Any] = {"texts": text}
            if model:
                payload["model"] = model
            response = self._client.post("/ai/embeddings/batch", json=payload)
            data = self._handle_response(response)
            return data.get("embeddings") or data
        else:
            payload = {"text": text}
            if model:
                payload["model"] = model
            response = self._client.post("/ai/embeddings", json=payload)
            data = self._handle_response(response)
            return data.get("embedding") or (data.get("embeddings") or [None])[0] or data
    
    def close(self) -> None:
        """Close the client connection."""
        self._client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # =================================================================
    # TABLE MANAGEMENT OPERATIONS
    # =================================================================

    def create_table(
        self,
        table_name: str,
        columns: List[ColumnDefinition],
        options: Optional[CreateTableOptions] = None,
    ) -> None:
        """
        Create a new table with the specified schema.

        Args:
            table_name: Name of the table to create
            columns: List of column definitions
            options: Additional table creation options

        Raises:
            SQLError: If table creation fails
            ValidationError: If schema is invalid
        """
        if not columns:
            raise ValidationError("At least one column must be specified")

        options = options or CreateTableOptions()

        # Build SQL statement
        sql_parts = ["CREATE TABLE"]
        if options.if_not_exists:
            sql_parts.append("IF NOT EXISTS")
        if options.temporary:
            sql_parts.append("TEMPORARY")

        sql_parts.append(f"{table_name} (")

        # Add column definitions
        column_defs = []
        for col in columns:
            col_def = f"{col.name} {col.data_type}"

            # Add constraints
            for constraint in col.constraints:
                if constraint.type == "PRIMARY_KEY":
                    col_def += " PRIMARY KEY"
                elif constraint.type == "UNIQUE":
                    col_def += " UNIQUE"
                elif constraint.type == "NOT_NULL":
                    col_def += " NOT NULL"
                elif constraint.type == "CHECK" and constraint.expression:
                    col_def += f" CHECK ({constraint.expression})"

            # Add default value
            if col.default_value is not None:
                if isinstance(col.default_value, str) and col.default_value.upper() in ["CURRENT_TIMESTAMP", "CURRENT_DATE", "CURRENT_TIME"]:
                    col_def += f" DEFAULT {col.default_value}"
                elif isinstance(col.default_value, bool):
                    col_def += f" DEFAULT {str(col.default_value).lower()}"
                elif isinstance(col.default_value, (int, float)):
                    col_def += f" DEFAULT {col.default_value}"
                else:
                    col_def += f" DEFAULT '{col.default_value}'"

            column_defs.append(col_def)

        # Add table constraints
        for constraint in options.constraints:
            if constraint.type == "CHECK" and constraint.expression:
                column_defs.append(f"CHECK ({constraint.expression})")

        sql = " ".join(sql_parts) + ", ".join(column_defs) + ")"

        try:
            self.sql(sql, as_dataframe=False)
        except Exception as e:
            raise SQLError(f"Failed to create table {table_name}: {str(e)}")

    def alter_table(
        self,
        table_name: str,
        options: AlterTableOptions,
    ) -> None:
        """
        Alter an existing table structure.

        Args:
            table_name: Name of the table to alter
            options: Alteration options

        Raises:
            SQLError: If table alteration fails
            NotFoundError: If table doesn't exist
        """
        sql_parts = ["ALTER TABLE", table_name]

        if options.action == "ADD_COLUMN":
            if not options.column_definition:
                raise ValidationError("Column definition required for ADD_COLUMN")

            col = options.column_definition
            col_def = f"ADD COLUMN {col.name} {col.data_type}"

            # Add constraints
            for constraint in col.constraints:
                if constraint.type == "NOT_NULL":
                    col_def += " NOT NULL"
                elif constraint.type == "UNIQUE":
                    col_def += " UNIQUE"
                elif constraint.type == "CHECK" and constraint.expression:
                    col_def += f" CHECK ({constraint.expression})"

            # Add default value
            if col.default_value is not None:
                if isinstance(col.default_value, str) and col.default_value.upper() in ["CURRENT_TIMESTAMP", "CURRENT_DATE", "CURRENT_TIME"]:
                    col_def += f" DEFAULT {col.default_value}"
                elif isinstance(col.default_value, bool):
                    col_def += f" DEFAULT {str(col.default_value).lower()}"
                elif isinstance(col.default_value, (int, float)):
                    col_def += f" DEFAULT {col.default_value}"
                else:
                    col_def += f" DEFAULT '{col.default_value}'"

            sql_parts.append(col_def)

        elif options.action == "DROP_COLUMN":
            if not options.old_column_name:
                raise ValidationError("old_column_name required for DROP_COLUMN")
            sql_parts.extend(["DROP COLUMN", options.old_column_name])

        elif options.action == "RENAME_COLUMN":
            if not options.old_column_name or not options.new_column_name:
                raise ValidationError("Both old_column_name and new_column_name required for RENAME_COLUMN")
            sql_parts.extend(["RENAME COLUMN", options.old_column_name, "TO", options.new_column_name])

        elif options.action == "RENAME_TABLE":
            if not options.new_table_name:
                raise ValidationError("new_table_name required for RENAME_TABLE")
            sql_parts = ["ALTER TABLE", table_name, "RENAME TO", options.new_table_name]

        else:
            raise ValidationError(f"Unsupported alter action: {options.action}")

        sql = " ".join(sql_parts)

        try:
            self.sql(sql, as_dataframe=False)
        except Exception as e:
            raise SQLError(f"Failed to alter table {table_name}: {str(e)}")

    def drop_table(
        self,
        table_name: str,
        if_exists: bool = False,
        cascade: bool = False,
    ) -> None:
        """
        Drop an existing table.

        Args:
            table_name: Name of the table to drop
            if_exists: Don't raise error if table doesn't exist
            cascade: Drop dependent objects as well

        Raises:
            SQLError: If table drop fails
            NotFoundError: If table doesn't exist and if_exists is False
        """
        sql_parts = ["DROP TABLE"]
        if if_exists:
            sql_parts.append("IF EXISTS")

        sql_parts.append(table_name)

        if cascade:
            sql_parts.append("CASCADE")

        sql = " ".join(sql_parts)

        try:
            self.sql(sql, as_dataframe=False)
        except Exception as e:
            if "not found" in str(e).lower() and not if_exists:
                raise NotFoundError(f"Table {table_name} not found")
            raise SQLError(f"Failed to drop table {table_name}: {str(e)}")

    def describe_table(self, table_name: str) -> TableInfo:
        """
        Get detailed information about a table structure.

        Args:
            table_name: Name of the table to describe

        Returns:
            TableInfo object with table details

        Raises:
            NotFoundError: If table doesn't exist
            SQLError: If operation fails
        """
        try:
            # Try to get table information using information_schema if available
            result = self.sql(f"""
                SELECT
                    column_name as name,
                    data_type as type,
                    is_nullable as nullable,
                    column_default as default_value,
                    ordinal_position as position
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """, as_dataframe=False)

            if result.row_count == 0:
                # Fallback to PRAGMA if SQLite or similar
                try:
                    result = self.sql(f"PRAGMA table_info({table_name})", as_dataframe=False)
                    if result.row_count == 0:
                        raise NotFoundError(f"Table {table_name} not found")

                    columns = []
                    for row in result.rows:
                        columns.append({
                            "name": row.get("name"),
                            "type": row.get("type"),
                            "nullable": row.get("notnull", 0) == 0,
                            "default_value": row.get("dflt_value"),
                            "isPrimaryKey": row.get("pk", 0) == 1,
                        })
                except Exception:
                    raise NotFoundError(f"Table {table_name} not found")
            else:
                columns = []
                for row in result.rows:
                    columns.append({
                        "name": row.get("name"),
                        "type": row.get("type"),
                        "nullable": row.get("nullable", "YES") == "YES",
                        "default_value": row.get("default_value"),
                        "isPrimaryKey": False,  # Would need separate query
                    })

            return TableInfo(
                name=table_name,
                columns=columns,
                indexes=[],  # Would need separate query
                constraints=[],  # Would need separate query
            )

        except Exception as e:
            if "not found" in str(e).lower():
                raise NotFoundError(f"Table {table_name} not found")
            raise SQLError(f"Failed to describe table {table_name}: {str(e)}")

    def show_tables(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all tables in the database.

        Args:
            pattern: Optional pattern to filter table names

        Returns:
            List of table names

        Raises:
            SQLError: If operation fails
        """
        try:
            # Try information_schema first
            try:
                sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                if pattern:
                    sql += f" AND table_name LIKE '{pattern}'"
                result = self.sql(sql, as_dataframe=False)
                return [row["table_name"] for row in result.rows]
            except Exception:
                # Fallback to SQLite system tables
                sql = "SELECT name FROM sqlite_master WHERE type='table'"
                if pattern:
                    sql += f" AND name LIKE '{pattern}'"
                result = self.sql(sql, as_dataframe=False)
                return [row["name"] for row in result.rows]

        except Exception as e:
            raise SQLError(f"Failed to list tables: {str(e)}")

    # =================================================================
    # INDEX MANAGEMENT OPERATIONS
    # =================================================================

    def create_index(self, index: IndexDefinition) -> None:
        """
        Create a database index.

        Args:
            index: Index definition

        Raises:
            SQLError: If index creation fails
            IndexError: If index already exists and if_not_exists is False
        """
        sql_parts = ["CREATE"]
        if index.unique:
            sql_parts.append("UNIQUE")
        sql_parts.append("INDEX")

        if index.if_not_exists:
            sql_parts.append("IF NOT EXISTS")

        sql_parts.append(index.name)
        sql_parts.extend(["ON", index.table_name])

        # Build column list
        column_specs = []
        for col in index.columns:
            spec = col.name
            if col.order:
                spec += f" {col.order}"
            if col.length:
                spec = f"{col.name}({col.length})"
                if col.order:
                    spec += f" {col.order}"
            column_specs.append(spec)

        sql_parts.append(f"({', '.join(column_specs)})")

        if index.where_clause:
            sql_parts.extend(["WHERE", index.where_clause])

        sql = " ".join(sql_parts)

        try:
            self.sql(sql, as_dataframe=False)
        except Exception as e:
            if "already exists" in str(e).lower() and not index.if_not_exists:
                raise IndexError(f"Index {index.name} already exists")
            raise SQLError(f"Failed to create index {index.name}: {str(e)}")

    def drop_index(self, index_name: str, if_exists: bool = False) -> None:
        """
        Drop a database index.

        Args:
            index_name: Name of the index to drop
            if_exists: Don't raise error if index doesn't exist

        Raises:
            SQLError: If index drop fails
            IndexError: If index doesn't exist and if_exists is False
        """
        sql_parts = ["DROP INDEX"]
        if if_exists:
            sql_parts.append("IF EXISTS")
        sql_parts.append(index_name)

        sql = " ".join(sql_parts)

        try:
            self.sql(sql, as_dataframe=False)
        except Exception as e:
            if "not found" in str(e).lower() and not if_exists:
                raise IndexError(f"Index {index_name} not found")
            raise SQLError(f"Failed to drop index {index_name}: {str(e)}")

    def show_indexes(self, table_name: Optional[str] = None) -> List[IndexInfo]:
        """
        List all indexes in the database or for a specific table.

        Args:
            table_name: Optional table name to filter indexes

        Returns:
            List of IndexInfo objects

        Raises:
            SQLError: If operation fails
        """
        try:
            # Try information_schema first
            try:
                sql = """
                    SELECT
                        indexname as name,
                        tablename as table,
                        indexdef as definition
                    FROM pg_indexes
                """
                if table_name:
                    sql += f" WHERE tablename = '{table_name}'"

                result = self.sql(sql, as_dataframe=False)
                indexes = []
                for row in result.rows:
                    indexes.append(IndexInfo(
                        name=row["name"],
                        table=row["table"],
                        columns=[],  # Would need to parse definition
                        unique="UNIQUE" in row.get("definition", "").upper(),
                        index_type="btree",  # Default assumption
                    ))
                return indexes

            except Exception:
                # Fallback to SQLite
                if table_name:
                    result = self.sql(f"PRAGMA index_list({table_name})", as_dataframe=False)
                else:
                    result = self.sql("SELECT name FROM sqlite_master WHERE type='index'", as_dataframe=False)

                indexes = []
                for row in result.rows:
                    if table_name:
                        indexes.append(IndexInfo(
                            name=row["name"],
                            table=table_name,
                            columns=[],  # Would need separate PRAGMA call
                            unique=bool(row.get("unique", 0)),
                            index_type="btree",
                        ))
                    else:
                        # For global index list, we need to get more info
                        indexes.append(IndexInfo(
                            name=row["name"],
                            table="",  # Unknown without additional query
                            columns=[],
                            unique=False,  # Unknown
                            index_type="btree",
                        ))
                return indexes

        except Exception as e:
            raise SQLError(f"Failed to list indexes: {str(e)}")

    # =================================================================
    # TRANSACTION MANAGEMENT
    # =================================================================

    def begin_transaction(self, options: Optional[TransactionOptions] = None) -> TransactionContext:
        """
        Begin a new database transaction.

        Args:
            options: Transaction configuration options

        Returns:
            TransactionContext with transaction details

        Raises:
            TransactionError: If transaction cannot be started
            SQLError: If another transaction is already active
        """
        if self._current_transaction:
            raise TransactionError("Transaction already active", transaction_id=self._current_transaction.id)

        options = options or TransactionOptions()

        # Build BEGIN statement
        sql_parts = ["BEGIN"]
        if options.isolation_level:
            sql_parts.extend(["ISOLATION LEVEL", options.isolation_level.value])

        if options.read_only:
            sql_parts.append("READ ONLY")

        sql = " ".join(sql_parts)

        try:
            result = self.sql(sql, as_dataframe=False)

            # Generate transaction ID
            import uuid
            transaction_id = str(uuid.uuid4())

            self._current_transaction = TransactionContext(
                id=transaction_id,
                isolation_level=options.isolation_level or "READ_COMMITTED",
                read_only=options.read_only,
                started_at=datetime.now(),
                timeout=options.timeout,
                name=options.name,
            )

            return self._current_transaction

        except Exception as e:
            raise TransactionError(f"Failed to begin transaction: {str(e)}")

    def commit_transaction(self) -> None:
        """
        Commit the current transaction.

        Raises:
            TransactionError: If no transaction is active or commit fails
        """
        if not self._current_transaction:
            raise TransactionError("No active transaction to commit")

        try:
            self.sql("COMMIT", as_dataframe=False)
            self._current_transaction = None
            self._savepoints.clear()

        except Exception as e:
            raise TransactionError(
                f"Failed to commit transaction: {str(e)}",
                transaction_id=self._current_transaction.id
            )

    def rollback_transaction(self) -> None:
        """
        Rollback the current transaction.

        Raises:
            TransactionError: If no transaction is active or rollback fails
        """
        if not self._current_transaction:
            raise TransactionError("No active transaction to rollback")

        try:
            self.sql("ROLLBACK", as_dataframe=False)
            self._current_transaction = None
            self._savepoints.clear()

        except Exception as e:
            raise TransactionError(
                f"Failed to rollback transaction: {str(e)}",
                transaction_id=self._current_transaction.id
            )

    def create_savepoint(self, name: str) -> SavepointInfo:
        """
        Create a savepoint within the current transaction.

        Args:
            name: Name of the savepoint

        Returns:
            SavepointInfo with savepoint details

        Raises:
            TransactionError: If no transaction is active or savepoint creation fails
        """
        if not self._current_transaction:
            raise TransactionError("No active transaction for savepoint")

        try:
            self.sql(f"SAVEPOINT {name}", as_dataframe=False)

            savepoint = SavepointInfo(
                name=name,
                created_at=datetime.now(),
                transaction_id=self._current_transaction.id,
            )

            self._savepoints[name] = savepoint
            return savepoint

        except Exception as e:
            raise TransactionError(f"Failed to create savepoint {name}: {str(e)}")

    def rollback_to_savepoint(self, name: str) -> None:
        """
        Rollback to a specific savepoint.

        Args:
            name: Name of the savepoint

        Raises:
            TransactionError: If savepoint doesn't exist or rollback fails
        """
        if name not in self._savepoints:
            raise TransactionError(f"Savepoint {name} not found")

        try:
            self.sql(f"ROLLBACK TO SAVEPOINT {name}", as_dataframe=False)

        except Exception as e:
            raise TransactionError(f"Failed to rollback to savepoint {name}: {str(e)}")

    def release_savepoint(self, name: str) -> None:
        """
        Release a savepoint.

        Args:
            name: Name of the savepoint

        Raises:
            TransactionError: If savepoint doesn't exist or release fails
        """
        if name not in self._savepoints:
            raise TransactionError(f"Savepoint {name} not found")

        try:
            self.sql(f"RELEASE SAVEPOINT {name}", as_dataframe=False)
            del self._savepoints[name]

        except Exception as e:
            raise TransactionError(f"Failed to release savepoint {name}: {str(e)}")

    def get_current_transaction(self) -> Optional[TransactionContext]:
        """
        Get the current active transaction context.

        Returns:
            TransactionContext if transaction is active, None otherwise
        """
        return self._current_transaction

    # =================================================================
    # BATCH OPERATIONS
    # =================================================================

    def batch_insert(self, options: BatchInsertOptions) -> BatchResult:
        """
        Perform batch insert operations with error tracking.

        Args:
            options: Batch insert configuration

        Returns:
            BatchResult with operation statistics

        Raises:
            BatchOperationError: If batch operation fails
            ValidationError: If options are invalid
        """
        if not options.rows:
            raise ValidationError("No rows provided for batch insert")

        if len(options.columns) == 0:
            raise ValidationError("No columns specified for batch insert")

        start_time = datetime.now()
        successful = 0
        failed = 0
        errors = []

        # Build base SQL
        placeholders = ", ".join([f"${i+1}" for i in range(len(options.columns))])
        base_sql = f"INSERT INTO {options.table_name} ({', '.join(options.columns)}) VALUES ({placeholders})"

        if options.on_conflict == "REPLACE":
            base_sql = base_sql.replace("INSERT", "INSERT OR REPLACE")
        elif options.on_conflict == "IGNORE":
            base_sql = base_sql.replace("INSERT", "INSERT OR IGNORE")

        # Process in batches
        for batch_start in range(0, len(options.rows), options.batch_size):
            batch_end = min(batch_start + options.batch_size, len(options.rows))
            batch_rows = options.rows[batch_start:batch_end]

            for i, row_data in enumerate(batch_rows):
                try:
                    # Convert row to proper format
                    if isinstance(row_data, dict):
                        # If row is a dict, extract values in column order
                        params = {str(j+1): row_data.get(col) for j, col in enumerate(options.columns)}
                    elif isinstance(row_data, (list, tuple)):
                        # If row is a list/tuple, map to positional parameters
                        params = {str(j+1): val for j, val in enumerate(row_data)}
                    else:
                        raise ValueError(f"Invalid row data type: {type(row_data)}")

                    self.sql(base_sql, params=params, as_dataframe=False)
                    successful += 1

                except Exception as e:
                    failed += 1
                    if options.return_errors:
                        from .models import BatchError
                        errors.append(BatchError(
                            item_index=batch_start + i,
                            error_message=str(e),
                            error_code=getattr(e, 'code', None),
                            item_data=row_data,
                        ))

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return BatchResult(
            total_processed=len(options.rows),
            successful=successful,
            failed=failed,
            errors=errors,
            took_ms=took_ms,
        )

    def batch_update(self, options: BatchUpdateOptions) -> BatchResult:
        """
        Perform batch update operations with error tracking.

        Args:
            options: Batch update configuration

        Returns:
            BatchResult with operation statistics

        Raises:
            BatchOperationError: If batch operation fails
            ValidationError: If options are invalid
        """
        if not options.updates:
            raise ValidationError("No updates provided for batch update")

        start_time = datetime.now()
        successful = 0
        failed = 0
        errors = []

        # Process in batches
        for batch_start in range(0, len(options.updates), options.batch_size):
            batch_end = min(batch_start + options.batch_size, len(options.updates))
            batch_updates = options.updates[batch_start:batch_end]

            for i, update_item in enumerate(batch_updates):
                try:
                    # Build SET clause
                    set_clauses = []
                    set_params = {}
                    param_counter = 1

                    for col, val in update_item.set.items():
                        set_clauses.append(f"{col} = ${param_counter}")
                        set_params[str(param_counter)] = val
                        param_counter += 1

                    # Build WHERE clause
                    where_clauses = []
                    for col, val in update_item.where.items():
                        where_clauses.append(f"{col} = ${param_counter}")
                        set_params[str(param_counter)] = val
                        param_counter += 1

                    sql = f"UPDATE {options.table_name} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"

                    self.sql(sql, params=set_params, as_dataframe=False)
                    successful += 1

                except Exception as e:
                    failed += 1
                    if options.return_errors:
                        from .models import BatchError
                        errors.append(BatchError(
                            item_index=batch_start + i,
                            error_message=str(e),
                            error_code=getattr(e, 'code', None),
                            item_data=update_item.dict(),
                        ))

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return BatchResult(
            total_processed=len(options.updates),
            successful=successful,
            failed=failed,
            errors=errors,
            took_ms=took_ms,
        )

    def batch_delete(self, options: BatchDeleteOptions) -> BatchResult:
        """
        Perform batch delete operations with error tracking.

        Args:
            options: Batch delete configuration

        Returns:
            BatchResult with operation statistics

        Raises:
            BatchOperationError: If batch operation fails
            ValidationError: If options are invalid
        """
        if not options.conditions:
            raise ValidationError("No conditions provided for batch delete")

        start_time = datetime.now()
        successful = 0
        failed = 0
        errors = []

        # Process in batches
        for batch_start in range(0, len(options.conditions), options.batch_size):
            batch_end = min(batch_start + options.batch_size, len(options.conditions))
            batch_conditions = options.conditions[batch_start:batch_end]

            for i, condition in enumerate(batch_conditions):
                try:
                    # Build WHERE clause
                    where_clauses = []
                    params = {}

                    for j, (col, val) in enumerate(condition.items(), 1):
                        where_clauses.append(f"{col} = ${j}")
                        params[str(j)] = val

                    sql = f"DELETE FROM {options.table_name} WHERE {' AND '.join(where_clauses)}"

                    self.sql(sql, params=params, as_dataframe=False)
                    successful += 1

                except Exception as e:
                    failed += 1
                    if options.return_errors:
                        from .models import BatchError
                        errors.append(BatchError(
                            item_index=batch_start + i,
                            error_message=str(e),
                            error_code=getattr(e, 'code', None),
                            item_data=condition,
                        ))

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return BatchResult(
            total_processed=len(options.conditions),
            successful=successful,
            failed=failed,
            errors=errors,
            took_ms=took_ms,
        )

    # =================================================================
    # PREPARED STATEMENTS
    # =================================================================

    def prepare_statement(
        self,
        query: str,
        options: Optional[PreparedStatementOptions] = None,
    ) -> PreparedStatement:
        """
        Prepare a SQL statement for repeated execution.

        Args:
            query: SQL query with parameter placeholders
            options: Preparation options

        Returns:
            PreparedStatement object

        Raises:
            PreparedStatementError: If statement preparation fails
        """
        options = options or PreparedStatementOptions()

        try:
            # In a real implementation, this would call a REST endpoint
            # For now, we'll simulate by parsing the query and storing it
            import uuid
            import re

            statement_id = options.name or str(uuid.uuid4())

            # Count parameters in query
            parameter_count = len(re.findall(r'\$\d+', query))

            prepared_stmt = PreparedStatement(
                id=statement_id,
                name=options.name,
                query=query,
                parameter_count=parameter_count,
                parameter_types=options.parameter_types,
                created_at=datetime.now(),
            )

            self._prepared_statements[statement_id] = prepared_stmt
            return prepared_stmt

        except Exception as e:
            raise PreparedStatementError(f"Failed to prepare statement: {str(e)}")

    def execute_prepared(
        self,
        statement_id: str,
        parameters: Optional[List[Any]] = None,
        as_dataframe: bool = True,
    ) -> Union[QueryResult, pd.DataFrame]:
        """
        Execute a prepared statement with parameters.

        Args:
            statement_id: ID or name of the prepared statement
            parameters: Parameter values
            as_dataframe: Return result as pandas DataFrame

        Returns:
            QueryResult or pandas DataFrame

        Raises:
            PreparedStatementError: If statement doesn't exist or execution fails
        """
        if statement_id not in self._prepared_statements:
            raise PreparedStatementError(f"Prepared statement {statement_id} not found", statement_id=statement_id)

        prepared_stmt = self._prepared_statements[statement_id]
        parameters = parameters or []

        if len(parameters) != prepared_stmt.parameter_count:
            raise PreparedStatementError(
                f"Parameter count mismatch: expected {prepared_stmt.parameter_count}, got {len(parameters)}",
                statement_id=statement_id
            )

        try:
            # Convert parameters to dict format
            params = {str(i+1): param for i, param in enumerate(parameters)}
            return self.sql(prepared_stmt.query, params=params, as_dataframe=as_dataframe)

        except Exception as e:
            raise PreparedStatementError(f"Failed to execute prepared statement: {str(e)}", statement_id=statement_id)

    def deallocate_prepared(self, statement_id: str) -> None:
        """
        Deallocate a prepared statement.

        Args:
            statement_id: ID or name of the prepared statement

        Raises:
            PreparedStatementError: If statement doesn't exist
        """
        if statement_id not in self._prepared_statements:
            raise PreparedStatementError(f"Prepared statement {statement_id} not found", statement_id=statement_id)

        del self._prepared_statements[statement_id]

    # =================================================================
    # ADVANCED SQL FEATURES
    # =================================================================

    def query_with_ctes(
        self,
        ctes: List[CTEDefinition],
        main_query: str,
        params: Optional[Dict[str, Any]] = None,
        as_dataframe: bool = True,
    ) -> Union[QueryResult, pd.DataFrame]:
        """
        Execute a query with Common Table Expressions (CTEs).

        Args:
            ctes: List of CTE definitions
            main_query: Main query that uses the CTEs
            params: Query parameters
            as_dataframe: Return result as pandas DataFrame

        Returns:
            QueryResult or pandas DataFrame

        Raises:
            SQLError: If query execution fails
        """
        if not ctes:
            return self.sql(main_query, params=params, as_dataframe=as_dataframe)

        # Build WITH clause
        cte_parts = []
        for cte in ctes:
            cte_def = cte.name
            if cte.columns:
                cte_def += f"({', '.join(cte.columns)})"
            cte_def += f" AS ({cte.query})"
            cte_parts.append(cte_def)

        # Check for recursive CTEs
        recursive = any(cte.recursive for cte in ctes)
        with_clause = "WITH RECURSIVE" if recursive else "WITH"

        full_query = f"{with_clause} {', '.join(cte_parts)} {main_query}"

        try:
            return self.sql(full_query, params=params, as_dataframe=as_dataframe)
        except Exception as e:
            raise SQLError(f"Failed to execute CTE query: {str(e)}")

    def query_with_window_functions(
        self,
        base_query: str,
        window_functions: List[WindowFunction],
        params: Optional[Dict[str, Any]] = None,
        as_dataframe: bool = True,
    ) -> Union[QueryResult, pd.DataFrame]:
        """
        Execute a query with window functions.

        Args:
            base_query: Base SELECT query
            window_functions: List of window function definitions
            params: Query parameters
            as_dataframe: Return result as pandas DataFrame

        Returns:
            QueryResult or pandas DataFrame

        Raises:
            SQLError: If query execution fails
        """
        if not window_functions:
            return self.sql(base_query, params=params, as_dataframe=as_dataframe)

        # Build window function clauses
        window_clauses = []
        for wf in window_functions:
            clause = f"{wf.function}"

            # Build OVER clause
            over_parts = []
            if wf.options.partition_by:
                over_parts.append(f"PARTITION BY {', '.join(wf.options.partition_by)}")

            if wf.options.order_by:
                order_items = []
                for order_item in wf.options.order_by:
                    order_items.append(f"{order_item['column']} {order_item.get('direction', 'ASC')}")
                over_parts.append(f"ORDER BY {', '.join(order_items)}")

            if wf.options.frame:
                frame_clause = f"{wf.options.frame.type} {wf.options.frame.start}"
                if wf.options.frame.end:
                    frame_clause += f" AND {wf.options.frame.end}"
                over_parts.append(frame_clause)

            over_clause = f"OVER ({' '.join(over_parts)})"
            window_clauses.append(f"{clause} {over_clause} AS {wf.alias}")

        # Modify the base query to include window functions
        # This is a simplified approach - in practice, you'd need more sophisticated SQL parsing
        if "SELECT" in base_query.upper():
            select_pos = base_query.upper().find("SELECT") + 6
            modified_query = (
                base_query[:select_pos] +
                " " + ", ".join(window_clauses) + "," +
                base_query[select_pos:]
            )
        else:
            raise SQLError("Base query must be a SELECT statement for window functions")

        try:
            return self.sql(modified_query, params=params, as_dataframe=as_dataframe)
        except Exception as e:
            raise SQLError(f"Failed to execute window function query: {str(e)}")

    def json_query(
        self,
        table_name: str,
        json_column: str,
        operation: str,
        path: str,
        value: Optional[Any] = None,
        where_clause: Optional[str] = None,
        as_dataframe: bool = True,
    ) -> Union[QueryResult, pd.DataFrame]:
        """
        Perform JSON operations on database columns.

        Args:
            table_name: Target table name
            json_column: JSON column name
            operation: JSON operation (extract, update, contains, etc.)
            path: JSON path expression
            value: Value for update operations
            where_clause: Optional WHERE clause
            as_dataframe: Return result as pandas DataFrame

        Returns:
            QueryResult or pandas DataFrame

        Raises:
            SQLError: If JSON operation fails
        """
        try:
            if operation == "extract":
                # Extract value from JSON
                sql = f"SELECT *, JSON_EXTRACT({json_column}, '$.{path}') AS extracted_value FROM {table_name}"

            elif operation == "update":
                # Update JSON field
                if value is None:
                    raise ValueError("Value required for JSON update operation")

                if isinstance(value, str):
                    value_expr = f"'{value}'"
                else:
                    value_expr = json.dumps(value)

                sql = f"UPDATE {table_name} SET {json_column} = JSON_SET({json_column}, '$.{path}', {value_expr})"

            elif operation == "contains":
                # Check if JSON contains a value
                if value is None:
                    raise ValueError("Value required for JSON contains operation")

                if isinstance(value, str):
                    value_expr = f"'{value}'"
                else:
                    value_expr = json.dumps(value)

                sql = f"SELECT * FROM {table_name} WHERE JSON_EXTRACT({json_column}, '$.{path}') = {value_expr}"

            elif operation == "array_length":
                # Get array length
                sql = f"SELECT *, JSON_ARRAY_LENGTH({json_column}, '$.{path}') AS array_length FROM {table_name}"

            elif operation == "object_keys":
                # Get object keys
                sql = f"SELECT *, JSON_KEYS({json_column}, '$.{path}') AS object_keys FROM {table_name}"

            else:
                raise ValueError(f"Unsupported JSON operation: {operation}")

            # Add WHERE clause if provided
            if where_clause:
                if "WHERE" in sql.upper():
                    sql += f" AND {where_clause}"
                else:
                    sql += f" WHERE {where_clause}"

            return self.sql(sql, as_dataframe=as_dataframe)

        except Exception as e:
            raise SQLError(f"Failed to execute JSON operation: {str(e)}")

    # =================================================================
    # VECTOR OPERATIONS
    # =================================================================

    def _normalize_vector(self, vector: Union[List[float], np.ndarray, Vector]) -> List[float]:
        """Normalize vector input to a list of floats."""
        if isinstance(vector, Vector):
            return vector.values
        elif isinstance(vector, np.ndarray):
            return vector.tolist()
        elif isinstance(vector, list):
            return vector
        else:
            raise VectorError(f"Unsupported vector type: {type(vector)}")

    def vector_add(
        self,
        vector1: Union[List[float], np.ndarray, Vector],
        vector2: Union[List[float], np.ndarray, Vector],
    ) -> VectorArithmeticResult:
        """
        Add two vectors element-wise.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            VectorArithmeticResult with the sum vector

        Raises:
            VectorError: If vectors have different dimensions
        """
        v1 = self._normalize_vector(vector1)
        v2 = self._normalize_vector(vector2)

        if len(v1) != len(v2):
            raise VectorError(
                f"Vector dimension mismatch: {len(v1)} vs {len(v2)}",
                vector_dimension_mismatch=True
            )

        start_time = datetime.now()

        try:
            # Use REST API endpoint if available
            payload = {"op": "add", "a": v1, "b": v2}

            response = self._client.post("/vector-algebra/operation", json=payload)
            data = self._handle_response(response)

            result_vec = data.get("result") or data.get("vector") or []
            result_vector = Vector(values=result_vec, dimensions=len(result_vec))

        except Exception:
            # Fallback to local computation
            result_values = [a + b for a, b in zip(v1, v2)]
            result_vector = Vector(values=result_values, dimensions=len(result_values))

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return VectorArithmeticResult(
            result=result_vector,
            took_ms=took_ms,
            operation="add",
        )

    def vector_subtract(
        self,
        vector1: Union[List[float], np.ndarray, Vector],
        vector2: Union[List[float], np.ndarray, Vector],
    ) -> VectorArithmeticResult:
        """
        Subtract vector2 from vector1 element-wise.

        Args:
            vector1: First vector (minuend)
            vector2: Second vector (subtrahend)

        Returns:
            VectorArithmeticResult with the difference vector

        Raises:
            VectorError: If vectors have different dimensions
        """
        v1 = self._normalize_vector(vector1)
        v2 = self._normalize_vector(vector2)

        if len(v1) != len(v2):
            raise VectorError(
                f"Vector dimension mismatch: {len(v1)} vs {len(v2)}",
                vector_dimension_mismatch=True
            )

        start_time = datetime.now()

        try:
            # Use REST API endpoint if available
            payload = {"op": "subtract", "a": v1, "b": v2}

            response = self._client.post("/vector-algebra/operation", json=payload)
            data = self._handle_response(response)

            result_vec = data.get("result") or data.get("vector") or []
            result_vector = Vector(values=result_vec, dimensions=len(result_vec))

        except Exception:
            # Fallback to local computation
            result_values = [a - b for a, b in zip(v1, v2)]
            result_vector = Vector(values=result_values, dimensions=len(result_values))

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return VectorArithmeticResult(
            result=result_vector,
            took_ms=took_ms,
            operation="subtract",
        )

    def vector_scalar_multiply(
        self,
        vector: Union[List[float], np.ndarray, Vector],
        scalar: float,
    ) -> VectorArithmeticResult:
        """
        Multiply a vector by a scalar value.

        Args:
            vector: Input vector
            scalar: Scalar multiplier

        Returns:
            VectorArithmeticResult with the scaled vector
        """
        v = self._normalize_vector(vector)
        start_time = datetime.now()

        try:
            # Use REST API endpoint if available
            payload = {"op": "scalar_multiply", "a": v, "scalar": scalar}

            response = self._client.post("/vector-algebra/operation", json=payload)
            data = self._handle_response(response)

            result_vec = data.get("result") or data.get("vector") or []
            result_vector = Vector(values=result_vec, dimensions=len(result_vec))

        except Exception:
            # Fallback to local computation
            result_values = [x * scalar for x in v]
            result_vector = Vector(values=result_values, dimensions=len(result_values))

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return VectorArithmeticResult(
            result=result_vector,
            took_ms=took_ms,
            operation="scalar_multiply",
        )

    def vector_dot_product(
        self,
        vector1: Union[List[float], np.ndarray, Vector],
        vector2: Union[List[float], np.ndarray, Vector],
    ) -> VectorSimilarityResult:
        """
        Calculate the dot product of two vectors.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            VectorSimilarityResult with the dot product

        Raises:
            VectorError: If vectors have different dimensions
        """
        v1 = self._normalize_vector(vector1)
        v2 = self._normalize_vector(vector2)

        if len(v1) != len(v2):
            raise VectorError(
                f"Vector dimension mismatch: {len(v1)} vs {len(v2)}",
                vector_dimension_mismatch=True
            )

        start_time = datetime.now()

        try:
            # Use REST API endpoint if available
            payload = {"op": "dot_product", "a": v1, "b": v2}

            response = self._client.post("/vector-algebra/operation", json=payload)
            data = self._handle_response(response)

            dot_product = data.get("scalar") or data.get("dot_product") or 0.0

        except Exception:
            # Fallback to local computation
            dot_product = sum(a * b for a, b in zip(v1, v2))

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return VectorSimilarityResult(
            similarity=dot_product,
            took_ms=took_ms,
            metric="dot_product",
        )

    def vector_magnitude(
        self,
        vector: Union[List[float], np.ndarray, Vector],
    ) -> VectorMagnitudeResult:
        """
        Calculate the magnitude (Euclidean norm) of a vector.

        Args:
            vector: Input vector

        Returns:
            VectorMagnitudeResult with the magnitude
        """
        v = self._normalize_vector(vector)
        start_time = datetime.now()

        try:
            # Use REST API endpoint if available
            payload = {"op": "magnitude", "a": v}

            response = self._client.post("/vector-algebra/operation", json=payload)
            data = self._handle_response(response)

            magnitude = data.get("scalar") or data.get("magnitude") or 0.0

        except Exception:
            # Fallback to local computation
            import math
            magnitude = math.sqrt(sum(x * x for x in v))

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return VectorMagnitudeResult(
            magnitude=magnitude,
            took_ms=took_ms,
        )

    def normalize_vector(
        self,
        vector: Union[List[float], np.ndarray, Vector],
    ) -> VectorArithmeticResult:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector

        Returns:
            VectorArithmeticResult with the normalized vector
        """
        v = self._normalize_vector(vector)
        start_time = datetime.now()

        try:
            # Use REST API endpoint if available
            payload = {"op": "normalize", "a": v}

            response = self._client.post("/vector-algebra/operation", json=payload)
            data = self._handle_response(response)

            result_vec = data.get("result") or data.get("vector") or []
            result_vector = Vector(values=result_vec, dimensions=len(result_vec))

        except Exception:
            # Fallback to local computation
            import math
            magnitude = math.sqrt(sum(x * x for x in v))
            if magnitude == 0:
                raise VectorError("Cannot normalize zero vector")

            result_values = [x / magnitude for x in v]
            result_vector = Vector(values=result_values, dimensions=len(result_values))

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return VectorArithmeticResult(
            result=result_vector,
            took_ms=took_ms,
            operation="normalize",
        )