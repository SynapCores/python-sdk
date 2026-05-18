"""
SynapCores Python SDK

Official Python SDK for SynapCores AI-Native Database Management System.
"""

from .client import SynapCores
from .collection import Collection
from .vector_collection import VectorCollection
from .exceptions import (
    SynapCoresError,
    ConnectionError,
    AuthenticationError,
    ValidationError,
    NotFoundError,
    ServerError,
    TimeoutError,
    RateLimitError,
    SQLError,
    VectorError,
    TransactionError,
    BatchOperationError,
    IndexError,
    ConstraintError,
    PreparedStatementError,
)
from .models import (
    Document,
    QueryResult,
    SearchResult,
    Schema,
    FieldType,
    VectorSearchParams,
    SubscriptionEvent,
    # SQL Operation Models
    ColumnDefinition,
    CreateTableOptions,
    AlterTableOptions,
    IndexDefinition,
    TableInfo,
    TransactionOptions,
    TransactionContext,
    BatchInsertOptions,
    BatchUpdateOptions,
    BatchDeleteOptions,
    BatchResult,
    PreparedStatement,
    CTEDefinition,
    WindowFunction,
    # Vector Operation Models
    Vector,
    VectorArithmeticResult,
    VectorSimilarityResult,
    VectorMagnitudeResult,
    VectorSearchResult,
    KNNSearchOptions,
    RangeSearchOptions,
    HybridSearchOptions,
)

# v0.2.0 modules
from .graph import GraphClient
from .nl2sql import NL2SqlClient
from .filesystem import FilesystemCollectionsClient
from .chat import ChatClient
from .multimodal import MultimodalClient
from .system import SystemClient
from .transactions import TransactionsClient, Tx
from .mcp import McpClient
from .recipes import RecipeClient
from .schema import SchemaClient

__version__ = "0.3.0"
__all__ = [
    # Core Classes
    "SynapCores",
    "Collection",
    "VectorCollection",

    # v0.2.0 sub-clients
    "GraphClient",
    "NL2SqlClient",
    "FilesystemCollectionsClient",
    "ChatClient",
    "MultimodalClient",
    "SystemClient",
    "TransactionsClient",
    "Tx",
    "McpClient",
    "RecipeClient",
    "SchemaClient",

    # Models
    "Document",
    "QueryResult",
    "SearchResult",
    "Schema",
    "FieldType",
    "VectorSearchParams",
    "SubscriptionEvent",

    # SQL Operation Models
    "ColumnDefinition",
    "CreateTableOptions",
    "AlterTableOptions",
    "IndexDefinition",
    "TableInfo",
    "TransactionOptions",
    "TransactionContext",
    "BatchInsertOptions",
    "BatchUpdateOptions",
    "BatchDeleteOptions",
    "BatchResult",
    "PreparedStatement",
    "CTEDefinition",
    "WindowFunction",

    # Vector Operation Models
    "Vector",
    "VectorArithmeticResult",
    "VectorSimilarityResult",
    "VectorMagnitudeResult",
    "VectorSearchResult",
    "KNNSearchOptions",
    "RangeSearchOptions",
    "HybridSearchOptions",

    # Exceptions
    "SynapCoresError",
    "ConnectionError",
    "AuthenticationError",
    "ValidationError",
    "NotFoundError",
    "ServerError",
    "TimeoutError",
    "RateLimitError",
    "SQLError",
    "VectorError",
    "TransactionError",
    "BatchOperationError",
    "IndexError",
    "ConstraintError",
    "PreparedStatementError",
]
