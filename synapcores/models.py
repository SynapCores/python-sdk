"""
Data models for SynapCores Python SDK.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal, Callable
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class FieldType(str, Enum):
    """Supported field types in SynapCores."""
    STRING = "string"
    INTEGER = "integer"
    BIGINT = "bigint"
    SMALLINT = "smallint"
    FLOAT = "float"
    DOUBLE = "double"
    DECIMAL = "decimal"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    TIME = "time"
    JSON = "json"
    JSONB = "jsonb"
    VECTOR = "vector"
    TEXT = "text"
    VARCHAR = "varchar"
    CHAR = "char"
    BINARY = "binary"
    BLOB = "blob"
    UUID = "uuid"


class SchemaField(BaseModel):
    """Schema field definition."""
    name: str
    type: Union[FieldType, str]
    dimensions: Optional[int] = None
    indexed: bool = False
    required: bool = False
    unique: bool = False
    default: Optional[Any] = None
    description: Optional[str] = None


class Schema(BaseModel):
    """Collection schema definition."""
    fields: List[SchemaField]
    primary_key: Optional[str] = None
    vector_fields: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Document(BaseModel):
    """Document model."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: Optional[str] = None
    data: Dict[str, Any]
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryOptions(BaseModel):
    """Options for query operations."""
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)
    sort: Optional[List[Dict[str, str]]] = None
    projection: Optional[List[str]] = None
    include_metadata: bool = False


class VectorSearchParams(BaseModel):
    """Parameters for vector search."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    vector: Union[List[float], np.ndarray]
    field: str = "embedding"
    top_k: int = Field(default=10, ge=1, le=1000)
    filter: Optional[Dict[str, Any]] = None
    include_metadata: bool = False
    distance_metric: str = "cosine"


class SearchResult(BaseModel):
    """Result from search operations."""
    documents: List[Document]
    total: int
    took_ms: float
    next_offset: Optional[int] = None


class QueryResult(BaseModel):
    """Result from SQL query operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    rows: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    took_ms: float
    query_plan: Optional[Dict[str, Any]] = None


class SubscriptionEvent(BaseModel):
    """Real-time subscription event."""
    operation: str
    collection: str
    document: Document
    timestamp: datetime
    sequence: int


class CollectionStats(BaseModel):
    """Collection statistics."""
    name: str
    document_count: int
    size_bytes: int
    index_count: int
    created_at: datetime
    updated_at: datetime


class ModelInfo(BaseModel):
    """AutoML model information."""
    id: str
    name: str
    task: str
    status: str
    accuracy: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    config: Dict[str, Any] = {}


class NLPAnalysis(BaseModel):
    """NLP analysis results."""

    class Sentiment(BaseModel):
        label: str
        score: float
        confidence: float

    class Entity(BaseModel):
        text: str
        type: str
        start: int
        end: int
        score: float

    sentiment: Optional[Sentiment] = None
    entities: Optional[List[Entity]] = None
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    language: Optional[str] = None


# SQL Operation Models
class ConstraintType(str, Enum):
    """Types of database constraints."""
    PRIMARY_KEY = "PRIMARY_KEY"
    UNIQUE = "UNIQUE"
    NOT_NULL = "NOT_NULL"
    CHECK = "CHECK"
    FOREIGN_KEY = "FOREIGN_KEY"
    DEFAULT = "DEFAULT"


class ColumnConstraint(BaseModel):
    """Database column constraint definition."""
    type: ConstraintType
    expression: Optional[str] = None
    reference_table: Optional[str] = None
    reference_column: Optional[str] = None
    on_delete: Optional[Literal["CASCADE", "SET_NULL", "RESTRICT"]] = None
    on_update: Optional[Literal["CASCADE", "SET_NULL", "RESTRICT"]] = None


class ColumnDefinition(BaseModel):
    """Database column definition."""
    name: str
    data_type: str
    constraints: List[ColumnConstraint] = Field(default_factory=list)
    default_value: Optional[Any] = None
    comment: Optional[str] = None


class TableConstraint(BaseModel):
    """Database table-level constraint."""
    type: ConstraintType
    name: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    expression: Optional[str] = None
    reference_table: Optional[str] = None
    reference_columns: List[str] = Field(default_factory=list)


class CreateTableOptions(BaseModel):
    """Options for CREATE TABLE operations."""
    if_not_exists: bool = False
    temporary: bool = False
    constraints: List[TableConstraint] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)


class AlterTableAction(str, Enum):
    """Types of ALTER TABLE actions."""
    ADD_COLUMN = "ADD_COLUMN"
    DROP_COLUMN = "DROP_COLUMN"
    RENAME_COLUMN = "RENAME_COLUMN"
    MODIFY_COLUMN = "MODIFY_COLUMN"
    ADD_CONSTRAINT = "ADD_CONSTRAINT"
    DROP_CONSTRAINT = "DROP_CONSTRAINT"
    RENAME_TABLE = "RENAME_TABLE"


class AlterTableOptions(BaseModel):
    """Options for ALTER TABLE operations."""
    action: AlterTableAction
    column_definition: Optional[ColumnDefinition] = None
    old_column_name: Optional[str] = None
    new_column_name: Optional[str] = None
    constraint: Optional[TableConstraint] = None
    constraint_name: Optional[str] = None
    new_table_name: Optional[str] = None


class IndexColumn(BaseModel):
    """Index column specification."""
    name: str
    order: Literal["ASC", "DESC"] = "ASC"
    length: Optional[int] = None


class IndexDefinition(BaseModel):
    """Database index definition."""
    name: str
    table_name: str
    columns: List[IndexColumn]
    unique: bool = False
    if_not_exists: bool = False
    index_type: Optional[str] = None
    where_clause: Optional[str] = None


class TableInfo(BaseModel):
    """Table information and metadata."""
    name: str
    columns: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class IndexInfo(BaseModel):
    """Index information."""
    name: str
    table: str
    columns: List[str]
    unique: bool
    index_type: str
    size_bytes: Optional[int] = None


# Transaction Models
class IsolationLevel(str, Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "READ_UNCOMMITTED"
    READ_COMMITTED = "READ_COMMITTED"
    REPEATABLE_READ = "REPEATABLE_READ"
    SERIALIZABLE = "SERIALIZABLE"


class TransactionOptions(BaseModel):
    """Transaction configuration options."""
    isolation_level: Optional[IsolationLevel] = None
    read_only: bool = False
    timeout: Optional[int] = None  # in milliseconds
    name: Optional[str] = None


class TransactionContext(BaseModel):
    """Active transaction context."""
    id: str
    isolation_level: IsolationLevel
    read_only: bool
    started_at: datetime
    timeout: Optional[int] = None
    name: Optional[str] = None


class SavepointInfo(BaseModel):
    """Savepoint information."""
    name: str
    created_at: datetime
    transaction_id: str


# Batch Operation Models
class ConflictResolution(str, Enum):
    """Conflict resolution strategies for batch operations."""
    REPLACE = "REPLACE"
    IGNORE = "IGNORE"
    UPDATE = "UPDATE"
    ABORT = "ABORT"


class BatchInsertOptions(BaseModel):
    """Options for batch insert operations."""
    table_name: str
    columns: List[str]
    rows: List[List[Any]]
    on_conflict: ConflictResolution = ConflictResolution.ABORT
    batch_size: int = 1000
    return_errors: bool = True


class BatchUpdateItem(BaseModel):
    """Single update item for batch operations."""
    set: Dict[str, Any]
    where: Dict[str, Any]


class BatchUpdateOptions(BaseModel):
    """Options for batch update operations."""
    table_name: str
    updates: List[BatchUpdateItem]
    batch_size: int = 500
    return_errors: bool = True


class BatchDeleteOptions(BaseModel):
    """Options for batch delete operations."""
    table_name: str
    conditions: List[Dict[str, Any]]
    batch_size: int = 1000
    return_errors: bool = True


class BatchError(BaseModel):
    """Error information for failed batch items."""
    item_index: int
    error_message: str
    error_code: Optional[str] = None
    item_data: Optional[Any] = None


class BatchResult(BaseModel):
    """Result of batch operations."""
    total_processed: int
    successful: int
    failed: int
    errors: List[BatchError] = Field(default_factory=list)
    took_ms: float


# Prepared Statement Models
class PreparedStatementOptions(BaseModel):
    """Options for prepared statements."""
    name: Optional[str] = None
    parameter_types: List[str] = Field(default_factory=list)
    cache_plan: bool = True


class PreparedStatement(BaseModel):
    """Prepared statement information."""
    id: str
    name: Optional[str] = None
    query: str
    parameter_count: int
    parameter_types: List[str]
    created_at: datetime


# Advanced SQL Feature Models
class CTEDefinition(BaseModel):
    """Common Table Expression definition."""
    name: str
    query: str
    recursive: bool = False
    columns: Optional[List[str]] = None


class WindowFrameType(str, Enum):
    """Window frame types."""
    ROWS = "ROWS"
    RANGE = "RANGE"
    GROUPS = "GROUPS"


class WindowFrame(BaseModel):
    """Window function frame specification."""
    type: WindowFrameType
    start: str = "UNBOUNDED PRECEDING"
    end: Optional[str] = None


class WindowFunctionOptions(BaseModel):
    """Window function configuration."""
    partition_by: List[str] = Field(default_factory=list)
    order_by: List[Dict[str, str]] = Field(default_factory=list)
    frame: Optional[WindowFrame] = None


class WindowFunction(BaseModel):
    """Window function definition."""
    alias: str
    function: str
    options: WindowFunctionOptions


class JSONOperation(str, Enum):
    """JSON operation types."""
    EXTRACT = "extract"
    UPDATE = "update"
    CONTAINS = "contains"
    ARRAY_LENGTH = "array_length"
    OBJECT_KEYS = "object_keys"


# Vector Operation Models
class Vector(BaseModel):
    """Vector data structure."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: Union[List[float], np.ndarray]
    dimensions: int
    dtype: str = "float32"
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, **data):
        if 'values' in data:
            values = data['values']
            if isinstance(values, np.ndarray):
                data['values'] = values.tolist()
                data['dimensions'] = len(values)
            else:
                data['dimensions'] = len(values)
        super().__init__(**data)


class VectorArithmeticResult(BaseModel):
    """Result of vector arithmetic operations."""
    result: Vector
    took_ms: float
    operation: str


class VectorSimilarityResult(BaseModel):
    """Result of vector similarity operations."""
    similarity: float
    distance: Optional[float] = None
    took_ms: float
    metric: str


class VectorMagnitudeResult(BaseModel):
    """Result of vector magnitude calculation."""
    magnitude: float
    took_ms: float


class VectorSearchItem(BaseModel):
    """Single item from vector search results."""
    id: Optional[str] = None
    vector: Optional[Vector] = None
    metadata: Optional[Dict[str, Any]] = None
    similarity: float
    distance: Optional[float] = None


class VectorSearchResult(BaseModel):
    """Result of vector search operations."""
    items: List[VectorSearchItem]
    total: int
    took_ms: float
    query_vector: Optional[Vector] = None
    metric: str


class KNNSearchOptions(BaseModel):
    """Options for K-nearest neighbors search."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_vector: Union[List[float], np.ndarray, Vector]
    k: int = Field(ge=1, le=1000)
    table_name: str
    vector_column: str = "embedding"
    metadata_columns: List[str] = Field(default_factory=list)
    filter: Optional[Dict[str, Any]] = None
    metric: Literal["cosine", "euclidean", "dot"] = "cosine"


class RangeSearchOptions(BaseModel):
    """Options for range-based similarity search."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_vector: Union[List[float], np.ndarray, Vector]
    threshold: float = Field(ge=0.0, le=1.0)
    table_name: str
    vector_column: str = "embedding"
    metadata_columns: List[str] = Field(default_factory=list)
    filter: Optional[Dict[str, Any]] = None
    metric: Literal["cosine", "euclidean", "dot"] = "cosine"
    max_results: int = Field(default=100, ge=1, le=10000)


class HybridSearchOptions(BaseModel):
    """Options for hybrid vector + text search."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: Union[List[float], np.ndarray, Vector]
    text_query: Optional[str] = None
    sql_filter: Optional[str] = None
    k: int = Field(default=10, ge=1, le=1000)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    metric: Literal["cosine", "euclidean", "dot"] = "cosine"
    weights: Dict[str, float] = Field(default_factory=lambda: {"vector": 0.8, "text": 0.2})
    table_name: Optional[str] = None
    vector_column: str = "embedding"
    text_columns: List[str] = Field(default_factory=list)