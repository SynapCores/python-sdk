"""
Exception classes for SynapCores Python SDK.
"""

from typing import Optional, Dict, Any, List


class SynapCoresError(Exception):
    """Base exception for all SynapCores errors."""

    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ConnectionError(SynapCoresError):
    """Raised when connection to SynapCores server fails."""
    pass


class AuthenticationError(SynapCoresError):
    """Raised when authentication fails."""
    pass


class ValidationError(SynapCoresError):
    """Raised when request validation fails."""
    pass


class NotFoundError(SynapCoresError):
    """Raised when requested resource is not found."""
    pass


class ServerError(SynapCoresError):
    """Raised when server returns an error."""
    pass


class TimeoutError(SynapCoresError):
    """Raised when request times out."""
    pass


class RateLimitError(SynapCoresError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class SQLError(SynapCoresError):
    """Raised when SQL operations fail."""

    def __init__(self, message: str, sql_state: Optional[str] = None, error_position: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.sql_state = sql_state
        self.error_position = error_position


class VectorError(SynapCoresError):
    """Raised when vector operations fail."""

    def __init__(self, message: str, vector_dimension_mismatch: bool = False, **kwargs):
        super().__init__(message, **kwargs)
        self.vector_dimension_mismatch = vector_dimension_mismatch


class TransactionError(SynapCoresError):
    """Raised when transaction operations fail."""

    def __init__(self, message: str, transaction_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.transaction_id = transaction_id


class BatchOperationError(SynapCoresError):
    """Raised when batch operations encounter errors."""

    def __init__(self, message: str, failed_items: Optional[List[Dict[str, Any]]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.failed_items = failed_items or []


class IndexError(SynapCoresError):
    """Raised when index operations fail."""
    pass


class ConstraintError(SynapCoresError):
    """Raised when database constraints are violated."""

    def __init__(self, message: str, constraint_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.constraint_name = constraint_name


class PreparedStatementError(SynapCoresError):
    """Raised when prepared statement operations fail."""

    def __init__(self, message: str, statement_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.statement_id = statement_id