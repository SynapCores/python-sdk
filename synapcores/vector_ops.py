"""
Vector operations extension for SynapCores Python SDK.
This module contains vector similarity functions and search operations.
"""

from typing import Union, List, Optional, Dict, Any
from datetime import datetime
import json
import numpy as np

from .models import (
    Vector,
    VectorSimilarityResult,
    VectorSearchResult,
    VectorSearchItem,
    KNNSearchOptions,
    RangeSearchOptions,
    HybridSearchOptions,
    QueryResult,
)
from .exceptions import VectorError, SQLError


class VectorOperationsMixin:
    """Mixin class providing vector similarity and search operations."""

    def cosine_similarity(
        self,
        vector1: Union[List[float], np.ndarray, Vector],
        vector2: Union[List[float], np.ndarray, Vector],
    ) -> VectorSimilarityResult:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            VectorSimilarityResult with cosine similarity score

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
            payload = {"op": "cosine_similarity", "a": v1, "b": v2}

            response = self._client.post("/vector-algebra/operation", json=payload)
            data = self._handle_response(response)

            similarity = data.get("scalar") or data.get("similarity") or 0.0

        except Exception:
            # Fallback to local computation
            import math

            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(v1, v2))

            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in v1))
            magnitude2 = math.sqrt(sum(b * b for b in v2))

            # Handle zero vectors
            if magnitude1 == 0 or magnitude2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (magnitude1 * magnitude2)

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return VectorSimilarityResult(
            similarity=similarity,
            took_ms=took_ms,
            metric="cosine",
        )

    def l2_distance(
        self,
        vector1: Union[List[float], np.ndarray, Vector],
        vector2: Union[List[float], np.ndarray, Vector],
    ) -> VectorSimilarityResult:
        """
        Calculate L2 (Euclidean) distance between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            VectorSimilarityResult with L2 distance

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
            payload = {"op": "l2_distance", "a": v1, "b": v2}

            response = self._client.post("/vector-algebra/operation", json=payload)
            data = self._handle_response(response)

            distance = data.get("scalar") or data.get("distance") or 0.0
            # Convert distance to similarity (closer = more similar)
            similarity = 1.0 / (1.0 + distance)

        except Exception:
            # Fallback to local computation
            import math

            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
            # Convert distance to similarity
            similarity = 1.0 / (1.0 + distance)

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return VectorSimilarityResult(
            similarity=similarity,
            distance=distance,
            took_ms=took_ms,
            metric="euclidean",
        )

    def inner_product(
        self,
        vector1: Union[List[float], np.ndarray, Vector],
        vector2: Union[List[float], np.ndarray, Vector],
    ) -> VectorSimilarityResult:
        """
        Calculate inner product (dot product) between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            VectorSimilarityResult with inner product

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
            payload = {"op": "inner_product", "a": v1, "b": v2}

            response = self._client.post("/vector-algebra/operation", json=payload)
            data = self._handle_response(response)

            similarity = data.get("scalar") or data.get("inner_product") or 0.0

        except Exception:
            # Fallback to local computation
            similarity = sum(a * b for a, b in zip(v1, v2))

        end_time = datetime.now()
        took_ms = (end_time - start_time).total_seconds() * 1000

        return VectorSimilarityResult(
            similarity=similarity,
            took_ms=took_ms,
            metric="inner_product",
        )

    def knn_search(self, options: KNNSearchOptions) -> List[VectorSearchItem]:
        """
        Perform K-nearest neighbors vector search.

        Args:
            options: KNN search configuration

        Returns:
            List of VectorSearchItem results

        Raises:
            VectorError: If search fails
        """
        query_vector = self._normalize_vector(options.query_vector)
        start_time = datetime.now()

        try:
            # v0.2.0: gateway exposes /vectors/collections/:c/search with
            # a `mode` discriminator. ``options.table_name`` is treated as
            # the collection name.
            collection = getattr(options, "collection_name", None) or options.table_name
            payload = {
                "mode": "knn",
                "vector": query_vector,
                "limit": options.k,
                "filter": options.filter,
                "metric": options.metric,
            }

            response = self._client.post(
                f"/vectors/collections/{collection}/search", json=payload
            )
            data = self._handle_response(response)

            results = []
            for item in data.get("matches") or data.get("results") or []:
                results.append(VectorSearchItem(
                    id=item.get("id"),
                    vector=Vector(values=item["vector"]) if "vector" in item else None,
                    metadata=item.get("metadata"),
                    similarity=item["similarity"],
                    distance=item.get("distance"),
                ))

            return results

        except Exception as e:
            # Fallback to SQL-based search
            try:
                return self._sql_vector_search(
                    query_vector=query_vector,
                    table_name=options.table_name,
                    vector_column=options.vector_column,
                    metadata_columns=options.metadata_columns,
                    k=options.k,
                    metric=options.metric,
                    filter_condition=options.filter,
                )
            except Exception as sql_error:
                raise VectorError(f"KNN search failed: {str(e)} (SQL fallback: {str(sql_error)})")

    def range_search(self, options: RangeSearchOptions) -> List[VectorSearchItem]:
        """
        Perform range-based similarity search.

        Args:
            options: Range search configuration

        Returns:
            List of VectorSearchItem results

        Raises:
            VectorError: If search fails
        """
        query_vector = self._normalize_vector(options.query_vector)
        start_time = datetime.now()

        try:
            # v0.2.0: routes through /vectors/collections/:c/search with
            # mode="range".
            collection = getattr(options, "collection_name", None) or options.table_name
            payload = {
                "mode": "range",
                "vector": query_vector,
                "threshold": options.threshold,
                "max_results": options.max_results,
                "filter": options.filter,
                "metric": options.metric,
            }

            response = self._client.post(
                f"/vectors/collections/{collection}/search", json=payload
            )
            data = self._handle_response(response)

            results = []
            for item in data.get("matches") or data.get("results") or []:
                results.append(VectorSearchItem(
                    id=item.get("id"),
                    vector=Vector(values=item["vector"]) if "vector" in item else None,
                    metadata=item.get("metadata"),
                    similarity=item["similarity"],
                    distance=item.get("distance"),
                ))

            return results

        except Exception as e:
            # Fallback to SQL-based search
            try:
                return self._sql_vector_search(
                    query_vector=query_vector,
                    table_name=options.table_name,
                    vector_column=options.vector_column,
                    metadata_columns=options.metadata_columns,
                    threshold=options.threshold,
                    metric=options.metric,
                    filter_condition=options.filter,
                    max_results=options.max_results,
                )
            except Exception as sql_error:
                raise VectorError(f"Range search failed: {str(e)} (SQL fallback: {str(sql_error)})")

    def hybrid_search(self, options: HybridSearchOptions) -> List[VectorSearchItem]:
        """
        Perform hybrid vector + text search.

        Args:
            options: Hybrid search configuration

        Returns:
            List of VectorSearchItem results

        Raises:
            VectorError: If search fails
        """
        query_vector = self._normalize_vector(options.vector)
        start_time = datetime.now()

        try:
            # v0.2.0: routes through /vectors/collections/:c/search with
            # mode="hybrid".
            collection = getattr(options, "collection_name", None) or options.table_name
            payload = {
                "mode": "hybrid",
                "vector": query_vector,
                "text_query": options.text_query,
                "sql_filter": options.sql_filter,
                "limit": options.k,
                "threshold": options.threshold,
                "metric": options.metric,
                "weights": options.weights,
            }

            response = self._client.post(
                f"/vectors/collections/{collection}/search", json=payload
            )
            data = self._handle_response(response)

            results = []
            for item in data.get("matches") or data.get("results") or []:
                results.append(VectorSearchItem(
                    id=item.get("id"),
                    vector=Vector(values=item["vector"]) if "vector" in item else None,
                    metadata=item.get("metadata"),
                    similarity=item["similarity"],
                    distance=item.get("distance"),
                ))

            return results

        except Exception as e:
            raise VectorError(f"Hybrid search failed: {str(e)}")

    def _sql_vector_search(
        self,
        query_vector: List[float],
        table_name: str,
        vector_column: str,
        metadata_columns: List[str],
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        metric: str = "cosine",
        filter_condition: Optional[Dict[str, Any]] = None,
        max_results: Optional[int] = None,
    ) -> List[VectorSearchItem]:
        """
        Fallback SQL-based vector search using similarity functions.

        Args:
            query_vector: Query vector
            table_name: Target table
            vector_column: Vector column name
            metadata_columns: Columns to return as metadata
            k: Number of results for KNN
            threshold: Similarity threshold for range search
            metric: Distance metric
            filter_condition: Additional filter conditions
            max_results: Maximum number of results

        Returns:
            List of VectorSearchItem results
        """
        # Build SQL query
        select_columns = ["rowid as id"]
        if metadata_columns:
            select_columns.extend(metadata_columns)

        # Choose similarity function based on metric
        if metric == "cosine":
            similarity_func = "COSINE_SIMILARITY"
        elif metric == "euclidean":
            similarity_func = "L2_DISTANCE"
        elif metric == "dot":
            similarity_func = "INNER_PRODUCT"
        else:
            similarity_func = "COSINE_SIMILARITY"  # Default

        select_columns.append(f"{similarity_func}({vector_column}, $1) as similarity")

        sql = f"SELECT {', '.join(select_columns)} FROM {table_name}"

        # Add filter conditions
        where_conditions = []

        if threshold is not None:
            where_conditions.append(f"{similarity_func}({vector_column}, $1) >= {threshold}")

        if filter_condition:
            for key, value in filter_condition.items():
                if isinstance(value, dict) and "$ne" in value:
                    where_conditions.append(f"{key} != '{value['$ne']}'")
                elif isinstance(value, str):
                    where_conditions.append(f"{key} = '{value}'")
                else:
                    where_conditions.append(f"{key} = {value}")

        if where_conditions:
            sql += f" WHERE {' AND '.join(where_conditions)}"

        # Add ordering and limit
        sql += " ORDER BY similarity DESC"

        if k is not None:
            sql += f" LIMIT {k}"
        elif max_results is not None:
            sql += f" LIMIT {max_results}"

        try:
            result = self.sql(sql, params={"1": query_vector}, as_dataframe=False)

            search_results = []
            for row in result.rows:
                metadata = {}
                for col in metadata_columns:
                    if col in row:
                        metadata[col] = row[col]

                search_results.append(VectorSearchItem(
                    id=str(row.get("id")),
                    metadata=metadata,
                    similarity=row.get("similarity", 0.0),
                ))

            return search_results

        except Exception as e:
            raise SQLError(f"SQL vector search failed: {str(e)}")

    def vector_sql_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        as_dataframe: bool = True,
    ) -> Union[QueryResult, Any]:
        """
        Execute SQL query with vector operations and functions.

        This method supports advanced vector SQL operations including:
        - Vector operators: <->, <=>, <#>
        - Vector functions: VECTOR_ADD, VECTOR_SUBTRACT, COSINE_SIMILARITY, etc.
        - Hybrid vector/text queries

        Args:
            query: SQL query with vector operations
            params: Query parameters (vectors should be passed as lists)
            as_dataframe: Return result as pandas DataFrame

        Returns:
            QueryResult or pandas DataFrame

        Raises:
            SQLError: If query execution fails
        """
        try:
            # Process parameters to handle vector inputs
            processed_params = {}
            if params:
                for key, value in params.items():
                    if isinstance(value, (Vector, np.ndarray)):
                        processed_params[key] = self._normalize_vector(value)
                    else:
                        processed_params[key] = value

            return self.sql(query, params=processed_params, as_dataframe=as_dataframe)

        except Exception as e:
            raise SQLError(f"Vector SQL query failed: {str(e)}")