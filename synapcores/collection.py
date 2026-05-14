"""
Collection class for SynapCores Python SDK.
"""

from typing import Optional, Dict, Any, List, Union, Callable, TYPE_CHECKING
import asyncio
import json
import numpy as np
from datetime import datetime

from .models import (
    Document,
    SearchResult,
    QueryOptions,
    VectorSearchParams,
    Schema,
    CollectionStats,
    SubscriptionEvent,
)
from .exceptions import ValidationError
from .subscription import Subscription

if TYPE_CHECKING:
    from .client import SynapCores


class Collection:
    """
    Represents a collection in SynapCores database.
    """
    
    def __init__(self, client: "SynapCores", name: str, schema: Optional[Dict[str, Any]] = None):
        self.client = client
        self.name = name
        self.schema = schema
        self._base_path = f"/collections/{name}"
    
    def insert(
        self,
        documents: Union[Dict[str, Any], List[Dict[str, Any]]],
        auto_embed: bool = True,
    ) -> Dict[str, Any]:
        """
        Insert documents into collection.
        
        Args:
            documents: Single document or list of documents
            auto_embed: Automatically generate embeddings for text fields
            
        Returns:
            Insert result with document IDs
        """
        is_single = isinstance(documents, dict)
        docs = [documents] if is_single else documents
        
        payload = {
            "documents": docs,
            "auto_embed": auto_embed,
        }
        
        response = self.client._client.post(
            f"{self._base_path}/documents",
            json=payload,
        )
        result = self.client._handle_response(response)
        
        return result
    
    def get(self, document_id: str) -> Optional[Document]:
        """
        Get document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document or None if not found
        """
        try:
            response = self.client._client.get(
                f"{self._base_path}/documents/{document_id}"
            )
            data = self.client._handle_response(response)
            return Document(**data)
        except NotFoundError:
            return None
    
    def update(
        self,
        document_id: str,
        data: Dict[str, Any],
        merge: bool = True,
    ) -> Document:
        """
        Update document.
        
        Args:
            document_id: Document ID
            data: Update data
            merge: Merge with existing data (True) or replace (False)
            
        Returns:
            Updated document
        """
        payload = {
            "data": data,
            "merge": merge,
        }
        
        response = self.client._client.patch(
            f"{self._base_path}/documents/{document_id}",
            json=payload,
        )
        result = self.client._handle_response(response)
        return Document(**result)
    
    def delete(self, document_id: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Delete documents.
        
        Args:
            document_id: Single ID or list of IDs
            
        Returns:
            Delete result
        """
        ids = [document_id] if isinstance(document_id, str) else document_id
        
        response = self.client._client.request(
            "DELETE",
            f"{self._base_path}/documents",
            json={"ids": ids},
        )
        return self.client._handle_response(response)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SearchResult:
        """
        Semantic search in collection.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter: Optional filter conditions
            **kwargs: Additional search options
            
        Returns:
            Search results
        """
        payload = {
            "query": query,
            "top_k": top_k,
            "filter": filter,
            **kwargs,
        }
        
        response = self.client._client.post(
            f"{self._base_path}/search",
            json=payload,
        )
        data = self.client._handle_response(response)
        
        return SearchResult(
            documents=[Document(**doc) for doc in data["documents"]],
            total=data["total"],
            took_ms=data["took_ms"],
            next_offset=data.get("next_offset"),
        )
    
    def vector_search(
        self,
        vector: Union[List[float], np.ndarray],
        field: str = "embedding",
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine",
    ) -> SearchResult:
        """
        Vector similarity search.
        
        Args:
            vector: Query vector
            field: Vector field name
            top_k: Number of results
            filter: Optional filter
            distance_metric: Distance metric (cosine, euclidean, dot_product)
            
        Returns:
            Search results
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        params = VectorSearchParams(
            vector=vector,
            field=field,
            top_k=top_k,
            filter=filter,
            distance_metric=distance_metric,
        )
        
        response = self.client._client.post(
            f"{self._base_path}/vector_search",
            json=params.model_dump(),
        )
        data = self.client._handle_response(response)
        
        return SearchResult(
            documents=[Document(**doc) for doc in data["documents"]],
            total=data["total"],
            took_ms=data["took_ms"],
        )
    
    def query(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        sort: Optional[List[Dict[str, str]]] = None,
        projection: Optional[List[str]] = None,
    ) -> SearchResult:
        """
        Query documents with filters.
        
        Args:
            filter: Filter conditions
            limit: Maximum results
            offset: Skip results
            sort: Sort order
            projection: Fields to return
            
        Returns:
            Query results
        """
        options = QueryOptions(
            limit=limit,
            offset=offset,
            sort=sort,
            projection=projection,
        )
        
        payload = {
            "filter": filter or {},
            **options.model_dump(exclude_none=True),
        }
        
        response = self.client._client.post(
            f"{self._base_path}/query",
            json=payload,
        )
        data = self.client._handle_response(response)
        
        return SearchResult(
            documents=[Document(**doc) for doc in data["documents"]],
            total=data["total"],
            took_ms=data["took_ms"],
            next_offset=data.get("next_offset"),
        )
    
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in collection.
        
        Args:
            filter: Optional filter conditions
            
        Returns:
            Document count
        """
        payload = {"filter": filter or {}}
        
        response = self.client._client.post(
            f"{self._base_path}/count",
            json=payload,
        )
        data = self.client._handle_response(response)
        return data["count"]
    
    def stats(self) -> CollectionStats:
        """
        Get collection statistics.
        
        Returns:
            Collection statistics
        """
        response = self.client._client.get(f"{self._base_path}/stats")
        data = self.client._handle_response(response)
        return CollectionStats(**data)
    
    async def subscribe(
        self,
        filter: Optional[Dict[str, Any]] = None,
        on_change: Optional[Callable[[SubscriptionEvent], None]] = None,
    ) -> Subscription:
        """
        Subscribe to real-time changes.
        
        Args:
            filter: Filter for events
            on_change: Callback for changes
            
        Returns:
            Subscription instance
        """
        sub = Subscription(self, filter, on_change)
        await sub.connect()
        return sub
    
    def create_index(
        self,
        field: str,
        index_type: str = "btree",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create an index on a field.
        
        Args:
            field: Field name
            index_type: Index type (btree, hash, vector)
            options: Index options
            
        Returns:
            Index creation result
        """
        payload = {
            "field": field,
            "type": index_type,
            "options": options or {},
        }
        
        response = self.client._client.post(
            f"{self._base_path}/indexes",
            json=payload,
        )
        return self.client._handle_response(response)
    
    def drop_index(self, field: str) -> None:
        """
        Drop an index.
        
        Args:
            field: Field name
        """
        response = self.client._client.delete(
            f"{self._base_path}/indexes/{field}"
        )
        self.client._handle_response(response)
    
    def __repr__(self) -> str:
        return f"Collection(name='{self.name}')"