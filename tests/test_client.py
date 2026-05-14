"""
Tests for SynapCores client.
"""

import pytest
from unittest.mock import Mock, patch
import httpx

from synapcores import SynapCores, Collection
from synapcores.exceptions import AuthenticationError, NotFoundError


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client."""
    with patch("synapcores.client.httpx.Client") as mock:
        yield mock


def test_client_initialization():
    """Test client initialization."""
    client = SynapCores(
        host="test.host",
        port=9090,
        api_key="ak_test-key",
        use_https=True,
    )

    assert client.host == "test.host"
    assert client.port == 9090
    assert client.api_key == "ak_test-key"
    assert client.use_https is True
    assert client.base_url == "https://test.host:9090/v1"


def test_create_collection(mock_httpx_client):
    """Test collection creation."""
    # Setup mock
    mock_response = Mock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "name": "test_collection",
        "schema": {"field1": "string"},
    }
    
    mock_client_instance = Mock()
    mock_client_instance.post.return_value = mock_response
    mock_httpx_client.return_value = mock_client_instance
    
    # Test
    client = SynapCores()
    collection = client.create_collection(
        name="test_collection",
        schema={"field1": "string"}
    )
    
    assert isinstance(collection, Collection)
    assert collection.name == "test_collection"
    mock_client_instance.post.assert_called_once_with(
        "/collections",
        json={"name": "test_collection", "schema": {"field1": "string"}}
    )


def test_authentication_error(mock_httpx_client):
    """Test authentication error handling."""
    # Setup mock
    mock_response = Mock()
    mock_response.status_code = 401
    
    mock_client_instance = Mock()
    mock_client_instance.get.return_value = mock_response
    mock_httpx_client.return_value = mock_client_instance
    
    # Test
    client = SynapCores()
    
    with pytest.raises(AuthenticationError) as exc_info:
        client.get_collection("test")
    
    assert "Authentication failed" in str(exc_info.value)


def test_not_found_error(mock_httpx_client):
    """Test not found error handling."""
    # Setup mock
    mock_response = Mock()
    mock_response.status_code = 404
    
    mock_client_instance = Mock()
    mock_client_instance.get.return_value = mock_response
    mock_httpx_client.return_value = mock_client_instance
    
    # Test
    client = SynapCores()
    
    with pytest.raises(NotFoundError) as exc_info:
        client.get_collection("nonexistent")
    
    assert "Resource not found" in str(exc_info.value)


def test_sql_query(mock_httpx_client):
    """Test SQL query execution."""
    # Setup mock
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "rows": [{"id": 1, "name": "test"}],
        "columns": ["id", "name"],
        "row_count": 1,
        "took_ms": 10.5,
    }
    
    mock_client_instance = Mock()
    mock_client_instance.post.return_value = mock_response
    mock_httpx_client.return_value = mock_client_instance
    
    # Test
    client = SynapCores()
    result = client.sql("SELECT * FROM test", as_dataframe=False)
    
    assert result.row_count == 1
    assert result.columns == ["id", "name"]
    assert result.took_ms == 10.5
    assert len(result.rows) == 1


def test_embed(mock_httpx_client):
    """Test embedding generation."""
    # Setup mock
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]]
    }
    
    mock_client_instance = Mock()
    mock_client_instance.post.return_value = mock_response
    mock_httpx_client.return_value = mock_client_instance
    
    # Test single text
    client = SynapCores()
    embedding = client.embed("test text")
    
    assert embedding == [0.1, 0.2, 0.3]
    
    # Test batch
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    }
    
    embeddings = client.embed(["text1", "text2"])
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]


def test_context_manager(mock_httpx_client):
    """Test client context manager."""
    mock_client_instance = Mock()
    mock_httpx_client.return_value = mock_client_instance
    
    with SynapCores() as client:
        assert isinstance(client, SynapCores)
    
    mock_client_instance.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])