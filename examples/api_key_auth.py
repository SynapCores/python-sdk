"""
Example: API Key Authentication with SynapCores Python SDK

This example demonstrates how to authenticate using API keys with the SynapCores Python SDK.
API keys provide a secure way to authenticate programmatic access to your AIDB instance.
"""

from synapcores import SynapCores
from synapcores.exceptions import AuthenticationError
import os

def main():
    # Get API key from environment variable (recommended for production)
    # You can set this with: export AIDB_API_KEY="aidb_sk_your_key_here"
    api_key = os.getenv("AIDB_API_KEY")
    
    if not api_key:
        print("Warning: AIDB_API_KEY environment variable not set")
        print("Using demo API key for testing (replace with your actual key)")
        api_key = "aidb_sk_demo_key_12345"  # Replace with your actual API key
    
    try:
        # Initialize client with API key authentication
        client = SynapCores(
            host="localhost",
            port=8080,
            api_key=api_key,
            use_https=False  # Set to True in production
        )
        
        print(f"✅ Successfully connected to AIDB with API key")
        
        # Test the connection by listing collections
        collections = client.list_collections()
        print(f"📚 Available collections: {collections}")
        
        # Execute a simple query
        result = client.sql("SELECT 1 as test, NOW() as current_time")
        print(f"🔍 Query result: {result.rows}")
        
        # Test permissions by trying to create a collection
        # This will fail if the API key has ReadOnly permission
        try:
            test_collection = client.create_collection(
                name="test_api_key_collection",
                schema={
                    "id": "integer",
                    "data": "string"
                }
            )
            print(f"✅ API key has write permissions - collection created")
            
            # Clean up
            client.delete_collection("test_api_key_collection")
            
        except AuthenticationError as e:
            print(f"ℹ️ API key has read-only permissions: {e}")
        
    except AuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("Please check that:")
        print("1. Your API key is valid and starts with 'aidb_'")
        print("2. The API key has not expired")
        print("3. The API key is active (not revoked)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check that AIDB is running on localhost:8080")

if __name__ == "__main__":
    main()