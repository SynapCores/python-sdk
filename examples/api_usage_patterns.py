#!/usr/bin/env python3
"""
SynapCores Python SDK - API Usage Patterns

This example demonstrates common usage patterns and best practices
for the enhanced SynapCores Python SDK with complete SQL and vector support.

Requirements:
    pip install synapcores pandas numpy

Usage:
    python api_usage_patterns.py
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any

from synapcores import (
    SynapCores,
    ColumnDefinition,
    CreateTableOptions,
    IndexDefinition,
    TransactionOptions,
    BatchInsertOptions,
    Vector,
    KNNSearchOptions,
    VectorError,
    SQLError,
    TransactionError,
)


def basic_sql_operations_pattern():
    """Demonstrates basic SQL operations pattern."""
    print("📊 Basic SQL Operations Pattern")
    print("-" * 30)

    with SynapCores(host="localhost", port=8080, api_key="aidb_your_api_key_here") as client:

        # Create table with proper type definitions
        columns = [
            ColumnDefinition(name="id", data_type="INTEGER", constraints=[{"type": "PRIMARY_KEY"}]),
            ColumnDefinition(name="name", data_type="VARCHAR(100)", constraints=[{"type": "NOT_NULL"}]),
            ColumnDefinition(name="score", data_type="DECIMAL(10,2)"),
            ColumnDefinition(name="metadata", data_type="JSONB"),
        ]

        client.create_table("demo_table", columns, CreateTableOptions(if_not_exists=True))
        print("✅ Table created")

        # Insert data using parameterized queries
        client.sql("""
            INSERT INTO demo_table (id, name, score, metadata)
            VALUES ($1, $2, $3, $4)
        """, {
            "1": 1,
            "2": "John Doe",
            "3": 95.5,
            "4": '{"department": "engineering", "level": "senior"}'
        })
        print("✅ Data inserted")

        # Query with parameters and return as DataFrame
        df = client.sql("""
            SELECT * FROM demo_table
            WHERE score > $1 AND JSON_EXTRACT(metadata, '$.level') = $2
        """, {"1": 90.0, "2": "senior"})

        print(f"📊 Query returned {len(df)} rows")
        return df


def transaction_management_pattern():
    """Demonstrates transaction management pattern."""
    print("\n💳 Transaction Management Pattern")
    print("-" * 35)

    with SynapCores(host="localhost", port=8080, api_key="aidb_your_api_key_here") as client:

        # Start transaction with specific isolation level
        transaction = client.begin_transaction(TransactionOptions(
            isolation_level="READ_COMMITTED",
            timeout=30000
        ))

        try:
            # Perform multiple related operations
            client.sql("INSERT INTO accounts (id, balance) VALUES (1, 1000.00)")
            client.sql("INSERT INTO accounts (id, balance) VALUES (2, 500.00)")

            # Create savepoint before risky operation
            savepoint = client.create_savepoint("before_transfer")

            try:
                # Transfer money
                client.sql("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
                client.sql("UPDATE accounts SET balance = balance + 100 WHERE id = 2")

                # Validate balances
                result = client.sql("SELECT SUM(balance) as total FROM accounts")
                if result.iloc[0]['total'] != 1500.00:
                    raise ValueError("Balance validation failed")

                # Release savepoint if validation passes
                client.release_savepoint("before_transfer")

            except Exception as e:
                print(f"⚠️  Transfer failed: {e}")
                client.rollback_to_savepoint("before_transfer")

            # Commit the transaction
            client.commit_transaction()
            print("✅ Transaction committed successfully")

        except Exception as e:
            print(f"❌ Transaction failed: {e}")
            client.rollback_transaction()
            raise


def batch_operations_pattern():
    """Demonstrates efficient batch operations pattern."""
    print("\n📦 Batch Operations Pattern")
    print("-" * 30)

    with SynapCores(host="localhost", port=8080, api_key="aidb_your_api_key_here") as client:

        # Prepare batch data
        employee_data = [
            {"id": i, "name": f"Employee {i}", "department": f"Dept {i % 3}", "salary": 50000 + (i * 1000)}
            for i in range(1, 101)  # 100 employees
        ]

        # Batch insert with error handling
        batch_options = BatchInsertOptions(
            table_name="employees",
            columns=["id", "name", "department", "salary"],
            rows=[[emp["id"], emp["name"], emp["department"], emp["salary"]] for emp in employee_data],
            on_conflict="REPLACE",
            batch_size=25,  # Process in chunks of 25
            return_errors=True
        )

        result = client.batch_insert(batch_options)
        print(f"✅ Batch insert: {result.successful}/{result.total_processed} successful")

        if result.errors:
            print(f"⚠️  {len(result.errors)} errors occurred")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"   Row {error.item_index}: {error.error_message}")


def vector_operations_pattern():
    """Demonstrates vector operations pattern."""
    print("\n🔢 Vector Operations Pattern")
    print("-" * 30)

    with SynapCores(host="localhost", port=8080, api_key="aidb_your_api_key_here") as client:

        # Create vectors using different input formats
        vector1 = [1.0, 2.0, 3.0, 4.0]  # Python list
        vector2 = np.array([0.5, 1.5, 2.5, 3.5])  # NumPy array
        vector3 = Vector(values=[2.0, 4.0, 6.0, 8.0])  # Vector object

        # Vector arithmetic
        try:
            # Add vectors
            result = client.vector_add(vector1, vector2)
            print(f"Vector addition result: {result.result.values}")

            # Calculate similarities
            cosine_sim = client.cosine_similarity(vector1, vector2)
            print(f"Cosine similarity: {cosine_sim.similarity:.4f}")

            # Calculate magnitude
            magnitude = client.vector_magnitude(vector1)
            print(f"Vector magnitude: {magnitude.magnitude:.4f}")

            # Normalize vector
            normalized = client.normalize_vector(vector1)
            print(f"Normalized vector: {[f'{v:.3f}' for v in normalized.result.values]}")

        except VectorError as e:
            print(f"❌ Vector operation failed: {e}")
            if e.vector_dimension_mismatch:
                print("   Cause: Vector dimension mismatch")


def vector_search_pattern():
    """Demonstrates vector search pattern."""
    print("\n🔍 Vector Search Pattern")
    print("-" * 25)

    with SynapCores(host="localhost", port=8080, api_key="aidb_your_api_key_here") as client:

        # Setup search table with embeddings
        try:
            client.sql("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    embedding VECTOR(384),
                    metadata JSONB
                )
            """)

            # Generate sample embeddings (in real use, these would come from an embedding model)
            documents = [
                {"id": 1, "title": "AI Research", "content": "Machine learning and neural networks"},
                {"id": 2, "title": "Cooking Tips", "content": "How to cook pasta perfectly"},
                {"id": 3, "title": "Data Science", "content": "Statistical analysis and data mining"},
            ]

            for doc in documents:
                # Generate deterministic embedding for demo
                embedding = np.random.RandomState(doc["id"]).normal(0, 1, 384)
                embedding = embedding / np.linalg.norm(embedding)  # Normalize

                client.sql("""
                    INSERT OR REPLACE INTO documents (id, title, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                """, {
                    "1": doc["id"],
                    "2": doc["title"],
                    "3": doc["content"],
                    "4": embedding.tolist(),
                    "5": f'{{"category": "demo", "created_at": "{datetime.now().isoformat()}"}}'
                })

            # Perform similarity search
            query_embedding = np.random.RandomState(1).normal(0, 1, 384)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # K-NN search
            knn_options = KNNSearchOptions(
                query_vector=query_embedding.tolist(),
                k=2,
                table_name="documents",
                vector_column="embedding",
                metadata_columns=["id", "title", "content"],
                metric="cosine"
            )

            results = client.knn_search(knn_options)
            print(f"🎯 KNN Search found {len(results)} results:")
            for i, result in enumerate(results):
                title = result.metadata.get("title", "Unknown")
                similarity = result.similarity
                print(f"  {i+1}. {title} (similarity: {similarity:.4f})")

        except Exception as e:
            print(f"⚠️  Vector search setup failed: {e}")


def advanced_sql_pattern():
    """Demonstrates advanced SQL features pattern."""
    print("\n⚡ Advanced SQL Pattern")
    print("-" * 25)

    with SynapCores(host="localhost", port=8080, api_key="aidb_your_api_key_here") as client:

        # Prepared statements for repeated queries
        try:
            # Prepare statement
            stmt = client.prepare_statement(
                "SELECT * FROM employees WHERE department = $1 AND salary > $2",
                {"name": "dept_high_earners"}
            )

            # Execute multiple times with different parameters
            engineering_result = client.execute_prepared("dept_high_earners", ["Engineering", 80000])
            sales_result = client.execute_prepared("dept_high_earners", ["Sales", 70000])

            print(f"Engineering high earners: {len(engineering_result)}")
            print(f"Sales high earners: {len(sales_result)}")

            # Clean up
            client.deallocate_prepared("dept_high_earners")

        except Exception as e:
            print(f"⚠️  Prepared statements: {e}")

        # JSON operations
        try:
            # Extract data from JSON columns
            result = client.json_query(
                table_name="employees",
                json_column="profile_data",
                operation="extract",
                path="skills"
            )
            print(f"📄 Extracted skills from {len(result)} employee profiles")

            # Update JSON data
            client.json_query(
                table_name="employees",
                json_column="profile_data",
                operation="update",
                path="last_review",
                value=datetime.now().isoformat(),
                where_clause="id = 1"
            )
            print("✅ Updated JSON data")

        except Exception as e:
            print(f"⚠️  JSON operations: {e}")


def error_handling_pattern():
    """Demonstrates proper error handling patterns."""
    print("\n⚠️  Error Handling Pattern")
    print("-" * 25)

    with SynapCores(host="localhost", port=8080, api_key="aidb_your_api_key_here") as client:

        try:
            # Example of handling specific exceptions
            client.sql("SELECT * FROM non_existent_table")

        except SQLError as e:
            print(f"SQL Error: {e.message}")
            if e.sql_state:
                print(f"SQL State: {e.sql_state}")

        try:
            # Vector dimension mismatch
            client.vector_add([1, 2, 3], [1, 2])  # Different dimensions

        except VectorError as e:
            print(f"Vector Error: {e.message}")
            if e.vector_dimension_mismatch:
                print("Cause: Dimension mismatch between vectors")

        try:
            # Transaction error
            client.commit_transaction()  # No active transaction

        except TransactionError as e:
            print(f"Transaction Error: {e.message}")
            if e.transaction_id:
                print(f"Transaction ID: {e.transaction_id}")


def performance_optimization_pattern():
    """Demonstrates performance optimization patterns."""
    print("\n🚀 Performance Optimization Pattern")
    print("-" * 38)

    with SynapCores(host="localhost", port=8080, api_key="aidb_your_api_key_here") as client:

        # Create indexes for better query performance
        try:
            # Create composite index
            index_def = IndexDefinition(
                name="idx_employee_dept_salary",
                table_name="employees",
                columns=[
                    {"name": "department", "order": "ASC"},
                    {"name": "salary", "order": "DESC"}
                ],
                if_not_exists=True
            )

            client.create_index(index_def)
            print("✅ Performance index created")

            # Use batch operations instead of individual inserts
            # Use prepared statements for repeated queries
            # Use appropriate data types (VECTOR vs TEXT for embeddings)
            # Use transactions for multi-step operations

            print("✅ Performance optimizations applied")

        except Exception as e:
            print(f"⚠️  Performance optimization: {e}")


def main():
    """Run all usage pattern demonstrations."""
    print("🚀 SynapCores Python SDK - API Usage Patterns")
    print("=" * 50)

    try:
        basic_sql_operations_pattern()
        transaction_management_pattern()
        batch_operations_pattern()
        vector_operations_pattern()
        vector_search_pattern()
        advanced_sql_pattern()
        error_handling_pattern()
        performance_optimization_pattern()

        print("\n🎉 All usage patterns demonstrated successfully!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()