#!/usr/bin/env python3
"""
Complete SQL and Vector Operations Demo for SynapCores Python SDK

This comprehensive example demonstrates all the new SQL and vector capabilities:
- Complete table management (CREATE, ALTER, DROP, DESCRIBE, SHOW)
- Transaction support with isolation levels and savepoints
- Index management operations
- Batch operations with error tracking
- Advanced SQL features (CTEs, window functions, JSON/JSONB operations)
- Vector arithmetic operations
- Vector similarity functions (cosine, L2 distance, inner product)
- Vector search operations (KNN, range-based, hybrid search)
- Vector SQL support with operators and functions

Requirements:
    pip install synapcores pandas numpy

Usage:
    python complete_sql_vector_demo.py
"""

import asyncio
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Import SynapCores SDK classes
from synapcores import (
    SynapCores,
    ColumnDefinition,
    CreateTableOptions,
    AlterTableOptions,
    IndexDefinition,
    TransactionOptions,
    BatchInsertOptions,
    BatchUpdateOptions,
    CTEDefinition,
    WindowFunction,
    WindowFunctionOptions,
    Vector,
    KNNSearchOptions,
    RangeSearchOptions,
    HybridSearchOptions,
    VectorArithmeticResult,
    SQLError,
    VectorError,
    TransactionError,
)


class SQLVectorDemo:
    """Comprehensive demo of SQL and vector operations."""

    def __init__(self):
        self.client = SynapCores(
            host="localhost",
            port=8080,
            api_key="aidb_your_api_key_here",  # Replace with your actual API key
            use_https=False,
            timeout=30.0,
        )

    def run_demo(self):
        """Run the complete demonstration."""
        print("🚀 SynapCores Complete SQL & Vector Operations Demo")
        print("=" * 60)

        try:
            # 1. Table Management Demo
            self.table_management_demo()

            # 2. Index Management Demo
            self.index_management_demo()

            # 3. Transaction Management Demo
            self.transaction_demo()

            # 4. Batch Operations Demo
            self.batch_operations_demo()

            # 5. Advanced SQL Features Demo
            self.advanced_sql_demo()

            # 6. Vector Arithmetic Demo
            self.vector_arithmetic_demo()

            # 7. Vector Similarity Demo
            self.vector_similarity_demo()

            # 8. Vector Search Demo
            self.vector_search_demo()

            # 9. Vector SQL Integration Demo
            self.vector_sql_demo()

            print("\n🎉 All demos completed successfully!")

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup()

    def table_management_demo(self):
        """Demonstrate table management operations."""
        print("\n📊 TABLE MANAGEMENT OPERATIONS")
        print("=" * 40)

        # Define comprehensive table schema
        employee_columns = [
            ColumnDefinition(
                name="id",
                data_type="INTEGER",
                constraints=[{"type": "PRIMARY_KEY"}]
            ),
            ColumnDefinition(
                name="employee_number",
                data_type="VARCHAR(20)",
                constraints=[{"type": "UNIQUE"}, {"type": "NOT_NULL"}]
            ),
            ColumnDefinition(
                name="first_name",
                data_type="VARCHAR(50)",
                constraints=[{"type": "NOT_NULL"}]
            ),
            ColumnDefinition(
                name="last_name",
                data_type="VARCHAR(50)",
                constraints=[{"type": "NOT_NULL"}]
            ),
            ColumnDefinition(
                name="email",
                data_type="VARCHAR(255)",
                constraints=[
                    {"type": "UNIQUE"},
                    {"type": "NOT_NULL"},
                    {"type": "CHECK", "expression": "email LIKE '%@%.%'"}
                ]
            ),
            ColumnDefinition(
                name="department_id",
                data_type="INTEGER"
            ),
            ColumnDefinition(
                name="salary",
                data_type="DECIMAL(10,2)",
                constraints=[{"type": "CHECK", "expression": "salary > 0"}]
            ),
            ColumnDefinition(
                name="hire_date",
                data_type="DATE",
                constraints=[{"type": "NOT_NULL"}]
            ),
            ColumnDefinition(
                name="profile_data",
                data_type="JSONB"
            ),
            ColumnDefinition(
                name="skills_embedding",
                data_type="VECTOR(384)"
            ),
            ColumnDefinition(
                name="is_active",
                data_type="BOOLEAN",
                default_value=True
            ),
            ColumnDefinition(
                name="created_at",
                data_type="TIMESTAMP",
                default_value="CURRENT_TIMESTAMP"
            )
        ]

        # Create table with options
        options = CreateTableOptions(if_not_exists=True)
        print("🏗️  Creating employees table...")
        self.client.create_table("employees", employee_columns, options)
        print("✅ Employees table created")

        # Create departments table for foreign key demonstration
        dept_columns = [
            ColumnDefinition(
                name="id",
                data_type="INTEGER",
                constraints=[{"type": "PRIMARY_KEY"}]
            ),
            ColumnDefinition(
                name="name",
                data_type="VARCHAR(100)",
                constraints=[{"type": "UNIQUE"}, {"type": "NOT_NULL"}]
            ),
            ColumnDefinition(
                name="budget",
                data_type="DECIMAL(12,2)"
            )
        ]

        print("🏗️  Creating departments table...")
        self.client.create_table("departments", dept_columns, CreateTableOptions(if_not_exists=True))
        print("✅ Departments table created")

        # Show all tables
        print("\n📋 Listing all tables:")
        tables = self.client.show_tables()
        for table in tables:
            print(f"  - {table}")

        # Describe table structure
        print("\n📖 Describing employees table:")
        try:
            table_info = self.client.describe_table("employees")
            print(f"Table: {table_info.name}")
            for col in table_info.columns[:5]:  # Show first 5 columns
                print(f"  {col['name']}: {col['type']} {'NOT NULL' if not col.get('nullable', True) else ''}")
        except Exception as e:
            print(f"⚠️  Table description: {e}")

        # Alter table - add new column
        print("\n🔧 Adding phone column to employees table...")
        try:
            alter_options = AlterTableOptions(
                action="ADD_COLUMN",
                column_definition=ColumnDefinition(
                    name="phone",
                    data_type="VARCHAR(20)"
                )
            )
            self.client.alter_table("employees", alter_options)
            print("✅ Phone column added")
        except Exception as e:
            print(f"⚠️  Column addition: {e}")

    def index_management_demo(self):
        """Demonstrate index management operations."""
        print("\n🔍 INDEX MANAGEMENT")
        print("=" * 25)

        # Create single-column unique index
        email_index = IndexDefinition(
            name="idx_employees_email",
            table_name="employees",
            columns=[{"name": "email", "order": "ASC"}],
            unique=True,
            if_not_exists=True
        )

        print("📇 Creating unique email index...")
        try:
            self.client.create_index(email_index)
            print("✅ Email index created")
        except Exception as e:
            print(f"⚠️  Index creation: {e}")

        # Create composite index
        name_index = IndexDefinition(
            name="idx_employees_name",
            table_name="employees",
            columns=[
                {"name": "last_name", "order": "ASC"},
                {"name": "first_name", "order": "ASC"}
            ],
            if_not_exists=True
        )

        print("📇 Creating composite name index...")
        try:
            self.client.create_index(name_index)
            print("✅ Composite name index created")
        except Exception as e:
            print(f"⚠️  Index creation: {e}")

        # Show all indexes
        print("\n📋 Listing indexes:")
        try:
            indexes = self.client.show_indexes()
            for idx in indexes[:3]:  # Show first 3 indexes
                unique_str = "UNIQUE" if idx.unique else ""
                print(f"  {idx.name} on {idx.table} {unique_str}")
        except Exception as e:
            print(f"⚠️  Index listing: {e}")

    def transaction_demo(self):
        """Demonstrate transaction management."""
        print("\n💳 TRANSACTION MANAGEMENT")
        print("=" * 30)

        # Insert sample departments first
        print("📝 Inserting sample departments...")
        self.client.sql("""
            INSERT OR REPLACE INTO departments (id, name, budget) VALUES
            (1, 'Engineering', 1000000.00),
            (2, 'Sales', 500000.00),
            (3, 'Marketing', 300000.00)
        """, as_dataframe=False)
        print("✅ Sample departments inserted")

        # Example 1: Successful transaction
        print("\n🔄 Transaction Example 1: Successful Transaction")
        transaction_options = TransactionOptions(
            isolation_level="READ_COMMITTED",
            read_only=False,
            timeout=10000
        )

        transaction = self.client.begin_transaction(transaction_options)
        print(f"✅ Transaction {transaction.id[:8]}... started")

        try:
            # Insert employees within transaction
            self.client.sql("""
                INSERT INTO employees (id, employee_number, first_name, last_name, email, department_id, salary, hire_date, profile_data)
                VALUES (1, 'EMP001', 'Alice', 'Johnson', 'alice.johnson@company.com', 1, 95000.00, '2024-01-15',
                        '{"skills": ["Python", "JavaScript"], "level": "senior"}')
            """, as_dataframe=False)

            self.client.sql("""
                INSERT INTO employees (id, employee_number, first_name, last_name, email, department_id, salary, hire_date, profile_data)
                VALUES (2, 'EMP002', 'Bob', 'Smith', 'bob.smith@company.com', 1, 87000.00, '2024-02-01',
                        '{"skills": ["Java", "React"], "level": "mid"}')
            """, as_dataframe=False)

            print("✅ Employees inserted successfully")
            self.client.commit_transaction()
            print("✅ Transaction committed successfully")

        except Exception as e:
            print(f"❌ Error in transaction: {e}")
            self.client.rollback_transaction()

        # Example 2: Savepoint demonstration
        print("\n🔄 Transaction Example 2: Savepoint Demonstration")
        transaction = self.client.begin_transaction()
        print(f"✅ Transaction {transaction.id[:8]}... started")

        try:
            # Create savepoint
            savepoint = self.client.create_savepoint("before_risky_operation")
            print(f"✅ Savepoint '{savepoint.name}' created")

            # Insert valid data
            self.client.sql("""
                INSERT INTO employees (id, employee_number, first_name, last_name, email, department_id, salary, hire_date)
                VALUES (10, 'EMP010', 'Charlie', 'Brown', 'charlie.brown@company.com', 2, 75000.00, '2024-03-01')
            """, as_dataframe=False)

            # Simulate error - try invalid data
            try:
                self.client.sql("""
                    INSERT INTO employees (id, employee_number, first_name, last_name, email, department_id, salary, hire_date)
                    VALUES (11, 'EMP010', 'Duplicate', 'Employee', 'duplicate@company.com', 2, -1000.00, '2024-03-01')
                """, as_dataframe=False)
            except Exception:
                print("❌ Expected error occurred, rolling back to savepoint...")
                self.client.rollback_to_savepoint("before_risky_operation")
                print("✅ Rolled back to savepoint successfully")

            self.client.commit_transaction()
            print("✅ Transaction committed with savepoint recovery")

        except Exception as e:
            print(f"❌ Transaction error: {e}")
            self.client.rollback_transaction()

    def batch_operations_demo(self):
        """Demonstrate batch operations."""
        print("\n📦 BATCH OPERATIONS")
        print("=" * 25)

        # Batch insert example
        print("📥 Batch Insert Example:")
        employee_data = [
            [20, 'EMP020', 'David', 'Wilson', 'david.wilson@company.com', 2, 82000.00, '2024-02-15', '{"skills": ["Sales"], "level": "senior"}'],
            [21, 'EMP021', 'Emma', 'Brown', 'emma.brown@company.com', 3, 70000.00, '2024-01-30', '{"skills": ["Marketing"], "level": "mid"}'],
            [22, 'EMP022', 'Frank', 'Miller', 'frank.miller@company.com', 1, 65000.00, '2024-02-10', '{"skills": ["DevOps"], "level": "mid"}'],
        ]

        batch_options = BatchInsertOptions(
            table_name="employees",
            columns=["id", "employee_number", "first_name", "last_name", "email", "department_id", "salary", "hire_date", "profile_data"],
            rows=employee_data,
            on_conflict="REPLACE",
            batch_size=100
        )

        try:
            result = self.client.batch_insert(batch_options)
            print(f"✅ Batch insert completed: {result.successful}/{result.total_processed} successful")
            if result.errors:
                print(f"⚠️  {len(result.errors)} errors occurred")
        except Exception as e:
            print(f"⚠️  Batch insert failed: {e}")

        # Batch update example
        print("\n📝 Batch Update Example:")
        update_data = [
            {"set": {"salary": 98000.00}, "where": {"employee_number": "EMP001"}},
            {"set": {"salary": 90000.00}, "where": {"employee_number": "EMP002"}},
        ]

        batch_update_options = BatchUpdateOptions(
            table_name="employees",
            updates=update_data,
            batch_size=50
        )

        try:
            result = self.client.batch_update(batch_update_options)
            print(f"✅ Batch update completed: {result.successful}/{result.total_processed} successful")
        except Exception as e:
            print(f"⚠️  Batch update failed: {e}")

    def advanced_sql_demo(self):
        """Demonstrate advanced SQL features."""
        print("\n⚡ ADVANCED SQL FEATURES")
        print("=" * 30)

        # Prepared Statements
        print("📝 Prepared Statements:")
        try:
            prepared_stmt = self.client.prepare_statement(
                "SELECT * FROM employees WHERE department_id = $1 AND salary > $2",
                {"name": "find_high_earners"}
            )
            print(f"✅ Prepared statement created: {prepared_stmt.id[:8]}...")

            result = self.client.execute_prepared("find_high_earners", [1, 80000], as_dataframe=False)
            print(f"📊 Found {result.row_count} high earners in Engineering")

            self.client.deallocate_prepared("find_high_earners")
            print("✅ Prepared statement deallocated")

        except Exception as e:
            print(f"⚠️  Prepared statements: {e}")

        # Common Table Expressions (CTEs)
        print("\n🔗 Common Table Expressions (CTEs):")
        ctes = [
            CTEDefinition(
                name="department_stats",
                query="""
                    SELECT
                        d.id,
                        d.name,
                        COUNT(e.id) as employee_count,
                        AVG(e.salary) as avg_salary
                    FROM departments d
                    LEFT JOIN employees e ON d.id = e.department_id
                    GROUP BY d.id, d.name
                """
            )
        ]

        try:
            result = self.client.query_with_ctes(
                ctes,
                """
                SELECT
                    name as department,
                    employee_count,
                    avg_salary
                FROM department_stats
                ORDER BY avg_salary DESC
                """,
                as_dataframe=False
            )

            print("📊 Department analysis with CTEs:")
            for row in result.rows:
                print(f"  {row['department']}: {row['employee_count']} employees, avg salary: ${row.get('avg_salary', 0):.2f}")

        except Exception as e:
            print(f"⚠️  CTEs not supported: {e}")

        # JSON Operations
        print("\n📄 JSON Operations:")
        try:
            # Extract skills from JSON
            result = self.client.json_query(
                "employees",
                "profile_data",
                "extract",
                "skills",
                as_dataframe=False
            )
            print("📊 Employee skills extracted from JSON:")
            for row in result.rows[:3]:  # Show first 3
                skills = row.get('extracted_value', '[]')
                print(f"  {row.get('first_name', 'Unknown')}: {skills}")

        except Exception as e:
            print(f"⚠️  JSON operations: {e}")

    def vector_arithmetic_demo(self):
        """Demonstrate vector arithmetic operations."""
        print("\n🧮 VECTOR ARITHMETIC OPERATIONS")
        print("=" * 40)

        # Sample vectors
        vector1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        vector2 = [0.5, 1.5, 2.5, 3.5, 4.5]
        scalar = 2.5

        print(f"Vector 1: {vector1}")
        print(f"Vector 2: {vector2}")
        print(f"Scalar: {scalar}\n")

        # Vector addition
        print("➕ Vector Addition:")
        try:
            result = self.client.vector_add(vector1, vector2)
            print(f"Result: {result.result.values}")
            print(f"Time: {result.took_ms:.2f}ms\n")
        except Exception as e:
            print(f"⚠️  Vector addition: {e}\n")

        # Vector subtraction
        print("➖ Vector Subtraction:")
        try:
            result = self.client.vector_subtract(vector1, vector2)
            print(f"Result: {result.result.values}")
            print(f"Time: {result.took_ms:.2f}ms\n")
        except Exception as e:
            print(f"⚠️  Vector subtraction: {e}\n")

        # Scalar multiplication
        print("✖️ Scalar Multiplication:")
        try:
            result = self.client.vector_scalar_multiply(vector1, scalar)
            print(f"Result: {result.result.values}")
            print(f"Time: {result.took_ms:.2f}ms\n")
        except Exception as e:
            print(f"⚠️  Scalar multiplication: {e}\n")

        # Dot product
        print("🔵 Dot Product:")
        try:
            result = self.client.vector_dot_product(vector1, vector2)
            print(f"Result: {result.similarity}")
            print(f"Time: {result.took_ms:.2f}ms\n")
        except Exception as e:
            print(f"⚠️  Dot product: {e}\n")

        # Vector magnitude
        print("📏 Vector Magnitude:")
        try:
            result = self.client.vector_magnitude(vector1)
            print(f"Magnitude: {result.magnitude:.4f}")
            print(f"Time: {result.took_ms:.2f}ms\n")
        except Exception as e:
            print(f"⚠️  Vector magnitude: {e}\n")

        # Vector normalization
        print("🎯 Vector Normalization:")
        try:
            result = self.client.normalize_vector(vector1)
            normalized_values = [f"{v:.4f}" for v in result.result.values]
            print(f"Normalized: [{', '.join(normalized_values)}]")
            print(f"Time: {result.took_ms:.2f}ms\n")
        except Exception as e:
            print(f"⚠️  Vector normalization: {e}\n")

    def vector_similarity_demo(self):
        """Demonstrate vector similarity functions."""
        print("\n📐 VECTOR SIMILARITY FUNCTIONS")
        print("=" * 40)

        # Create similar and orthogonal vectors
        vector1 = [0.8, 0.6, 0.0, 0.0]
        vector2 = [0.9, 0.436, 0.0, 0.0]  # Similar to vector1
        vector3 = [0.0, 0.0, 0.8, 0.6]    # Orthogonal to vector1

        print(f"Vector 1: {vector1}")
        print(f"Vector 2: {vector2} (similar)")
        print(f"Vector 3: {vector3} (orthogonal)\n")

        # Cosine similarity
        print("📊 Cosine Similarity:")
        try:
            sim12 = self.client.cosine_similarity(vector1, vector2)
            sim13 = self.client.cosine_similarity(vector1, vector3)
            print(f"Vector1 ↔ Vector2: {sim12.similarity:.4f} (should be high)")
            print(f"Vector1 ↔ Vector3: {sim13.similarity:.4f} (should be low)\n")
        except Exception as e:
            print(f"⚠️  Cosine similarity: {e}\n")

        # L2 distance
        print("📏 L2 (Euclidean) Distance:")
        try:
            dist12 = self.client.l2_distance(vector1, vector2)
            dist13 = self.client.l2_distance(vector1, vector3)
            print(f"Vector1 ↔ Vector2: {dist12.distance:.4f} (should be small)")
            print(f"Vector1 ↔ Vector3: {dist13.distance:.4f} (should be large)\n")
        except Exception as e:
            print(f"⚠️  L2 distance: {e}\n")

        # Inner product
        print("🔶 Inner Product:")
        try:
            inner12 = self.client.inner_product(vector1, vector2)
            inner13 = self.client.inner_product(vector1, vector3)
            print(f"Vector1 ↔ Vector2: {inner12.similarity:.4f}")
            print(f"Vector1 ↔ Vector3: {inner13.similarity:.4f} (should be ~0)\n")
        except Exception as e:
            print(f"⚠️  Inner product: {e}\n")

    def vector_search_demo(self):
        """Demonstrate vector search operations."""
        print("\n🔍 VECTOR SEARCH OPERATIONS")
        print("=" * 35)

        # Setup vector search data
        self.setup_vector_search_data()

        # Generate query vector
        query_vector = self.generate_category_embedding("technology", 384)

        # K-Nearest Neighbors Search
        print("🎯 K-Nearest Neighbors (KNN) Search:")
        knn_options = KNNSearchOptions(
            query_vector=query_vector,
            k=5,
            table_name="vector_search_demo",
            vector_column="embedding",
            metadata_columns=["id", "name", "category"]
        )

        try:
            results = self.client.knn_search(knn_options)
            print(f"Found {len(results)} nearest neighbors:")
            for i, result in enumerate(results):
                name = result.metadata.get("name", "Unknown")
                category = result.metadata.get("category", "Unknown")
                print(f"  {i+1}. {name} ({category}) - similarity: {result.similarity:.4f}")
        except Exception as e:
            print(f"⚠️  KNN search: {e}")

        print()

        # Range-based Search
        print("📊 Range-based Similarity Search:")
        range_options = RangeSearchOptions(
            query_vector=query_vector,
            threshold=0.5,
            table_name="vector_search_demo",
            vector_column="embedding",
            metadata_columns=["id", "name", "category"],
            max_results=10
        )

        try:
            results = self.client.range_search(range_options)
            print(f"Found {len(results)} items above similarity threshold:")
            for i, result in enumerate(results):
                name = result.metadata.get("name", "Unknown")
                category = result.metadata.get("category", "Unknown")
                print(f"  {i+1}. {name} ({category}) - similarity: {result.similarity:.4f}")
        except Exception as e:
            print(f"⚠️  Range search: {e}")

    def vector_sql_demo(self):
        """Demonstrate vector SQL integration."""
        print("\n🔄 VECTOR SQL INTEGRATION")
        print("=" * 30)

        # Vector operators in SQL
        print("📊 Vector SQL with Similarity Functions:")
        query_vector = self.generate_category_embedding("technology", 384)

        try:
            result = self.client.vector_sql_query("""
                SELECT
                    name,
                    category,
                    COSINE_SIMILARITY(embedding, $1) as similarity
                FROM vector_search_demo
                WHERE COSINE_SIMILARITY(embedding, $1) > 0.5
                ORDER BY similarity DESC
                LIMIT 5
            """, {"1": query_vector}, as_dataframe=False)

            print(f"Found {result.row_count} similar items using Vector SQL:")
            for row in result.rows:
                print(f"  {row['name']} ({row['category']}) - similarity: {row['similarity']:.4f}")

        except Exception as e:
            print(f"⚠️  Vector SQL: {e}")

    def setup_vector_search_data(self):
        """Setup sample data for vector search demonstrations."""
        print("📋 Setting up vector search data...")

        # Create vector search demo table
        try:
            self.client.sql("""
                CREATE TABLE IF NOT EXISTS vector_search_demo (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    embedding VECTOR(384),
                    metadata JSONB
                )
            """, as_dataframe=False)

            # Generate and insert sample data
            categories = ['technology', 'sports', 'cooking', 'travel', 'music']
            for i in range(20):
                category = categories[i % len(categories)]
                embedding = self.generate_category_embedding(category, 384)

                self.client.sql("""
                    INSERT OR REPLACE INTO vector_search_demo (id, name, category, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                """, {
                    "1": i + 1,
                    "2": f"Item {i + 1}",
                    "3": category,
                    "4": embedding,
                    "5": json.dumps({
                        "score": np.random.rand() * 100,
                        "tags": [category, f"tag{i % 3}"],
                        "created_at": datetime.now().isoformat()
                    })
                }, as_dataframe=False)

            print(f"✅ Inserted 20 sample items for vector search")

        except Exception as e:
            print(f"⚠️  Vector search setup: {e}")

    def generate_category_embedding(self, category: str, dimensions: int) -> List[float]:
        """Generate a deterministic embedding based on category."""
        # Simple deterministic embedding generation for demo purposes
        seed = sum(ord(char) for char in category)
        np.random.seed(seed)

        embedding = np.random.normal(0, 1, dimensions)
        # Normalize to unit vector
        magnitude = np.linalg.norm(embedding)
        if magnitude > 0:
            embedding = embedding / magnitude

        return embedding.tolist()

    def cleanup(self):
        """Clean up demo resources."""
        print("\n🧹 Cleaning up...")
        try:
            # Drop demo tables
            self.client.drop_table("vector_search_demo", if_exists=True)
            self.client.drop_table("employees", if_exists=True)
            self.client.drop_table("departments", if_exists=True)
            print("✅ Demo tables dropped")

            # Close client connection
            self.client.close()
            print("✅ Client connection closed")

        except Exception as e:
            print(f"⚠️  Cleanup: {e}")


def main():
    """Main function to run the demo."""
    demo = SQLVectorDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()