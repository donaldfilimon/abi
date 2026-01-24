#!/usr/bin/env python3
"""
ABI Framework Vector Database Example

Demonstrates vector database capabilities including
CRUD operations, search, filtering, and batch operations.
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import abi
from abi.database import (
    VectorDatabase,
    DatabaseConfig,
    DistanceMetric,
    IndexType,
)


def main():
    print("=" * 60)
    print("ABI Framework Vector Database Example")
    print("=" * 60)

    # Initialize
    abi.init()
    print(f"\nABI version: {abi.version()}")

    # Create database with configuration
    print("\n1. Creating Vector Database")
    print("-" * 40)

    config = DatabaseConfig(
        path=":memory:",  # In-memory database
        dimensions=384,
        distance_metric=DistanceMetric.COSINE,
        index_type=IndexType.HNSW,
    )

    db = VectorDatabase(name="embeddings", config=config)
    print(f"   Database: {db.name}")
    print(f"   Dimensions: {db.dimensions}")
    print(f"   Distance metric: {config.distance_metric.name}")
    print(f"   Index type: {config.index_type.name}")

    # Add vectors
    print("\n2. Adding Vectors")
    print("-" * 40)

    # Sample documents with embeddings (using small vectors for demo)
    documents = [
        {
            "text": "Machine learning is a subset of artificial intelligence",
            "category": "tech",
            "year": 2023,
        },
        {
            "text": "Deep learning uses neural networks with many layers",
            "category": "tech",
            "year": 2022,
        },
        {
            "text": "Natural language processing enables computers to understand text",
            "category": "tech",
            "year": 2023,
        },
        {
            "text": "The weather today is sunny and warm",
            "category": "weather",
            "year": 2024,
        },
        {
            "text": "Climate change affects global weather patterns",
            "category": "weather",
            "year": 2023,
        },
    ]

    # Generate mock embeddings (in production, use a real embedding model)
    import random

    random.seed(42)

    for i, doc in enumerate(documents):
        # Create a mock embedding
        embedding = [random.gauss(0, 1) for _ in range(384)]
        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        embedding = [x / norm for x in embedding]

        doc_id = db.add(embedding, metadata=doc, id=i)
        print(f"   Added document {doc_id}: {doc['text'][:40]}...")

    print(f"   Total vectors: {db.count}")

    # Search
    print("\n3. Similarity Search")
    print("-" * 40)

    # Create a query embedding (mock)
    query_embedding = [random.gauss(0, 1) for _ in range(384)]
    norm = sum(x * x for x in query_embedding) ** 0.5
    query_embedding = [x / norm for x in query_embedding]

    results = db.search(query_embedding, top_k=3)
    print(f"   Query: [vector with {len(query_embedding)} dimensions]")
    print(f"   Top 3 results:")
    for r in results:
        print(f"      ID: {r.id}, Score: {r.score:.4f}")
        print(f"         Text: {r.metadata.get('text', '')[:50]}...")

    # Filtered search
    print("\n4. Filtered Search")
    print("-" * 40)

    # Search only in 'tech' category
    results = db.search(
        query_embedding,
        top_k=3,
        filter={"category": "tech"},
    )
    print("   Filter: category == 'tech'")
    print(f"   Results:")
    for r in results:
        print(f"      ID: {r.id}, Score: {r.score:.4f}, Category: {r.metadata.get('category')}")

    # Search with year filter
    results = db.search(
        query_embedding,
        top_k=3,
        filter={"year": {"$gte": 2023}},
    )
    print("\n   Filter: year >= 2023")
    print(f"   Results:")
    for r in results:
        print(f"      ID: {r.id}, Score: {r.score:.4f}, Year: {r.metadata.get('year')}")

    # Batch operations
    print("\n5. Batch Operations")
    print("-" * 40)

    batch_records = [
        {
            "vector": [random.gauss(0, 1) for _ in range(384)],
            "metadata": {"text": "Batch document 1", "batch": True},
        },
        {
            "vector": [random.gauss(0, 1) for _ in range(384)],
            "metadata": {"text": "Batch document 2", "batch": True},
        },
        {
            "vector": [random.gauss(0, 1) for _ in range(384)],
            "metadata": {"text": "Batch document 3", "batch": True},
        },
    ]

    result = db.add_batch(batch_records)
    print(f"   Batch insert: {result.success_count} success, {result.error_count} errors")
    print(f"   Total vectors: {db.count}")

    # CRUD operations
    print("\n6. CRUD Operations")
    print("-" * 40)

    # Get vector by ID
    vec = db.get(0)
    if vec:
        print(f"   Get ID 0: {vec['metadata'].get('text', '')[:40]}...")

    # Update vector metadata
    updated = db.update(0, metadata={"text": "Updated document", "updated": True})
    print(f"   Update ID 0: {updated}")

    # Delete vector
    deleted = db.delete(0)
    print(f"   Delete ID 0: {deleted}")
    print(f"   Total vectors after delete: {db.count}")

    # Statistics
    print("\n7. Database Statistics")
    print("-" * 40)

    stats = db.stats()
    print(f"   Vector count: {stats.vector_count}")
    print(f"   Dimensions: {stats.dimensions}")
    print(f"   Index type: {stats.index_type}")
    print(f"   Memory usage: {stats.memory_usage_bytes / 1024:.2f} KB")

    # Hybrid search
    print("\n8. Hybrid Search")
    print("-" * 40)

    # Add vectors back for hybrid search demo
    db.add(
        [random.gauss(0, 1) for _ in range(384)],
        metadata={"text": "Machine learning and AI", "category": "tech"},
    )

    results = db.hybrid_search(
        query_vector=query_embedding,
        query_text="machine learning",
        top_k=3,
        alpha=0.7,  # 70% vector, 30% text
    )
    print("   Query text: 'machine learning'")
    print("   Alpha: 0.7 (70% vector, 30% text)")
    print(f"   Results:")
    for r in results:
        text = r.metadata.get("text", "")[:40]
        print(f"      Score: {r.score:.4f}, Text: {text}...")

    # Persistence
    print("\n9. Persistence")
    print("-" * 40)

    # Save to file
    save_path = "example_vectors.json"
    db.save(save_path)
    print(f"   Saved to: {save_path}")

    # Load from file
    loaded_db = VectorDatabase.load(save_path)
    print(f"   Loaded: {loaded_db.count} vectors")

    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"   Cleaned up: {save_path}")

    # Using convenience functions
    print("\n10. Convenience Functions")
    print("-" * 40)

    # Create database with helper
    quick_db = abi.create_database(name="quick", dimensions=128)
    print(f"   Created quick database: {quick_db.name}")

    # Cleanup
    print("\n11. Cleanup")
    print("-" * 40)
    db.clear()
    abi.shutdown()
    print("   Database cleared and framework shut down")

    print("\n" + "=" * 60)
    print("Vector Database Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
