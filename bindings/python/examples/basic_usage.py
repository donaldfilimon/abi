#!/usr/bin/env python3
"""
ABI Framework Basic Usage Example

Demonstrates core functionality including initialization,
vector operations, and database usage.
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import abi


def main():
    print("=" * 60)
    print("ABI Framework Basic Usage Example")
    print("=" * 60)

    # Initialize the framework
    print("\n1. Initializing ABI framework...")
    abi.init()
    print(f"   Version: {abi.version()}")
    print(f"   Initialized: {abi.is_initialized()}")
    print(f"   SIMD available: {abi.has_simd()}")

    # Vector operations
    print("\n2. Vector Operations")
    print("-" * 40)

    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]

    similarity = abi.cosine_similarity(a, b)
    print(f"   Vectors: a = {a}")
    print(f"            b = {b}")
    print(f"   Cosine similarity: {similarity:.4f}")

    dot = abi.vector_dot(a, b)
    print(f"   Dot product: {dot:.4f}")

    sum_vec = abi.vector_add(a, b)
    print(f"   Sum: {sum_vec}")

    norm = abi.l2_norm(a)
    print(f"   L2 norm of a: {norm:.4f}")

    # Vector database
    print("\n3. Vector Database")
    print("-" * 40)

    db = abi.VectorDatabase(name="example_db", dimensions=4)
    print(f"   Created database: {db.name}")

    # Add vectors with metadata
    db.add([1.0, 0.0, 0.0, 0.0], metadata={"label": "x-axis", "category": "axis"})
    db.add([0.0, 1.0, 0.0, 0.0], metadata={"label": "y-axis", "category": "axis"})
    db.add([0.0, 0.0, 1.0, 0.0], metadata={"label": "z-axis", "category": "axis"})
    db.add([0.0, 0.0, 0.0, 1.0], metadata={"label": "w-axis", "category": "axis"})
    db.add([0.5, 0.5, 0.0, 0.0], metadata={"label": "diagonal-xy", "category": "diagonal"})

    print(f"   Added {db.count} vectors")

    # Search for similar vectors
    query = [0.9, 0.1, 0.0, 0.0]
    results = db.search(query, top_k=3)
    print(f"   Query: {query}")
    print(f"   Top 3 results:")
    for r in results:
        print(f"      ID: {r.id}, Score: {r.score:.4f}, Label: {r.metadata.get('label')}")

    # Get statistics
    stats = db.stats()
    print(f"   Database stats:")
    print(f"      Vectors: {stats.vector_count}")
    print(f"      Dimensions: {stats.dimensions}")
    print(f"      Index type: {stats.index_type}")

    # Agent example
    print("\n4. AI Agent")
    print("-" * 40)

    agent = abi.Agent(name="assistant")
    response = agent.process("Hello! What can you do?")
    print(f"   User: Hello! What can you do?")
    print(f"   Agent: {response}")

    response = agent.process("Tell me about vectors.")
    print(f"   User: Tell me about vectors.")
    print(f"   Agent: {response}")

    print(f"   Conversation history: {len(agent.get_history())} messages")

    # Cleanup
    print("\n5. Cleanup")
    print("-" * 40)
    abi.shutdown()
    print("   Framework shut down successfully")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
