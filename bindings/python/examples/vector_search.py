#!/usr/bin/env python3
"""Example: Vector similarity search with ABI Framework.

This example demonstrates:
1. Initializing the ABI framework
2. Creating a vector database
3. Inserting vectors with IDs
4. Performing similarity search
5. Handling results

Run from the ABI repository root after building:
    zig build lib
    python bindings/python/examples/vector_search.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abi import ABI, AbiError
import random
import time


def normalize(vector):
    """Normalize a vector to unit length."""
    magnitude = sum(x * x for x in vector) ** 0.5
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    print("=" * 60)
    print("ABI Framework - Vector Similarity Search Example")
    print("=" * 60)

    # Initialize ABI
    try:
        abi = ABI()
        print(f"\nABI Framework version: {abi.version()}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure to build the shared library first: zig build lib")
        return 1

    # Configuration
    dimension = 128
    num_vectors = 1000
    k = 10

    print(f"\nConfiguration:")
    print(f"  Vector dimension: {dimension}")
    print(f"  Number of vectors: {num_vectors}")
    print(f"  Top-K results: {k}")

    # Create database
    print(f"\nCreating vector database...")
    db = abi.create_db(dimension)
    print(f"  Database created successfully")

    # Generate and insert random vectors
    print(f"\nInserting {num_vectors} random vectors...")
    vectors = {}
    start_time = time.time()

    for i in range(num_vectors):
        # Create random vector and normalize it
        vec = [random.gauss(0, 1) for _ in range(dimension)]
        vec = normalize(vec)
        vectors[i] = vec
        db.insert(i, vec)

    insert_time = time.time() - start_time
    print(f"  Inserted {num_vectors} vectors in {insert_time:.3f}s")
    print(f"  Rate: {num_vectors / insert_time:.0f} vectors/sec")

    # Perform searches
    print(f"\nPerforming similarity search...")

    # Search 1: Use an existing vector (should return itself as top match)
    query_id = random.randint(0, num_vectors - 1)
    query = vectors[query_id]

    start_time = time.time()
    results = db.search(query, k=k)
    search_time = time.time() - start_time

    print(f"\n  Query: vector ID {query_id}")
    print(f"  Search time: {search_time * 1000:.2f}ms")
    print(f"\n  Top {k} results:")
    print(f"  {'Rank':<6} {'ID':<8} {'Score':<12} {'Expected Sim':<12}")
    print(f"  {'-' * 38}")

    for rank, result in enumerate(results, 1):
        result_id = result['id']
        score = result['score']
        # Compute expected similarity for verification
        if result_id in vectors:
            expected = cosine_similarity(query, vectors[result_id])
        else:
            expected = float('nan')
        print(f"  {rank:<6} {result_id:<8} {score:<12.6f} {expected:<12.6f}")

    # Search 2: Random query vector
    print(f"\n  Query: random vector (not in database)")
    random_query = normalize([random.gauss(0, 1) for _ in range(dimension)])

    start_time = time.time()
    results = db.search(random_query, k=k)
    search_time = time.time() - start_time

    print(f"  Search time: {search_time * 1000:.2f}ms")
    print(f"\n  Top {k} results:")
    print(f"  {'Rank':<6} {'ID':<8} {'Score':<12}")
    print(f"  {'-' * 26}")

    for rank, result in enumerate(results, 1):
        print(f"  {rank:<6} {result['id']:<8} {result['score']:<12.6f}")

    # Benchmark: Multiple searches
    print(f"\n" + "=" * 60)
    print("Benchmark: 100 random searches")
    print("=" * 60)

    num_searches = 100
    start_time = time.time()

    for _ in range(num_searches):
        q = normalize([random.gauss(0, 1) for _ in range(dimension)])
        db.search(q, k=k)

    total_time = time.time() - start_time
    avg_time = total_time / num_searches

    print(f"\n  Total time: {total_time:.3f}s")
    print(f"  Average per search: {avg_time * 1000:.2f}ms")
    print(f"  Queries per second: {num_searches / total_time:.0f}")

    print(f"\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
