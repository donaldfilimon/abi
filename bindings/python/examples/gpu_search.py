#!/usr/bin/env python3
"""Example: GPU-accelerated vector search with ABI Framework.

This example demonstrates:
1. Selecting a GPU backend for acceleration
2. Creating a vector database with GPU support
3. Inserting vectors and performing similarity search
4. Comparing different GPU backend options

Run from the ABI repository root after building:
    zig build lib
    python bindings/python/examples/gpu_search.py

Or with a specific backend:
    python bindings/python/examples/gpu_search.py --backend cuda
    python bindings/python/examples/gpu_search.py --backend vulkan
    python bindings/python/examples/gpu_search.py --backend cpu
"""

import sys
import os
import argparse
import time
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abi import ABI, GpuBackend


def normalize(vector):
    """Normalize a vector to unit length."""
    magnitude = sum(x * x for x in vector) ** 0.5
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


def main():
    parser = argparse.ArgumentParser(
        description='GPU-accelerated vector search example'
    )
    parser.add_argument(
        '--backend',
        choices=GpuBackend.VALID_BACKENDS,
        default='auto',
        help='GPU backend to use (default: auto)'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=128,
        help='Vector dimension (default: 128)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1000,
        help='Number of vectors to insert (default: 1000)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of nearest neighbors to find (default: 10)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ABI Framework - GPU-Accelerated Vector Search")
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
    print(f"\nConfiguration:")
    print(f"  GPU Backend: {args.backend}")
    print(f"  Vector dimension: {args.dimension}")
    print(f"  Number of vectors: {args.count}")
    print(f"  Top-K results: {args.k}")

    # Create database with specified backend
    print(f"\nCreating vector database with {args.backend} backend...")
    try:
        db = abi.create_db(args.dimension, backend=args.backend)
        print(f"  Database created successfully")
        print(f"  Backend: {db.backend}")
    except ValueError as e:
        print(f"  Error: {e}")
        return 1

    # Generate and insert random vectors
    print(f"\nInserting {args.count} random vectors...")
    vectors = {}
    start_time = time.time()

    for i in range(args.count):
        # Create random vector and normalize it
        vec = [random.gauss(0, 1) for _ in range(args.dimension)]
        vec = normalize(vec)
        vectors[i] = vec
        db.insert(i, vec)

    insert_time = time.time() - start_time
    print(f"  Inserted {args.count} vectors in {insert_time:.3f}s")
    print(f"  Rate: {args.count / insert_time:.0f} vectors/sec")

    # Perform search
    print(f"\nPerforming similarity search...")

    # Use an existing vector (should return itself as top match)
    query_id = random.randint(0, args.count - 1)
    query = vectors[query_id]

    start_time = time.time()
    results = db.search(query, k=args.k)
    search_time = time.time() - start_time

    print(f"\n  Query: vector ID {query_id}")
    print(f"  Search time: {search_time * 1000:.2f}ms")
    print(f"  Backend used: {db.backend}")
    print(f"\n  Top {args.k} results:")
    print(f"  {'Rank':<6} {'ID':<8} {'Score':<12}")
    print(f"  {'-' * 26}")

    for rank, result in enumerate(results, 1):
        print(f"  {rank:<6} {result['id']:<8} {result['score']:<12.6f}")

    # Benchmark: Multiple searches
    print(f"\n" + "=" * 60)
    print(f"Benchmark: 100 random searches (backend: {db.backend})")
    print("=" * 60)

    num_searches = 100
    start_time = time.time()

    for _ in range(num_searches):
        q = normalize([random.gauss(0, 1) for _ in range(args.dimension)])
        db.search(q, k=args.k)

    total_time = time.time() - start_time
    avg_time = total_time / num_searches

    print(f"\n  Total time: {total_time:.3f}s")
    print(f"  Average per search: {avg_time * 1000:.2f}ms")
    print(f"  Queries per second: {num_searches / total_time:.0f}")

    # Show available backends
    print(f"\n" + "=" * 60)
    print("Available GPU Backends")
    print("=" * 60)
    print(f"\n  Available options: {', '.join(GpuBackend.VALID_BACKENDS)}")
    print(f"\n  Usage examples:")
    print(f"    db = abi.create_db(128, backend='cuda')    # NVIDIA CUDA")
    print(f"    db = abi.create_db(128, backend='vulkan')  # Vulkan (cross-platform)")
    print(f"    db = abi.create_db(128, backend='metal')   # Apple Metal (macOS)")
    print(f"    db = abi.create_db(128, backend='cpu')     # CPU fallback")
    print(f"    db = abi.create_db(128, backend='auto')    # Auto-select best")

    print(f"\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
