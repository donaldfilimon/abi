#!/bin/bash
# Run benchmarks for Linux/macOS

set -e

echo "=== Running Benchmarks ==="

echo "Building benchmarks..."
zig build benchmarks

echo "Running quick benchmark..."
zig build run -- bench quick

echo "=== Benchmarks Complete ==="
