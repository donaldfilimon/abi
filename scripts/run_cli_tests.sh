#!/bin/bash
# CLI smoke tests for Linux/macOS

set -e

echo "=== Running CLI Smoke Tests ==="

echo "Testing --help..."
zig build run -- --help > /dev/null

echo "Testing system-info..."
zig build run -- system-info > /dev/null

echo "Testing gpu backends..."
zig build run -- gpu backends > /dev/null

echo "Testing db --help..."
zig build run -- db --help > /dev/null

echo "Testing llm --help..."
zig build run -- llm --help > /dev/null

echo "Testing model --help..."
zig build run -- model --help > /dev/null

echo "=== All CLI Tests Passed ==="
