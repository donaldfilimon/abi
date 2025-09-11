.PHONY: all build release test test-all test-heavy clean install run tui web bench bench-all bench-db bench-neural bench-simple bench-simd docs fmt fmt-check dev quick ci analyze profile perf-guard perf-ci cross-platform test-network help

# Default target
all: build

# Build the project
build:
	zig build -Doptimize=Debug

# Build release version
release:
	zig build -Doptimize=ReleaseFast -Dgpu=true -Dsimd=true

# Run tests
test:
	zig build test

# Aggregate tests
test-all:
	zig build test-all

# Heavy/integration tests
test-heavy:
	zig build test-heavy

# Run specific module tests
test-agent:
	zig build test-agent

test-database:
	zig build test-database

# SIMD tests
test-simd:
	zig build test-simd_vector
	zig build test-simd_text

# Clean build artifacts
clean:
	rm -rf zig-out zig-cache .zig-cache

# Install the binaries
install:
	zig build install

# Run the main application
run:
	zig build run

# Run TUI mode
tui:
	zig build run -- tui

# Run web server
web:
	zig build run -- web

# Run benchmarks (unified)
bench:
	zig build bench

# Run all benchmark suites
bench-all:
	zig build bench-all

# Specific benchmark suites
bench-db:
	zig build benchmark-db

bench-neural:
	zig build benchmark-neural

bench-simple:
	zig build benchmark-simple

bench-simd:
	zig build bench-simd

# Build documentation
docs:
	zig build docs

# Static analysis
analyze:
	zig build analyze

# Performance profiling
profile:
	zig build profile

# Performance regression guard (pass threshold in ns: make perf-guard PERF=50000000)
perf-guard:
	zig build perf-guard -- $(PERF)

# Performance CI pipeline
perf-ci:
	zig build perf-ci

# Cross-platform verification
cross-platform:
	zig build cross-platform

# Windows network diagnostics (Windows only)
test-network:
	zig build test-network

# Format code
fmt:
	zig fmt src/ tests/ tools/ benchmarks/

# Check formatting
fmt-check:
	zig fmt --check src/ tests/ tools/ benchmarks/

# Development workflow
dev: fmt build test

# Quick build and run
quick: build run

# Full CI pipeline locally
ci: fmt-check build test bench-all docs analyze

# Help
help:
	@echo "Available targets:"
	@echo "  all            - Build the project (default)"
	@echo "  build          - Build debug version"
	@echo "  release        - Build release version with optimizations"
	@echo "  test           - Run unit tests"
	@echo "  test-all       - Run all tests (unit + integration + heavy)"
	@echo "  test-heavy     - Run heavy tests"
	@echo "  clean          - Clean build artifacts"
	@echo "  install        - Install binaries"
	@echo "  run            - Run the main application"
	@echo "  tui            - Run in TUI mode"
	@echo "  web            - Run web server"
	@echo "  bench          - Run unified benchmark"
	@echo "  bench-all      - Run all benchmarks"
	@echo "  bench-db       - Database benchmarks"
	@echo "  bench-neural   - Neural network benchmarks"
	@echo "  bench-simple   - Simple benchmarks"
	@echo "  bench-simd     - SIMD micro-benchmark"
	@echo "  docs           - Build documentation"
	@echo "  analyze        - Run static analysis"
	@echo "  profile        - Run performance profiling"
	@echo "  perf-guard     - Run perf regression guard (PERF=threshold_ns)"
	@echo "  perf-ci        - Run performance CI suite"
	@echo "  cross-platform - Cross-target build verification"
	@echo "  test-network   - Windows network diagnostics"
	@echo "  fmt/fmt-check  - Format code / verify formatting"
	@echo "  dev            - Format, build, and test"
	@echo "  ci             - Run full CI pipeline locally" 