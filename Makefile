.PHONY: all build test clean install run bench docs fmt

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

# Run specific module tests
test-agent:
	zig build test-agent

test-database:
	zig build test-database

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

# Run benchmarks
bench:
	zig build bench

# Build documentation
docs:
	zig build docs

# Format code
fmt:
	zig fmt src/

# Check formatting
fmt-check:
	zig fmt --check src/

# Development workflow
dev: fmt build test

# Quick build and run
quick: build run

# Full CI pipeline locally
ci: fmt-check build test bench docs

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Build the project (default)"
	@echo "  build     - Build debug version"
	@echo "  release   - Build release version with optimizations"
	@echo "  test      - Run all tests"
	@echo "  clean     - Clean build artifacts"
	@echo "  install   - Install binaries"
	@echo "  run       - Run the main application"
	@echo "  tui       - Run in TUI mode"
	@echo "  web       - Run web server"
	@echo "  bench     - Run benchmarks"
	@echo "  docs      - Build documentation"
	@echo "  fmt       - Format code"
	@echo "  dev       - Development workflow (format, build, test)"
	@echo "  ci        - Run full CI pipeline locally" 