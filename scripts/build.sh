#!/bin/bash
# Build script for WDBX-AI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ZIG_VERSION="0.15.1"
BUILD_DIR="zig-out"
RELEASE_DIR="release"

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_msg "$BLUE" "Checking prerequisites..."
    
    # Check Zig installation
    if ! command -v zig &> /dev/null; then
        print_msg "$RED" "Error: Zig is not installed"
        print_msg "$YELLOW" "Please install Zig $ZIG_VERSION from https://ziglang.org/download/"
        exit 1
    fi
    
    # Check Zig version
    local zig_version=$(zig version 2>&1)
    print_msg "$GREEN" "Found Zig version: $zig_version"
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_msg "$YELLOW" "Warning: Git is not installed"
    fi
}

# Clean build artifacts
clean() {
    print_msg "$BLUE" "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR"
    rm -rf "$RELEASE_DIR"
    rm -rf zig-cache
    print_msg "$GREEN" "Clean complete"
}

# Build project
build() {
    local mode=${1:-debug}
    print_msg "$BLUE" "Building WDBX-AI in $mode mode..."
    
    case $mode in
        debug)
            zig build
            ;;
        release)
            zig build -Doptimize=ReleaseFast
            ;;
        release-safe)
            zig build -Doptimize=ReleaseSafe
            ;;
        release-small)
            zig build -Doptimize=ReleaseSmall
            ;;
        *)
            print_msg "$RED" "Unknown build mode: $mode"
            exit 1
            ;;
    esac
    
    print_msg "$GREEN" "Build complete"
}

# Run tests
run_tests() {
    print_msg "$BLUE" "Running tests..."
    
    # Run unit tests
    print_msg "$YELLOW" "Running unit tests..."
    zig build test
    
    # Run integration tests
    print_msg "$YELLOW" "Running integration tests..."
    zig build test-integration
    
    print_msg "$GREEN" "All tests passed"
}

# Run benchmarks
run_benchmarks() {
    print_msg "$BLUE" "Running benchmarks..."
    zig build bench
    print_msg "$GREEN" "Benchmarks complete"
}

# Format code
format_code() {
    print_msg "$BLUE" "Formatting code..."
    zig build fmt
    print_msg "$GREEN" "Code formatting complete"
}

# Generate documentation
generate_docs() {
    print_msg "$BLUE" "Generating documentation..."
    zig build docs
    print_msg "$GREEN" "Documentation generated in $BUILD_DIR/docs"
}

# Create release package
create_release() {
    local version=${1:-$(git describe --tags --always 2>/dev/null || echo "dev")}
    print_msg "$BLUE" "Creating release package v$version..."
    
    # Build release binaries
    build release
    
    # Create release directory
    mkdir -p "$RELEASE_DIR/wdbx-ai-$version"
    
    # Copy binaries
    cp -r "$BUILD_DIR/bin" "$RELEASE_DIR/wdbx-ai-$version/"
    
    # Copy configuration
    mkdir -p "$RELEASE_DIR/wdbx-ai-$version/config"
    cp -r config/* "$RELEASE_DIR/wdbx-ai-$version/config/" 2>/dev/null || true
    
    # Copy documentation
    cp README.md "$RELEASE_DIR/wdbx-ai-$version/"
    cp LICENSE "$RELEASE_DIR/wdbx-ai-$version/" 2>/dev/null || true
    cp -r docs "$RELEASE_DIR/wdbx-ai-$version/" 2>/dev/null || true
    
    # Create version file
    echo "$version" > "$RELEASE_DIR/wdbx-ai-$version/VERSION"
    
    # Create archive
    cd "$RELEASE_DIR"
    tar -czf "wdbx-ai-$version.tar.gz" "wdbx-ai-$version"
    cd ..
    
    print_msg "$GREEN" "Release package created: $RELEASE_DIR/wdbx-ai-$version.tar.gz"
}

# Show help
show_help() {
    cat << EOF
WDBX-AI Build Script

Usage: $0 [command] [options]

Commands:
    build [mode]     Build the project (debug, release, release-safe, release-small)
    test             Run tests
    bench            Run benchmarks
    clean            Clean build artifacts
    format           Format source code
    docs             Generate documentation
    release [ver]    Create release package
    all              Run all build steps
    help             Show this help message

Examples:
    $0 build                 # Build in debug mode
    $0 build release         # Build in release mode
    $0 test                  # Run all tests
    $0 release 2.0.0         # Create release v2.0.0
    $0 all                   # Run complete build pipeline

EOF
}

# Main script
main() {
    local command=${1:-help}
    shift || true
    
    case $command in
        build)
            check_prerequisites
            build "$@"
            ;;
        test)
            check_prerequisites
            run_tests
            ;;
        bench)
            check_prerequisites
            run_benchmarks
            ;;
        clean)
            clean
            ;;
        format)
            check_prerequisites
            format_code
            ;;
        docs)
            check_prerequisites
            generate_docs
            ;;
        release)
            check_prerequisites
            create_release "$@"
            ;;
        all)
            check_prerequisites
            clean
            format_code
            build release
            run_tests
            run_benchmarks
            generate_docs
            create_release
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_msg "$RED" "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"