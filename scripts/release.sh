#!/bin/bash

# Abi AI Framework Release Script
# This script builds and packages the Abi AI Framework for release

set -e

echo "🚀 Abi AI Framework Release Build Script"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    if ! command -v zig &> /dev/null; then
        print_error "Zig is not installed or not in PATH"
        exit 1
    fi

    ZIG_VERSION=$(zig version)
    print_success "Zig version: $ZIG_VERSION"

    if ! command -v git &> /dev/null; then
        print_error "Git is not installed or not in PATH"
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Clean build artifacts
clean_build() {
    print_status "Cleaning previous build artifacts..."
    rm -rf zig-cache zig-out
    print_success "Build artifacts cleaned"
}

# Run tests
run_tests() {
    print_status "Running comprehensive test suite..."

    if ! zig build test; then
        print_error "Tests failed!"
        exit 1
    fi

    print_success "All tests passed"
}

# Build release binaries
build_release() {
    print_status "Building release binaries..."

    # Build optimized release version
    if ! zig build -Doptimize=ReleaseFast; then
        print_error "Release build failed!"
        exit 1
    fi

    print_success "Release binaries built successfully"

    # List built artifacts
    print_status "Built artifacts:"
    find zig-out/bin -type f -executable | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "  $file ($size)"
    done
}

# Generate documentation
generate_docs() {
    print_status "Generating API documentation..."

    if ! zig build docs; then
        print_warning "Documentation generation failed, but continuing..."
    else
        print_success "API documentation generated"
    fi
}

# Create release archive
create_archive() {
    local version=$(git describe --tags --abbrev=0 2>/dev/null || echo "v1.0.0")
    local archive_name="abi-framework-${version#v}-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"

    print_status "Creating release archive: $archive_name.tar.gz"

    # Create temporary directory for archive
    local temp_dir=$(mktemp -d)
    local archive_dir="$temp_dir/$archive_name"

    mkdir -p "$archive_dir"

    # Copy built artifacts
    cp -r zig-out "$archive_dir/"
    cp -r docs "$archive_dir/" 2>/dev/null || true
    cp README.md "$archive_dir/"
    cp CHANGELOG.md "$archive_dir/"
    cp LICENSE "$archive_dir/"

    # Create archive
    cd "$temp_dir"
    tar -czf "$archive_name.tar.gz" "$archive_name"

    # Move archive to current directory
    mv "$archive_name.tar.gz" "$(pwd)/"

    # Cleanup
    rm -rf "$temp_dir"

    print_success "Release archive created: $archive_name.tar.gz"
}

# Show release information
show_release_info() {
    local version=$(git describe --tags --abbrev=0 2>/dev/null || echo "v1.0.0")

    echo
    echo "🎉 Release Build Complete!"
    echo "=========================="
    echo "Version: $version"
    echo "Date: $(date)"
    echo
    echo "Built Artifacts:"
    ls -la zig-out/bin/ 2>/dev/null || echo "  No binaries found"
    echo
    echo "Release Archive:"
    ls -la *.tar.gz 2>/dev/null || echo "  No archive created"
    echo
    echo "Next Steps:"
    echo "1. Test the built binaries: ./zig-out/bin/abi --help"
    echo "2. Run benchmarks: zig build benchmark"
    echo "3. Deploy using: See deploy/ directory for Kubernetes manifests"
    echo "4. Publish documentation: See docs/ directory"
}

# Main execution
main() {
    echo "Starting Abi AI Framework release build..."
    echo

    check_prerequisites
    clean_build
    run_tests
    build_release
    generate_docs
    create_archive
    show_release_info

    echo
    print_success "🎉 Abi AI Framework release build completed successfully!"
    print_success "Ready for production deployment!"
}

# Run main function
main "$@"
