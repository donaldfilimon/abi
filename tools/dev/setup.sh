#!/bin/bash
# Development Setup Script for ABI Framework
# Sets up development environment and dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check for Zig
    if ! command -v zig &> /dev/null; then
        print_error "Zig is not installed. Please install Zig 0.16.0-dev or later."
        print_status "Visit: https://ziglang.org/download/"
        exit 1
    fi
    
    local zig_version=$(zig version)
    print_success "Found Zig: $zig_version"
    
    # Check for Git
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed"
        exit 1
    fi
    
    print_success "Found Git: $(git --version)"
    
    # Check for C++ compiler
    if command -v g++ &> /dev/null; then
        print_success "Found C++ compiler: $(g++ --version | head -n1)"
    elif command -v clang++ &> /dev/null; then
        print_success "Found C++ compiler: $(clang++ --version | head -n1)"
    else
        print_warning "No C++ compiler found. Some dependencies may not build."
    fi
}

# Setup git hooks
setup_git_hooks() {
    print_status "Setting up Git hooks..."
    
    mkdir -p .git/hooks
    
    # Pre-commit hook
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for ABI Framework

set -e

echo "Running pre-commit checks..."

# Format code
echo "Formatting code..."
zig fmt .

# Run linter
echo "Running linter..."
zig build lint

# Run tests
echo "Running tests..."
zig build test

echo "Pre-commit checks passed!"
EOF

    chmod +x .git/hooks/pre-commit
    print_success "Git hooks configured"
}

# Setup development configuration
setup_dev_config() {
    print_status "Setting up development configuration..."
    
    # Create .env file for development
    cat > .env << 'EOF'
# ABI Framework Development Environment
ZIG_BUILD_OPTIMIZE=Debug
ABI_LOG_LEVEL=debug
ABI_ENABLE_PROFILING=true
ABI_MEMORY_LIMIT_MB=1024
EOF

    print_success "Development configuration created"
}

# Install development dependencies
install_dependencies() {
    print_status "Installing development dependencies..."
    
    # Create tools directory structure
    mkdir -p tools/{build,dev,deploy}
    
    # Make scripts executable
    find tools -name "*.sh" -exec chmod +x {} \;
    
    print_success "Development dependencies installed"
}

# Run initial build and test
run_initial_setup() {
    print_status "Running initial build and test..."
    
    # Clean build
    if [ -d "zig-out" ]; then
        rm -rf zig-out
    fi
    if [ -d "zig-cache" ]; then
        rm -rf zig-cache
    fi
    
    # Build project
    print_status "Building project..."
    if zig build; then
        print_success "Initial build successful"
    else
        print_error "Initial build failed"
        exit 1
    fi
    
    # Run tests
    print_status "Running tests..."
    if zig build test; then
        print_success "Initial tests passed"
    else
        print_error "Initial tests failed"
        exit 1
    fi
    
    # Generate documentation
    print_status "Generating documentation..."
    if zig build docs; then
        print_success "Documentation generated"
    else
        print_warning "Documentation generation failed (non-critical)"
    fi
}

# Main setup function
main() {
    print_status "ABI Framework Development Setup"
    print_status "==============================="
    
    check_requirements
    setup_git_hooks
    setup_dev_config
    install_dependencies
    run_initial_setup
    
    print_success "Development environment setup completed!"
    print_status ""
    print_status "Next steps:"
    print_status "1. Run './tools/build/build.sh --help' to see build options"
    print_status "2. Run './tools/dev/setup.sh' to verify setup"
    print_status "3. Check docs/guides/getting-started.md for usage examples"
    print_status "4. Read CONTRIBUTING.md for development guidelines"
}

# Run main function
main "$@"