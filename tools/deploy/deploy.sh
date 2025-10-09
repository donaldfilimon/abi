#!/bin/bash
# Deployment Script for ABI Framework
# Handles building and deploying the framework for different environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
BUILD_TYPE="release"
TARGET="native"
PACKAGE=false
DOCKER=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --package)
            PACKAGE=true
            shift
            ;;
        --docker)
            DOCKER=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --environment <env>  Target environment (development|staging|production)"
            echo "  --build-type <type>  Build type (debug|release)"
            echo "  --target <target>    Build target (native|x86_64-linux|aarch64-linux|...)"
            echo "  --package           Create distribution package"
            echo "  --docker            Build Docker image"
            echo "  --verbose           Verbose output"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        development|staging|production)
            print_status "Deploying to: $ENVIRONMENT"
            ;;
        *)
            print_error "Invalid environment: $ENVIRONMENT"
            print_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

# Build the project
build_project() {
    print_status "Building ABI Framework for $ENVIRONMENT..."
    
    local build_cmd="zig build"
    
    # Set optimization level
    case $BUILD_TYPE in
        debug)
            build_cmd="$build_cmd -Doptimize=Debug"
            ;;
        release)
            build_cmd="$build_cmd -Doptimize=ReleaseFast"
            ;;
        *)
            print_error "Invalid build type: $BUILD_TYPE"
            exit 1
            ;;
    esac
    
    # Set target
    if [ "$TARGET" != "native" ]; then
        build_cmd="$build_cmd -Dtarget=$TARGET"
    fi
    
    if [ "$VERBOSE" = true ]; then
        build_cmd="$build_cmd --verbose"
    fi
    
    if $build_cmd; then
        print_success "Build completed successfully"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    local test_cmd="zig build test"
    if [ "$BUILD_TYPE" = "release" ]; then
        test_cmd="$test_cmd -Doptimize=ReleaseFast"
    fi
    
    if [ "$TARGET" != "native" ]; then
        test_cmd="$test_cmd -Dtarget=$TARGET"
    fi
    
    if $test_cmd; then
        print_success "All tests passed"
    else
        print_error "Tests failed"
        exit 1
    fi
}

# Create distribution package
create_package() {
    print_status "Creating distribution package..."
    
    local package_name="abi-framework-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
    local package_dir="dist/$package_name"
    
    # Create package directory
    mkdir -p "$package_dir"
    
    # Copy binaries
    if [ -d "zig-out/bin" ]; then
        cp -r zig-out/bin "$package_dir/"
    fi
    
    # Copy documentation
    if [ -d "docs" ]; then
        cp -r docs "$package_dir/"
    fi
    
    # Copy examples
    if [ -d "examples" ]; then
        cp -r examples "$package_dir/"
    fi
    
    # Copy configuration files
    if [ -d "config" ]; then
        cp -r config "$package_dir/"
    fi
    
    # Create package info
    cat > "$package_dir/package-info.txt" << EOF
ABI Framework Package
====================
Environment: $ENVIRONMENT
Build Type: $BUILD_TYPE
Target: $TARGET
Build Date: $(date)
Zig Version: $(zig version)
EOF
    
    # Create archive
    cd dist
    tar -czf "${package_name}.tar.gz" "$package_name"
    cd ..
    
    print_success "Package created: dist/${package_name}.tar.gz"
}

# Build Docker image
build_docker() {
    print_status "Building Docker image..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    local image_tag="abi-framework:${ENVIRONMENT}-latest"
    
    # Create Dockerfile if it doesn't exist
    if [ ! -f "Dockerfile" ]; then
        cat > Dockerfile << 'EOF'
FROM alpine:latest

# Install runtime dependencies
RUN apk add --no-cache libc6-compat

# Copy binaries
COPY zig-out/bin/ /usr/local/bin/

# Copy documentation and examples
COPY docs/ /usr/local/share/abi/docs/
COPY examples/ /usr/local/share/abi/examples/
COPY config/ /usr/local/share/abi/config/

# Create abi user
RUN adduser -D -s /bin/sh abi

# Switch to abi user
USER abi

# Set working directory
WORKDIR /home/abi

# Default command
CMD ["abi", "help"]
EOF
    fi
    
    # Build image
    if docker build -t "$image_tag" .; then
        print_success "Docker image built: $image_tag"
    else
        print_error "Docker build failed"
        exit 1
    fi
}

# Deploy based on environment
deploy() {
    print_status "Deploying to $ENVIRONMENT environment..."
    
    case $ENVIRONMENT in
        development)
            print_status "Development deployment - local installation"
            if [ -f "zig-out/bin/abi" ]; then
                print_success "Binary ready at: zig-out/bin/abi"
            fi
            ;;
        staging)
            print_status "Staging deployment - preparing for testing"
            if [ "$PACKAGE" = true ]; then
                create_package
            fi
            ;;
        production)
            print_status "Production deployment - creating release artifacts"
            if [ "$PACKAGE" = true ]; then
                create_package
            fi
            if [ "$DOCKER" = true ]; then
                build_docker
            fi
            ;;
    esac
}

# Main deployment function
main() {
    print_status "ABI Framework Deployment"
    print_status "========================"
    
    validate_environment
    build_project
    run_tests
    deploy
    
    print_success "Deployment completed successfully!"
    print_status ""
    print_status "Deployment Summary:"
    print_status "- Environment: $ENVIRONMENT"
    print_status "- Build Type: $BUILD_TYPE"
    print_status "- Target: $TARGET"
    print_status "- Package: $([ "$PACKAGE" = true ] && echo "Yes" || echo "No")"
    print_status "- Docker: $([ "$DOCKER" = true ] && echo "Yes" || echo "No")"
}

# Run main function
main "$@"