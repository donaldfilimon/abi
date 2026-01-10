#!/bin/bash
# =============================================================================
# Abbey Complete Deployment Script
# One-command deployment for the ABI AI Agent System
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
    ___    ____  ____  _______  __
   /   |  / __ )/ __ )/ ____\ \/ /
  / /| | / __  / __  / __/   \  /
 / ___ |/ /_/ / /_/ / /___   / /
/_/  |_/_____/_____/_____/  /_/

AI Agent System - Production Deployment
EOF
    echo -e "${NC}"
}

# Print step
step() {
    echo -e "\n${GREEN}==>${NC} ${BLUE}$1${NC}"
}

# Print warning
warn() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

# Print error
error() {
    echo -e "${RED}Error:${NC} $1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    step "Checking prerequisites..."

    local missing=()

    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing+=("docker-compose")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        error "Missing required tools: ${missing[*]}"
    fi

    echo "  All prerequisites satisfied"
}

# Build the application
build() {
    step "Building Abbey container..."

    docker build -t abbey:latest \
        --build-arg ENABLE_AI=true \
        --build-arg ENABLE_GPU=false \
        --build-arg ENABLE_WEB=true \
        --build-arg ENABLE_DATABASE=true \
        --build-arg ENABLE_PROFILING=true \
        .

    echo "  Build complete"
}

# Start services with Docker Compose
start() {
    step "Starting Abbey services..."

    # Check if docker-compose or docker compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    $COMPOSE_CMD up -d

    echo "  Services started"
}

# Stop services
stop() {
    step "Stopping Abbey services..."

    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    $COMPOSE_CMD down

    echo "  Services stopped"
}

# Check service health
health() {
    step "Checking service health..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            echo -e "  ${GREEN}Abbey is healthy!${NC}"
            return 0
        fi
        echo "  Waiting for Abbey to be ready... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done

    warn "Abbey did not become healthy within expected time"
    return 1
}

# Show service status
status() {
    step "Service Status"

    echo ""
    echo "  Service URLs:"
    echo "  ============================================"
    echo "  Abbey API:      http://localhost:8080"
    echo "  Chat Endpoint:  http://localhost:8080/api/chat"
    echo "  Metrics:        http://localhost:9090/metrics"
    echo "  Prometheus:     http://localhost:9091"
    echo "  Grafana:        http://localhost:3000"
    echo "  Jaeger:         http://localhost:16686"
    echo ""

    if command -v docker-compose &> /dev/null; then
        docker-compose ps
    else
        docker compose ps
    fi
}

# View logs
logs() {
    step "Viewing logs..."

    if command -v docker-compose &> /dev/null; then
        docker-compose logs -f "${1:-abbey}"
    else
        docker compose logs -f "${1:-abbey}"
    fi
}

# Deploy to Kubernetes
deploy_k8s() {
    step "Deploying to Kubernetes..."

    if ! command -v kubectl &> /dev/null; then
        error "kubectl is required for Kubernetes deployment"
    fi

    # Create namespace
    kubectl apply -f deploy/k8s/namespace.yaml

    # Apply all resources
    kubectl apply -k deploy/k8s/

    echo "  Kubernetes deployment complete"
    echo ""
    echo "  Check status with: kubectl get pods -n abbey"
    echo "  View logs with: kubectl logs -f -n abbey -l app.kubernetes.io/name=abbey"
}

# Clean up everything
clean() {
    step "Cleaning up..."

    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi

    $COMPOSE_CMD down -v --remove-orphans
    docker image rm abbey:latest 2>/dev/null || true

    echo "  Cleanup complete"
}

# Run a quick test
test_deployment() {
    step "Testing deployment..."

    echo "  Testing health endpoint..."
    if curl -s http://localhost:8080/health | grep -q "ok\|healthy"; then
        echo -e "  ${GREEN}Health check passed${NC}"
    else
        warn "Health check returned unexpected response"
    fi

    echo ""
    echo "  Testing metrics endpoint..."
    if curl -s http://localhost:9090/metrics | head -5; then
        echo -e "  ${GREEN}Metrics endpoint working${NC}"
    else
        warn "Metrics endpoint not responding"
    fi

    echo ""
    echo "  Testing chat endpoint..."
    response=$(curl -s -X POST http://localhost:8080/api/chat \
        -H "Content-Type: application/json" \
        -d '{"message":"Hello, Abbey!"}' 2>/dev/null || echo "error")

    if [ "$response" != "error" ]; then
        echo -e "  ${GREEN}Chat endpoint working${NC}"
        echo "  Response: $response"
    else
        warn "Chat endpoint not responding"
    fi
}

# Complete deployment (build + start + health check)
all() {
    print_banner
    check_prerequisites
    build
    start

    echo ""
    step "Waiting for services to start..."
    sleep 5

    health
    status

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Abbey deployment complete!           ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

# Print usage
usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  all         Complete deployment (build + start + health)"
    echo "  build       Build the Abbey container"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  status      Show service status and URLs"
    echo "  health      Check if Abbey is healthy"
    echo "  logs [svc]  View logs (default: abbey)"
    echo "  test        Run deployment tests"
    echo "  k8s         Deploy to Kubernetes"
    echo "  clean       Stop and remove everything"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 all          # Full deployment"
    echo "  $0 start        # Start services (if already built)"
    echo "  $0 logs jaeger  # View Jaeger logs"
    echo ""
}

# Main entrypoint
case "${1:-help}" in
    all)
        all
        ;;
    build)
        build
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    health)
        health
        ;;
    logs)
        logs "$2"
        ;;
    test)
        test_deployment
        ;;
    k8s)
        deploy_k8s
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        print_banner
        usage
        ;;
    *)
        error "Unknown command: $1"
        ;;
esac
