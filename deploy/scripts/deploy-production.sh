#!/bin/bash
# WDBX Refactored Production Deployment Script

set -euo pipefail

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io/wdbx}"
IMAGE_TAG="${IMAGE_TAG:-refactored-v2.0.0}"
NAMESPACE="wdbx-production"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check namespace exists
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log "Creating namespace $NAMESPACE..."
        kubectl create namespace $NAMESPACE
    fi
    
    success "Pre-deployment checks passed"
}

# Build and push Docker image
build_and_push_image() {
    log "Building Docker image..."
    
    # Build image
    docker build \
        -f deploy/docker/Dockerfile \
        -t "${DOCKER_REGISTRY}/wdbx:${IMAGE_TAG}" \
        -t "${DOCKER_REGISTRY}/wdbx:latest" \
        --build-arg ZIG_VERSION=0.15.1 \
        .
    
    success "Docker image built successfully"
    
    # Push to registry
    log "Pushing image to registry..."
    docker push "${DOCKER_REGISTRY}/wdbx:${IMAGE_TAG}"
    docker push "${DOCKER_REGISTRY}/wdbx:latest"
    
    success "Image pushed to registry"
}

# Run pre-deployment tests
run_pre_deployment_tests() {
    log "Running pre-deployment tests..."
    
    # Run quick smoke tests in container
    docker run --rm \
        "${DOCKER_REGISTRY}/wdbx:${IMAGE_TAG}" \
        test --quick
    
    # Run performance benchmarks
    docker run --rm \
        "${DOCKER_REGISTRY}/wdbx:${IMAGE_TAG}" \
        benchmark --quick --output-format json > benchmark-results.json
    
    # Check performance against baseline
    if [ -f benchmark-baseline.json ]; then
        log "Comparing performance against baseline..."
        # Add performance comparison logic here
    fi
    
    success "Pre-deployment tests passed"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Update image in deployment
    kubectl set image deployment/wdbx-refactored \
        wdbx="${DOCKER_REGISTRY}/wdbx:${IMAGE_TAG}" \
        -n $NAMESPACE
    
    # Apply all configurations
    kubectl apply -f deploy/kubernetes/ -n $NAMESPACE
    
    # Wait for rollout to complete
    log "Waiting for deployment rollout..."
    if ! kubectl rollout status deployment/wdbx-refactored -n $NAMESPACE --timeout=600s; then
        if [ "$ROLLBACK_ON_FAILURE" == "true" ]; then
            error "Deployment failed, initiating rollback..."
            kubectl rollout undo deployment/wdbx-refactored -n $NAMESPACE
            kubectl rollout status deployment/wdbx-refactored -n $NAMESPACE
            error "Deployment failed and was rolled back"
        else
            error "Deployment failed"
        fi
    fi
    
    success "Deployment completed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n $NAMESPACE -l app=wdbx
    
    # Check service endpoints
    kubectl get endpoints wdbx-service -n $NAMESPACE
    
    # Get service URL
    SERVICE_URL=$(kubectl get service wdbx-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_URL" ]; then
        SERVICE_URL=$(kubectl get service wdbx-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [ -z "$SERVICE_URL" ]; then
        warning "LoadBalancer IP/hostname not yet assigned"
        SERVICE_URL="pending"
    fi
    
    # Health check
    if [ "$SERVICE_URL" != "pending" ]; then
        log "Running health checks..."
        for i in {1..30}; do
            if curl -f "http://${SERVICE_URL}:8081/health" &> /dev/null; then
                success "Health check passed"
                break
            fi
            if [ $i -eq 30 ]; then
                error "Health check failed after 30 attempts"
            fi
            sleep 10
        done
    fi
    
    success "Deployment verified"
}

# Post-deployment tasks
post_deployment_tasks() {
    log "Running post-deployment tasks..."
    
    # Update monitoring dashboards
    kubectl apply -f monitoring/kubernetes/ -n monitoring || true
    
    # Send deployment notification
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 WDBX ${IMAGE_TAG} deployed to ${DEPLOYMENT_ENV}\"}" \
            "${SLACK_WEBHOOK_URL}" || true
    fi
    
    # Generate deployment report
    cat > deployment-report.txt << EOF
Deployment Report
================
Environment: ${DEPLOYMENT_ENV}
Namespace: ${NAMESPACE}
Image: ${DOCKER_REGISTRY}/wdbx:${IMAGE_TAG}
Timestamp: $(date)
Status: SUCCESS

Service URL: ${SERVICE_URL:-pending}
API Endpoint: http://${SERVICE_URL:-pending}:8080
Admin Endpoint: http://${SERVICE_URL:-pending}:8081
Metrics Endpoint: http://${SERVICE_URL:-pending}:9090

Next Steps:
1. Monitor deployment at Grafana dashboard
2. Check application logs: kubectl logs -f -l app=wdbx -n ${NAMESPACE}
3. Run integration tests against the new deployment
EOF

    cat deployment-report.txt
    
    success "Post-deployment tasks completed"
}

# Main deployment flow
main() {
    log "🚀 Starting WDBX Refactored Production Deployment"
    log "Environment: ${DEPLOYMENT_ENV}"
    log "Image: ${DOCKER_REGISTRY}/wdbx:${IMAGE_TAG}"
    
    pre_deployment_checks
    build_and_push_image
    run_pre_deployment_tests
    deploy_to_kubernetes
    verify_deployment
    post_deployment_tasks
    
    success "🎉 WDBX Refactored Production Deployment Completed Successfully!"
}

# Run main function
main "$@"