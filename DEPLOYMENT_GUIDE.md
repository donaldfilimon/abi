---
title: "DEPLOYMENT_GUIDE"
tags: []
---
# ABI Deployment Guide
> **Codebase Status:** Synced with repository as of 2026-01-24.

This guide covers deploying the ABI AI Agent System to production environments.

## Quick Start

```bash
# Build release binary
zig build -Doptimize=ReleaseFast

# Or use Docker
docker-compose up -d
curl http://localhost:8080/health
```

## Prerequisites

- Docker 20.10+ and Docker Compose v2
- (Optional) kubectl for Kubernetes deployment
- (Optional) Zig 0.16.x for local development

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

Key variables:
- `ABI_OPENAI_API_KEY` - OpenAI API key for GPT models
- `ABI_HF_API_TOKEN` - HuggingFace token for HF Inference API
- `ABI_OLLAMA_HOST` - Ollama host for local LLM (default: `http://127.0.0.1:11434`)
- `GRAFANA_ADMIN_PASSWORD` - Grafana dashboard password

### Build Options

Configure features at build time:

```bash
# Build with specific features
zig build -Doptimize=ReleaseFast \
    -Denable-ai=true \
    -Denable-gpu=false \
    -Denable-web=true \
    -Denable-database=true \
    -Denable-profiling=true

# Docker build with features
docker build -t abi:latest \
    --build-arg ENABLE_AI=true \
    --build-arg ENABLE_GPU=false \
    --build-arg ENABLE_WEB=true \
    --build-arg ENABLE_DATABASE=true \
    --build-arg ENABLE_PROFILING=true \
    .
```

## Docker Compose Deployment

### Basic Deployment

```bash
# Start core services (ABI, Prometheus, Grafana, Jaeger)
docker-compose up -d

# View logs
docker-compose logs -f abi

# Stop services
docker-compose down
```

### With Optional Services

```bash
# Include Ollama for local LLM
docker-compose --profile with-ollama up -d

# Include Neo4j graph database
docker-compose --profile with-neo4j up -d

# Include Redis caching
docker-compose --profile with-redis up -d

# All optional services
docker-compose --profile with-ollama --profile with-neo4j --profile with-redis up -d
```

## Kubernetes Deployment

### Using Kustomize

```bash
# Create namespace and deploy
kubectl apply -k k8s/

# Check deployment status
kubectl get pods -n abi

# View logs
kubectl logs -f -n abi -l app.kubernetes.io/name=abi
```

### Customizing for Your Cluster

1. Update `k8s/kustomization.yaml` with your container registry:
   ```yaml
   images:
     - name: abi
       newName: your-registry.io/abi
       newTag: v1.0.0
   ```

2. Configure secrets:
   ```bash
   kubectl create secret generic abi-secrets -n abi \
       --from-literal=ABI_OPENAI_API_KEY=sk-your-key \
       --from-literal=ABI_HF_API_TOKEN=hf_your-token
   ```

3. Update ingress host in `k8s/ingress.yaml`

## Service Endpoints

| Service | Docker | Kubernetes | Description |
|---------|--------|------------|-------------|
| ABI API | http://localhost:8080 | http://abi.abi.svc:8080 | Main API |
| Metrics | http://localhost:9090 | http://abi.abi.svc:9090 | Prometheus metrics |
| Prometheus | http://localhost:9091 | - | Metrics database |
| Grafana | http://localhost:3000 | - | Dashboards |
| Jaeger | http://localhost:16686 | - | Distributed tracing |

## API Endpoints

```bash
# Health check
curl http://localhost:8080/health

# Readiness probe
curl http://localhost:8080/ready

# Chat with AI
curl -X POST http://localhost:8080/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello, ABI!"}'

# Get agent status
curl http://localhost:8080/api/status

# Prometheus metrics
curl http://localhost:9090/metrics
```

## Monitoring

### Grafana Dashboards

Access Grafana at http://localhost:3000 (default: admin/abi_grafana)

Pre-configured dashboards:
- **ABI Overview** - Request rates, latency, errors, AI backend metrics

### Prometheus Alerts

Configured alerts include:
- `ABIDown` - Service unreachable
- `ABIHighErrorRate` - Error rate > 5%
- `ABIHighLatency` - p95 latency > 1s
- `ABIAIBackendFailure` - AI backend errors

### Distributed Tracing

Access Jaeger at http://localhost:16686 to view distributed traces.

## Scaling

### Docker Compose

```bash
# Scale to 3 instances
docker-compose up -d --scale abi=3
```

### Kubernetes

```bash
# Manual scaling
kubectl scale deployment/abi --replicas=10 -n abi

# HPA is configured (3-20 pods based on CPU/memory)
kubectl get hpa -n abi
```

## Security Considerations

1. **Secrets Management**
   - Never commit API keys to version control
   - Use Kubernetes secrets or external secret managers
   - Rotate credentials regularly

2. **Network Policies**
   - `k8s/networkpolicy.yaml` implements zero-trust networking
   - Only required traffic is allowed

3. **Container Security**
   - Non-root user (uid 1000)
   - Read-only root filesystem
   - Dropped capabilities

4. **TLS/HTTPS**
   - Configure TLS in ingress for production
   - Use cert-manager for automatic certificate management

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs abi

# Check resource constraints
docker stats abi
```

### Health Check Failing

```bash
# Test health endpoint manually
curl -v http://localhost:8080/health

# Check application logs
docker-compose logs -f abi | grep -i error
```

### High Memory Usage

```bash
# Check metrics
curl http://localhost:9090/metrics | grep memory

# Adjust limits in docker-compose.yaml or deployment.yaml
```

### AI Backend Errors

```bash
# Verify API keys are set
docker-compose exec abi env | grep API

# Test connectivity to AI providers
docker-compose exec abi curl -I https://api.openai.com/v1/models
```

## Build from Source

```bash
# Build release binary
zig build -Doptimize=ReleaseFast \
    -Denable-ai=true \
    -Denable-gpu=false \
    -Denable-web=true \
    -Denable-database=true \
    -Denable-profiling=true

# Run tests
zig build test --summary all

# Run benchmarks
zig build benchmarks
```



