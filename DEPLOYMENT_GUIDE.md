# Abbey Deployment Guide

This guide covers deploying the ABI AI Agent System (Abbey) to production environments.

## Quick Start

```bash
# One-command deployment
./abbey-complete.sh all

# Or step by step
docker-compose up -d
curl http://localhost:8080/health
```

## Prerequisites

- Docker 20.10+ and Docker Compose v2
- (Optional) kubectl for Kubernetes deployment
- (Optional) Zig 0.13+ for local development

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

Key variables:
- `OPENAI_API_KEY` - OpenAI API key for GPT models
- `HF_API_TOKEN` - HuggingFace token for HF Inference API
- `GRAFANA_ADMIN_PASSWORD` - Grafana dashboard password

### Build Options

Configure features at build time:

```bash
# Build with specific features
docker build -t abbey:latest \
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
# Start core services (Abbey, Prometheus, Grafana, Jaeger)
docker-compose up -d

# View logs
docker-compose logs -f abbey

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
kubectl apply -k deploy/k8s/

# Check deployment status
kubectl get pods -n abbey

# View logs
kubectl logs -f -n abbey -l app.kubernetes.io/name=abbey
```

### Using the Deployment Script

```bash
./abbey-complete.sh k8s
```

### Customizing for Your Cluster

1. Update `deploy/k8s/kustomization.yaml` with your container registry:
   ```yaml
   images:
     - name: abbey
       newName: your-registry.io/abbey
       newTag: v1.0.0
   ```

2. Configure secrets:
   ```bash
   kubectl create secret generic abbey-secrets -n abbey \
       --from-literal=ABI_OPENAI_API_KEY=sk-your-key \
       --from-literal=ABI_HF_API_TOKEN=hf_your-token
   ```

3. Update ingress host in `deploy/k8s/ingress.yaml`

## Service Endpoints

| Service | Docker | Kubernetes | Description |
|---------|--------|------------|-------------|
| Abbey API | http://localhost:8080 | http://abbey.abbey.svc:8080 | Main API |
| Metrics | http://localhost:9090 | http://abbey.abbey.svc:9090 | Prometheus metrics |
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
    -d '{"message": "Hello, Abbey!"}'

# Get agent status
curl http://localhost:8080/api/status

# Prometheus metrics
curl http://localhost:9090/metrics
```

## Monitoring

### Grafana Dashboards

Access Grafana at http://localhost:3000 (default: admin/abbey_grafana)

Pre-configured dashboards:
- **Abbey Overview** - Request rates, latency, errors, AI backend metrics

### Prometheus Alerts

Configured alerts include:
- `AbbeyDown` - Service unreachable
- `AbbeyHighErrorRate` - Error rate > 5%
- `AbbeyHighLatency` - p95 latency > 1s
- `AbbeyAIBackendFailure` - AI backend errors

### Distributed Tracing

Access Jaeger at http://localhost:16686 to view distributed traces.

## Scaling

### Docker Compose

```bash
# Scale to 3 instances
docker-compose up -d --scale abbey=3
```

### Kubernetes

```bash
# Manual scaling
kubectl scale deployment/abbey --replicas=10 -n abbey

# HPA is configured (3-20 pods based on CPU/memory)
kubectl get hpa -n abbey
```

## Security Considerations

1. **Secrets Management**
   - Never commit API keys to version control
   - Use Kubernetes secrets or external secret managers
   - Rotate credentials regularly

2. **Network Policies**
   - `deploy/k8s/networkpolicy.yaml` implements zero-trust networking
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
docker-compose logs abbey

# Check resource constraints
docker stats abbey
```

### Health Check Failing

```bash
# Test health endpoint manually
curl -v http://localhost:8080/health

# Check application logs
docker-compose logs -f abbey | grep -i error
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
docker-compose exec abbey env | grep API

# Test connectivity to AI providers
docker-compose exec abbey curl -I https://api.openai.com/v1/models
```

## Deployment Script Reference

```bash
./abbey-complete.sh <command>

Commands:
  all         Complete deployment (build + start + health)
  build       Build the Abbey container
  start       Start all services
  stop        Stop all services
  status      Show service status and URLs
  health      Check if Abbey is healthy
  logs [svc]  View logs (default: abbey)
  test        Run deployment tests
  k8s         Deploy to Kubernetes
  clean       Stop and remove everything
  help        Show help message
```

## Build from Source

```bash
# Build release binary
zig build -Doptimize=ReleaseSafe \
    -Denable-ai=true \
    -Denable-gpu=false \
    -Denable-web=true \
    -Denable-database=true \
    -Denable-profiling=true

# Run tests
zig build test --summary all

# Run benchmarks
zig build benchmark
```
