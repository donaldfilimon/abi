---
title: "Deployment"
description: "Docker, Kubernetes, and production deployment"
section: "Operations"
order: 4
---

# Deployment

This guide covers deploying ABI framework applications in production using
Docker, Kubernetes, and bare-metal setups. It includes environment variable
configuration, feature tuning, monitoring integration, and an operational
checklist.

## Docker Compose

A minimal `docker-compose.yml` for running an ABI service with Prometheus and
Grafana:

```yaml
version: "3.8"

services:
  abi:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"   # ACP HTTP server
      - "9090:9090"   # Prometheus metrics endpoint
    environment:
      - ABI_MASTER_KEY=${ABI_MASTER_KEY}
      - ABI_OPENAI_API_KEY=${ABI_OPENAI_API_KEY}
      - ABI_ANTHROPIC_API_KEY=${ABI_ANTHROPIC_API_KEY}
      - ABI_OLLAMA_HOST=http://ollama:11434
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "4317:4317"    # OTLP gRPC
    restart: unless-stopped

volumes:
  ollama_data:
  grafana_data:
```

### Dockerfile

```dockerfile
FROM ghcr.io/ziglang/zig:master AS builder

WORKDIR /app
COPY . .

# Production build with optimized feature set
RUN zig build \
    -Doptimize=ReleaseFast \
    -Denable-mobile=false \
    -Denable-benchmarks=false

FROM debian:bookworm-slim
COPY --from=builder /app/zig-out/bin/abi /usr/local/bin/abi
EXPOSE 8080 9090
CMD ["abi", "acp", "serve", "--port", "8080"]
```

## Kubernetes with Kustomize

### Base Deployment

```yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: abi
  labels:
    app: abi
spec:
  replicas: 2
  selector:
    matchLabels:
      app: abi
  template:
    metadata:
      labels:
        app: abi
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: abi
          image: your-registry/abi:latest
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 9090
              name: metrics
          env:
            - name: ABI_MASTER_KEY
              valueFrom:
                secretKeyRef:
                  name: abi-secrets
                  key: master-key
            - name: ABI_OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: abi-secrets
                  key: openai-api-key
            - name: ABI_ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: abi-secrets
                  key: anthropic-api-key
            - name: ABI_OLLAMA_HOST
              value: "http://ollama-svc:11434"
          readinessProbe:
            httpGet:
              path: /.well-known/agent.json
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /.well-known/agent.json
              port: http
            initialDelaySeconds: 10
            periodSeconds: 30
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
```

### Service

```yaml
# k8s/base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: abi-svc
spec:
  selector:
    app: abi
  ports:
    - name: http
      port: 8080
      targetPort: http
    - name: metrics
      port: 9090
      targetPort: metrics
```

### Secrets

```yaml
# k8s/base/secrets.yaml (use sealed-secrets or external-secrets in production)
apiVersion: v1
kind: Secret
metadata:
  name: abi-secrets
type: Opaque
stringData:
  master-key: "${ABI_MASTER_KEY}"
  openai-api-key: "${ABI_OPENAI_API_KEY}"
  anthropic-api-key: "${ABI_ANTHROPIC_API_KEY}"
```

## Environment Variables

All environment variables used by the ABI framework:

| Variable | Purpose | Required |
|----------|---------|----------|
| `ABI_MASTER_KEY` | Secrets encryption key (production) | Yes (prod) |
| `ABI_OPENAI_API_KEY` | OpenAI API access | If using OpenAI |
| `ABI_ANTHROPIC_API_KEY` | Anthropic API access | If using Anthropic |
| `ABI_OLLAMA_HOST` | Ollama server URL (default: `http://127.0.0.1:11434`) | No |
| `ABI_OLLAMA_MODEL` | Default Ollama model | No |
| `ABI_HF_API_TOKEN` | HuggingFace API token | If using HuggingFace |
| `ABI_LM_STUDIO_HOST` | LM Studio server (default: `http://localhost:1234`) | No |
| `ABI_LM_STUDIO_MODEL` | Default LM Studio model | No |
| `ABI_VLLM_HOST` | vLLM server (default: `http://localhost:8000`) | No |
| `ABI_VLLM_MODEL` | Default vLLM model | No |
| `ABI_MLX_HOST` | MLX server (default: `http://localhost:8080`) | No |
| `ABI_MLX_MODEL` | Default MLX model | No |
| `DISCORD_BOT_TOKEN` | Discord bot integration | If using Discord |

## Feature Tuning for Production

Disable unused features to reduce binary size and attack surface:

```bash
# Minimal production build (API server only)
zig build -Doptimize=ReleaseFast \
    -Denable-mobile=false \
    -Denable-benchmarks=false \
    -Denable-gpu=false

# Full AI production build
zig build -Doptimize=ReleaseFast \
    -Denable-mobile=false \
    -Denable-benchmarks=false \
    -Denable-ai=true \
    -Denable-gpu=true \
    -Dgpu-backend=metal       # macOS
    # -Dgpu-backend=cuda      # NVIDIA
    # -Dgpu-backend=vulkan    # Cross-platform
```

Feature modules disabled at build time have zero binary overhead -- they are
replaced by stub implementations that the compiler eliminates entirely.

## Monitoring Stack

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "abi"
    static_configs:
      - targets: ["abi:9090"]
    metrics_path: "/metrics"
```

### Grafana Dashboard Queries

Key metrics to monitor:

| Metric | PromQL | Description |
|--------|--------|-------------|
| Request rate | `rate(http_requests_total[5m])` | Requests per second |
| Error rate | `rate(errors_total[5m]) / rate(http_requests_total[5m])` | Error percentage |
| Latency P99 | `histogram_quantile(0.99, rate(request_latency_ms_bucket[5m]))` | 99th percentile latency |
| Active connections | `active_connections` | Current connection count |
| Circuit breaker state | `circuit_breaker_state_transitions` | State transition count |

### Jaeger / OpenTelemetry

Configure the observability bundle to export traces:

```zig
var bundle = try abi.observability.ObservabilityBundle.init(allocator, .{
    .otel = .{
        .endpoint = "http://jaeger:4317",
        .service_name = "abi-production",
    },
    .prometheus = .{
        .port = 9090,
        .path = "/metrics",
    },
});
```

## Operational Checklist

Before deploying to production:

- [ ] Set `ABI_MASTER_KEY` to a strong, unique value for secrets encryption
- [ ] Store all API keys in a secrets manager (Kubernetes Secrets, HashiCorp Vault)
- [ ] Disable unused features (`-Denable-mobile=false`, `-Denable-benchmarks=false`)
- [ ] Build with `-Doptimize=ReleaseFast` for production performance
- [ ] Configure rate limiting in the auth module (`abi.auth.rate_limit`)
- [ ] Enable Prometheus metrics export via the observability bundle
- [ ] Set up health check probes (`/.well-known/agent.json` for ACP)
- [ ] Configure resource limits in Kubernetes
- [ ] Enable TLS termination (load balancer or `abi.auth.tls`)
- [ ] Set up alerting rules for error rate and latency thresholds
- [ ] Verify with `zig build full-check` before building the container image

## Related

- [Auth & Security](auth.html) -- Secrets management, TLS, rate limiting
- [Observability](observability.html) -- Metrics, tracing, and alerting configuration
- [Connectors](connectors.html) -- LLM provider environment variables
- [MCP Server](mcp.html) -- MCP stdio server for AI client integration
- [ACP Protocol](acp.html) -- HTTP-based agent communication
