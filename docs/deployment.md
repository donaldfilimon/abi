# Production Deployment Guide

This guide covers deploying ABI in production environments.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Zig | 0.16.x | Required |
| Git | Any | For cloning |
| GPU Drivers | Latest | Optional, for GPU acceleration |
| Docker | 20.10+ | Optional, for containerized deployment |

## Building for Production

```bash
# Release build with all features
zig build -Doptimize=ReleaseFast \
  -Denable-ai=true \
  -Denable-gpu=true \
  -Denable-database=true \
  -Denable-profiling=true

# Minimal build (AI only)
zig build -Doptimize=ReleaseFast \
  -Denable-ai=true \
  -Denable-gpu=false \
  -Denable-database=false
```

## Environment Configuration

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ABI_MASTER_KEY` | 32+ byte encryption key | `$(openssl rand -hex 32)` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ABI_OPENAI_API_KEY` | - | OpenAI API key |
| `ABI_ANTHROPIC_API_KEY` | - | Anthropic/Claude API key |
| `ABI_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `ABI_HF_API_TOKEN` | - | HuggingFace token |

### Generating a Master Key

```bash
# Generate a secure 32-byte key
export ABI_MASTER_KEY=$(openssl rand -hex 32)

# Or use a password-derived key
export ABI_MASTER_KEY=$(echo -n "your-secure-password" | sha256sum | cut -c1-64)
```

## Security Checklist

Before deploying to production:

- [ ] `ABI_MASTER_KEY` environment variable is set
- [ ] Secrets manager initialized with `require_master_key: true`
- [ ] JWT `allow_none_algorithm` is `false` (default)
- [ ] Rate limiting enabled for public endpoints
- [ ] TLS enabled for all external connections
- [ ] API keys stored securely (not in code or logs)
- [ ] Review [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for known issues

## Docker Deployment

### Using Docker Compose

```bash
# Standard deployment
docker compose up -d abi

# GPU-enabled deployment (requires NVIDIA Container Toolkit)
docker compose up -d abi-gpu

# With Ollama for local LLM inference
docker compose --profile ollama up -d
```

### Manual Docker Build

```bash
# Build the image
docker build -t abi:latest .

# Run with environment variables
docker run -d \
  --name abi \
  -e ABI_MASTER_KEY="your-32-byte-key" \
  -e ABI_OPENAI_API_KEY="sk-..." \
  -p 8080:8080 \
  abi:latest
```

## Health Checks

### HTTP Health Endpoint

The streaming server exposes a health endpoint:

```bash
curl http://localhost:8080/health
```

Expected response: `200 OK` with JSON status.

### Process Health

```bash
# Check if running
pgrep -f abi

# Check memory usage
ps aux | grep abi
```

## Streaming Server

### Starting the Server

```bash
# Start with local model
zig build run -- llm serve -m ./models/llama-7b.gguf --preload

# With authentication
zig build run -- llm serve -m ./model.gguf -a 0.0.0.0:8000 --auth-token $AUTH_TOKEN
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (SSE) |
| `/api/stream` | POST | Custom ABI streaming (SSE) |
| `/api/stream/ws` | GET | WebSocket streaming |
| `/admin/reload` | POST | Hot-reload model |
| `/health` | GET | Health check |

### Circuit Breaker Behavior

When a backend fails repeatedly:
1. Circuit breaker opens after `failure_threshold` failures (default: 5)
2. Server returns `503 Service Unavailable` with `Retry-After` header
3. After `timeout_ms` (default: 60s), circuit moves to half-open
4. Successful requests close the circuit

## Monitoring

### Enable Profiling

Build with profiling enabled:

```bash
zig build -Denable-profiling=true
```

### Metrics

Access metrics via the observability module:
- Request counts
- Latency histograms
- Error rates
- Circuit breaker states

### Logging

ABI uses Zig's standard logging. Set log level via:

```bash
# In code
std.log.default_level = .info;
```

## Graceful Shutdown

Send `SIGTERM` to initiate graceful shutdown:

```bash
kill -TERM $(pgrep -f abi)
```

The server will:
1. Stop accepting new connections
2. Wait for active streams to complete (30s timeout)
3. Close database connections
4. Release GPU resources
5. Exit

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "MasterKeyRequired" error | Set `ABI_MASTER_KEY` environment variable |
| Circuit breaker always open | Check backend connectivity, increase `failure_threshold` |
| High memory usage | Enable memory profiling, check for leaks |
| Slow startup | Use `--preload` to load models at startup |

### Debug Build

For troubleshooting, build with debug symbols:

```bash
zig build -Doptimize=Debug
gdb ./zig-out/bin/abi
```

### Logs

Check logs for errors:
- JWT warnings indicate `allow_none_algorithm` is enabled
- Master key warnings indicate random key generation

## Performance Tuning

### GPU Backend Selection

```bash
# Auto-detect best backend
zig build -Dgpu-backend=auto

# Specific backend
zig build -Dgpu-backend=cuda
zig build -Dgpu-backend=vulkan
zig build -Dgpu-backend=metal
```

### Memory Settings

For large models, increase system limits:

```bash
# Increase open file limit
ulimit -n 65536

# Increase memory lock limit (for mlock)
ulimit -l unlimited
```

## Backup and Recovery

### Database Backup

```bash
zig build run -- db backup --path /backup/db-$(date +%Y%m%d).db
```

### Restore

```bash
zig build run -- db restore --path /backup/db-20260130.db
```

## Further Reading

- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [docs/streaming.md](streaming.md) - Streaming API details
- [docs/gpu.md](gpu.md) - GPU configuration
- [docs/troubleshooting.md](troubleshooting.md) - Common issues
