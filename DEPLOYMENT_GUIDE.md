# Deployment Guide

This repository ships deployment assets under `deploy/` for Docker, Kubernetes,
monitoring, and nginx. The assets are intentionally minimal and should be
reviewed for your environment.

## Build a Release Binary
```bash
zig build -Doptimize=ReleaseFast
```

## Configure Features
```bash
zig build -Denable-ai=true -Denable-gpu=false -Denable-web=true -Denable-database=true
```

## Deploy Assets
- `deploy/docker` - Compose definitions
- `deploy/kubernetes` - Base manifests
- `deploy/monitoring` - Prometheus/Grafana templates
- `deploy/nginx` - Reverse proxy templates

## Additional Details
- Ensure secrets and API keys are provided via your platform secrets manager.
- Validate CPU/GPU availability before enabling acceleration.
