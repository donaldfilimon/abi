# WDBX Production Deployment Guide

## 🚀 Production-Ready Vector Database for High-Performance Workloads

This guide covers deploying and operating WDBX in production environments handling millions of vectors with enterprise-grade reliability, efficiency, and observability.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Production Configuration](#production-configuration)
3. [Deployment Strategies](#deployment-strategies)
4. [Monitoring & Observability](#monitoring--observability)
5. [Performance Tuning](#performance-tuning)
6. [Scaling Strategies](#scaling-strategies)
7. [Disaster Recovery](#disaster-recovery)
8. [Security Best Practices](#security-best-practices)

## System Requirements

### Hardware Requirements

#### Minimum (1M vectors)
- **CPU**: 8 cores (x86_64 with AVX2 or ARM64 with NEON)
- **RAM**: 32GB
- **Storage**: 500GB NVMe SSD
- **Network**: 10 Gbps

#### Recommended (10M vectors)
- **CPU**: 32 cores
- **RAM**: 128GB
- **Storage**: 2TB NVMe SSD RAID 10
- **Network**: 25 Gbps
- **GPU**: Optional (NVIDIA A100 for acceleration)

#### Enterprise (100M+ vectors)
- **CPU**: 64+ cores across multiple nodes
- **RAM**: 512GB+ per node
- **Storage**: Distributed storage (Ceph, GlusterFS)
- **Network**: 100 Gbps InfiniBand
- **GPU**: Multiple GPUs for parallel processing

### Software Requirements

- **OS**: Linux (Ubuntu 22.04 LTS, RHEL 9, or similar)
- **Zig**: 0.14.1 or later
- **Monitoring**: Prometheus 2.40+, Grafana 9.0+
- **Container**: Docker 24.0+ or Kubernetes 1.28+

## Production Configuration

### 1. Optimized Build

```bash
# Build with maximum optimizations
zig build -Doptimize=ReleaseFast \
    -Dcpu=native \
    -Denable_simd=true \
    -Denable_gpu=true \
    -Dstrip=true

# Build for specific architecture
zig build -Doptimize=ReleaseFast \
    -Dtarget=x86_64-linux-gnu \
    -Dcpu=znver3  # AMD Zen 3
```

### 2. Configuration File

Create `wdbx_production.yaml`:

```yaml
# Scalability Configuration
scalability:
  shard_count: 16
  max_vectors_per_shard: 1000000
  auto_rebalance: true
  rebalance_threshold: 0.2

# Reliability Configuration
reliability:
  checkpoint_interval_ms: 60000
  health_check_interval_ms: 5000
  recovery_retry_attempts: 5
  enable_auto_recovery: true
  enable_wal: true

# Efficiency Configuration
efficiency:
  l1_cache_size_mb: 256
  l2_cache_size_mb: 1024
  l3_cache_size_mb: 4096
  compression_type: quantization_8bit
  compression_batch_size: 1000

# Observability Configuration
observability:
  enable_metrics: true
  metrics_export_interval_ms: 10000
  enable_tracing: true
  log_level: info
  prometheus_port: 9090
  jaeger_endpoint: http://localhost:14268/api/traces

# Performance Configuration
performance:
  max_concurrent_operations: 10000
  thread_pool_size: 0  # Auto-detect
  enable_simd: true
  enable_gpu: false
  prefetch_size: 100
  batch_size: 1000

# Persistence Configuration
persistence:
  data_dir: /var/lib/wdbx/data
  backup_dir: /var/lib/wdbx/backups
  backup_retention_days: 30
  snapshot_interval_hours: 6
```

### 3. Environment Variables

```bash
# Required
export WDBX_DATA_DIR=/var/lib/wdbx/data
export WDBX_CONFIG=/etc/wdbx/production.yaml
export WDBX_LOG_LEVEL=info

# Optional
export WDBX_METRICS_PORT=9090
export WDBX_API_PORT=8080
export WDBX_ADMIN_PORT=8081
export WDBX_MAX_MEMORY=128GB
export WDBX_NUMA_NODES=0,1
```

## Deployment Strategies

### 1. Bare Metal Deployment

```bash
#!/bin/bash
# deploy_baremetal.sh

# Set up directories
sudo mkdir -p /var/lib/wdbx/{data,backups,logs}
sudo mkdir -p /etc/wdbx

# Copy configuration
sudo cp wdbx_production.yaml /etc/wdbx/

# Set up systemd service
cat << EOF | sudo tee /etc/systemd/system/wdbx.service
[Unit]
Description=WDBX Production Vector Database
After=network.target

[Service]
Type=simple
User=wdbx
Group=wdbx
ExecStart=/usr/local/bin/wdbx_production
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
LimitNOFILE=1048576
LimitNPROC=512000

# Performance tuning
CPUAccounting=true
MemoryAccounting=true
TasksAccounting=true
IOAccounting=true
MemoryMax=128G
CPUQuota=800%

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable wdbx
sudo systemctl start wdbx
```

### 2. Docker Deployment

```dockerfile
# Dockerfile.production
FROM alpine:3.19 AS builder

RUN apk add --no-cache zig

WORKDIR /build
COPY . .

RUN zig build -Doptimize=ReleaseFast \
    -Dtarget=x86_64-linux-musl \
    -Denable_simd=true

FROM alpine:3.19

RUN apk add --no-cache \
    ca-certificates \
    numactl \
    hwloc

COPY --from=builder /build/zig-out/bin/wdbx_production /usr/local/bin/
COPY wdbx_production.yaml /etc/wdbx/

EXPOSE 8080 8081 9090

VOLUME ["/var/lib/wdbx/data", "/var/lib/wdbx/backups"]

ENTRYPOINT ["numactl", "--interleave=all", "/usr/local/bin/wdbx_production"]
```

### 3. Kubernetes Deployment

```yaml
# wdbx-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: wdbx
spec:
  serviceName: wdbx
  replicas: 3
  selector:
    matchLabels:
      app: wdbx
  template:
    metadata:
      labels:
        app: wdbx
    spec:
      containers:
      - name: wdbx
        image: wdbx:production
        resources:
          requests:
            memory: "64Gi"
            cpu: "8"
          limits:
            memory: "128Gi"
            cpu: "16"
        env:
        - name: WDBX_SHARD_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        volumeMounts:
        - name: data
          mountPath: /var/lib/wdbx/data
        - name: config
          mountPath: /etc/wdbx
        ports:
        - containerPort: 8080
          name: api
        - containerPort: 9090
          name: metrics
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: wdbx-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-nvme
      resources:
        requests:
          storage: 500Gi
```

## Monitoring & Observability

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'wdbx'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    
  - job_name: 'wdbx_cluster'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - wdbx
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: wdbx
```

### 2. Key Metrics to Monitor

```promql
# Query performance
rate(wdbx_operations_total[5m])
histogram_quantile(0.99, rate(wdbx_latency_bucket[5m]))

# Resource utilization
wdbx_memory_bytes / (128 * 1024 * 1024 * 1024) * 100  # Memory %
rate(wdbx_cpu_seconds_total[5m]) * 100  # CPU %

# Cache performance
wdbx_cache_hit_rate
rate(wdbx_cache_evictions_total[5m])

# Health status
wdbx_health_score
up{job="wdbx"}

# Shard distribution
stddev(wdbx_shard_vector_count) / avg(wdbx_shard_vector_count)
```

### 3. Grafana Dashboard

Import the production dashboard:

```json
{
  "dashboard": {
    "title": "WDBX Production Monitoring",
    "panels": [
      {
        "title": "Operations per Second",
        "targets": [
          {
            "expr": "rate(wdbx_operations_total[5m])"
          }
        ]
      },
      {
        "title": "Latency (p50, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(wdbx_latency_bucket[5m]))"
          },
          {
            "expr": "histogram_quantile(0.99, rate(wdbx_latency_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "wdbx_memory_bytes"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "wdbx_cache_hit_rate"
          }
        ]
      }
    ]
  }
}
```

### 4. Alerting Rules

```yaml
# alerts.yml
groups:
  - name: wdbx
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(wdbx_latency_bucket[5m])) > 100
        for: 5m
        annotations:
          summary: "High query latency detected"
          
      - alert: LowCacheHitRate
        expr: wdbx_cache_hit_rate < 0.8
        for: 10m
        annotations:
          summary: "Cache hit rate below 80%"
          
      - alert: HighMemoryUsage
        expr: wdbx_memory_bytes / (128 * 1024 * 1024 * 1024) > 0.9
        for: 5m
        annotations:
          summary: "Memory usage above 90%"
          
      - alert: UnhealthyStatus
        expr: wdbx_health_score < 0.8
        for: 2m
        annotations:
          summary: "Database health score below threshold"
```

## Performance Tuning

### 1. System Tuning

```bash
#!/bin/bash
# tune_system.sh

# Disable swap
sudo swapoff -a

# Set swappiness
echo 1 | sudo tee /proc/sys/vm/swappiness

# Increase file descriptors
echo "* soft nofile 1048576" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 1048576" | sudo tee -a /etc/security/limits.conf

# Network tuning
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"

# NUMA optimization
sudo numactl --hardware
sudo numactl --interleave=all

# Huge pages
echo 4096 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# CPU frequency scaling
sudo cpupower frequency-set -g performance

# Disable CPU throttling
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

### 2. WDBX Tuning Parameters

```zig
// Performance tuning configuration
const tuning = .{
    // Memory
    .arena_size_mb = 1024,
    .stack_size_mb = 16,
    .allocation_batch_size = 4096,
    
    // Threading
    .worker_threads = std.Thread.getCpuCount(),
    .io_threads = 4,
    .background_threads = 2,
    
    // Caching
    .l1_cache_line_size = 64,
    .prefetch_distance = 8,
    .cache_associativity = 16,
    
    // SIMD
    .vector_width = 8,  // AVX2
    .unroll_factor = 4,
    
    // I/O
    .direct_io = true,
    .mmap_threshold_mb = 128,
    .read_ahead_kb = 2048,
};
```

### 3. Query Optimization

```zig
// Optimized search with batching
pub fn searchBatch(queries: [][]f32, k: usize) ![][]SearchResult {
    // Prefetch data
    for (queries) |query| {
        @prefetch(query.ptr, .{ .rw = .read, .locality = 3 });
    }
    
    // Process in parallel
    const results = try allocator.alloc([]SearchResult, queries.len);
    
    var wg = std.Thread.WaitGroup{};
    for (queries, 0..) |query, i| {
        wg.spawnManager(searchWorker, .{ query, k, &results[i] });
    }
    wg.wait();
    
    return results;
}
```

## Scaling Strategies

### 1. Vertical Scaling

```yaml
# Scale up resources
resources:
  requests:
    memory: "256Gi"
    cpu: "32"
  limits:
    memory: "512Gi"
    cpu: "64"
```

### 2. Horizontal Scaling

```bash
# Scale out replicas
kubectl scale statefulset wdbx --replicas=6

# Auto-scaling
kubectl autoscale statefulset wdbx \
    --min=3 --max=10 \
    --cpu-percent=70
```

### 3. Sharding Strategy

```zig
// Dynamic sharding based on load
const sharding = .{
    .strategy = .consistent_hash,
    .replication_factor = 3,
    .virtual_nodes = 150,
    .rebalance_threshold = 0.2,
    .migration_batch_size = 10000,
};
```

## Disaster Recovery

### 1. Backup Strategy

```bash
#!/bin/bash
# backup.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/lib/wdbx/backups/${TIMESTAMP}"

# Create backup
wdbx_cli backup create --output="${BACKUP_DIR}"

# Compress
tar -czf "${BACKUP_DIR}.tar.gz" "${BACKUP_DIR}"

# Upload to S3
aws s3 cp "${BACKUP_DIR}.tar.gz" s3://wdbx-backups/

# Clean old backups
find /var/lib/wdbx/backups -type f -mtime +30 -delete
```

### 2. Point-in-Time Recovery

```bash
# Enable WAL
export WDBX_ENABLE_WAL=true
export WDBX_WAL_DIR=/var/lib/wdbx/wal

# Restore to specific point
wdbx_cli restore --timestamp="2024-01-15T10:30:00Z" \
    --wal-dir=/var/lib/wdbx/wal \
    --data-dir=/var/lib/wdbx/data
```

### 3. Multi-Region Replication

```yaml
# Cross-region replication
replication:
  primary_region: us-east-1
  replica_regions:
    - us-west-2
    - eu-west-1
    - ap-southeast-1
  replication_lag_ms: 100
  consistency_level: eventual
```

## Security Best Practices

### 1. Network Security

```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: wdbx-network-policy
spec:
  podSelector:
    matchLabels:
      app: wdbx
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: wdbx-client
    ports:
    - protocol: TCP
      port: 8080
```

### 2. Authentication & Authorization

```zig
// TLS configuration
const tls_config = .{
    .cert_file = "/etc/wdbx/tls/cert.pem",
    .key_file = "/etc/wdbx/tls/key.pem",
    .ca_file = "/etc/wdbx/tls/ca.pem",
    .verify_mode = .peer,
    .min_version = .tls_1_3,
};

// JWT authentication
const auth_config = .{
    .enabled = true,
    .jwt_secret = std.os.getenv("WDBX_JWT_SECRET"),
    .token_expiry_hours = 24,
    .refresh_enabled = true,
};
```

### 3. Encryption

```bash
# Enable encryption at rest
export WDBX_ENCRYPTION_KEY=$(openssl rand -base64 32)
export WDBX_ENCRYPTION_ALGORITHM=AES256-GCM

# Enable audit logging
export WDBX_AUDIT_LOG=/var/log/wdbx/audit.log
export WDBX_AUDIT_LEVEL=all
```

## Production Checklist

### Pre-Deployment

- [ ] System requirements met
- [ ] Performance benchmarks completed
- [ ] Load testing passed
- [ ] Security audit completed
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] Documentation updated

### Deployment

- [ ] Configuration validated
- [ ] Health checks passing
- [ ] Metrics exporting
- [ ] Logs aggregating
- [ ] Alerts configured
- [ ] Runbooks prepared

### Post-Deployment

- [ ] Performance baseline established
- [ ] SLA targets defined
- [ ] Incident response plan ready
- [ ] Capacity planning completed
- [ ] Cost optimization reviewed

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check cache hit rate
   - Verify shard distribution
   - Review compression settings
   - Analyze query patterns

2. **Memory Pressure**
   - Adjust cache sizes
   - Increase compression
   - Review retention policies
   - Consider sharding

3. **Shard Imbalance**
   - Enable auto-rebalancing
   - Manual rebalance trigger
   - Review hash function
   - Adjust virtual nodes

## Support

For production support:
- Documentation: `/docs/production/`
- Metrics Dashboard: `http://monitoring.example.com/wdbx`
- Support Email: wdbx-support@example.com
- On-Call: PagerDuty integration

---

*Last Updated: January 2024*
*Version: 2.0.0-production*
