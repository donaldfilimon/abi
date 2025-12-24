# ABI Framework Deployment Guide

This guide covers deploying the ABI AI Framework in production environments using Docker, Kubernetes, and monitoring stacks.

## 🚀 Quick Start with Docker Compose

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for GPU acceleration)

### Development Deployment

1. **Clone and navigate to the project:**
   ```bash
   git clone <repository-url>
   cd abi/deploy/docker
   ```

2. **Start the development stack:**
   ```bash
   docker-compose up -d
   ```

3. **Check service status:**
   ```bash
   docker-compose ps
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f abi-ai-framework
   ```

5. **Access the API:**
   - HTTP API: http://localhost:8080
   - Health Check: http://localhost:8080/health
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

### GPU-Enabled Deployment

For GPU acceleration, use the GPU profile:

```bash
# Start only GPU-enabled services
docker-compose --profile gpu up -d

# Or start all services including GPU
docker-compose --profile gpu up -d
```

## ☸️ Kubernetes Production Deployment

### Prerequisites
- Kubernetes 1.24+
- kubectl configured
- Helm 3.0+ (recommended)
- NVIDIA GPU Operator (for GPU workloads)

### Deploy to Kubernetes

1. **Create namespace:**
   ```bash
   kubectl create namespace ai-production
   ```

2. **Deploy with kubectl:**
   ```bash
   cd deploy/kubernetes
   kubectl apply -f .
   ```

3. **Check deployment status:**
   ```bash
   kubectl get pods -n ai-production
   kubectl get services -n ai-production
   ```

4. **Monitor deployment:**
   ```bash
   kubectl logs -f deployment/abi-ai-framework -n ai-production
   ```

### Scaling and High Availability

The deployment includes:
- **3 replicas** by default for high availability
- **Pod anti-affinity** to distribute across nodes
- **Resource limits** and requests
- **Health checks** and readiness probes
- **Rolling updates** for zero-downtime deployments

### GPU Support in Kubernetes

For GPU workloads, ensure:
1. **NVIDIA GPU Operator** is installed
2. **GPU nodes** have appropriate labels
3. **Resource requests** include GPU requirements

## 📊 Monitoring and Observability

### Accessing Monitoring Stack

- **Grafana Dashboard**: http://localhost:3000
  - Username: admin
  - Password: admin (change in production!)

- **Prometheus Metrics**: http://localhost:9090
  - Query ABI metrics directly
  - Configure alerting rules

### Key Metrics to Monitor

- **API Performance**: Request rate, response times, error rates
- **Resource Usage**: CPU, memory, GPU utilization
- **Database Performance**: Query latency, connection counts
- **Model Inference**: Inference times, throughput
- **System Health**: Service availability, error rates

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ABI_PORT` | HTTP server port | 8080 |
| `ABI_HOST` | Server bind address | 0.0.0.0 |
| `ABI_DATABASE_PATH` | Vector database path | /app/data/vectors.wdbx |
| `ABI_ENABLE_GPU` | Enable GPU acceleration | false |
| `ABI_ENABLE_SIMD` | Enable SIMD optimizations | true |
| `ABI_WORKER_THREADS` | Number of worker threads | auto |
| `ABI_MEMORY_LIMIT_MB` | Memory limit in MB | 2048 |

### Configuration Files

- `development.wdbx-config`: Development configuration
- `gpu.wdbx-config`: GPU-optimized configuration
- `production.wdbx-config`: Production settings (Kubernetes)

## 🔒 Security Considerations

### Production Hardening

1. **Change default passwords** for Grafana and databases
2. **Enable SSL/TLS** for all services
3. **Configure authentication** and authorization
4. **Set up network policies** in Kubernetes
5. **Enable audit logging** and monitoring
6. **Regular security updates** for all containers

### Network Security

- **Use internal networking** for service-to-service communication
- **Enable TLS everywhere** in production
- **Configure firewalls** and security groups
- **Implement rate limiting** and DDoS protection

## 📈 Scaling and Performance

### Horizontal Scaling

```bash
# Scale the deployment
kubectl scale deployment abi-ai-framework --replicas=5 -n ai-production
```

### Vertical Scaling

Update resource requests/limits in the Kubernetes deployment:
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Database Scaling

The deployment includes:
- **Persistent volume claims** for data persistence
- **Connection pooling** configuration
- **Read replicas** support (configure in database settings)

## 🚨 Troubleshooting

### Common Issues

1. **GPU not detected:**
   ```bash
   # Check GPU status
   nvidia-smi
   # Verify GPU operator
   kubectl get pods -n gpu-operator
   ```

2. **Database connection failures:**
   ```bash
   # Check database logs
   docker-compose logs abi-database
   # Verify connection string
   docker-compose exec abi-database psql -U abi -d abi_metadata
   ```

3. **Out of memory errors:**
   - Increase memory limits in docker-compose.yml or Kubernetes deployment
   - Monitor memory usage in Grafana dashboard

4. **Slow performance:**
   - Enable GPU acceleration if available
   - Increase worker thread count
   - Check database performance metrics

### Logs and Debugging

```bash
# View all service logs
docker-compose logs -f

# Check specific service
docker-compose logs -f abi-ai-framework

# Kubernetes logs
kubectl logs -f deployment/abi-ai-framework -n ai-production

# Check resource usage
kubectl top pods -n ai-production
```

## 📚 Additional Resources

- [ABI Framework Documentation](../docs/)
- [API Reference](../docs/api/)
- [Performance Tuning Guide](../docs/performance/)
- [Troubleshooting Guide](../docs/troubleshooting/)

## 🤝 Support

For deployment issues or questions:
- Check the [troubleshooting guide](../docs/troubleshooting/)
- Review [GitHub Issues](https://github.com/your-org/abi/issues)
- Contact the development team

---

**Last Updated**: December 2025
**Version**: ABI Framework v0.2.0