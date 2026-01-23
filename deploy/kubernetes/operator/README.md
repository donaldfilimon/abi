# ABI Kubernetes Operator

The ABI Operator is a Kubernetes operator for managing ABI (Abbey) instances in your cluster. It provides declarative management of ABI deployments through custom resources.

## Overview

The operator watches for `AbiInstance` custom resources and reconciles the desired state with the actual cluster state, creating and managing:

- Deployments/StatefulSets for ABI instances
- Services for HTTP, gRPC, and metrics endpoints
- ConfigMaps and Secrets for configuration
- PersistentVolumeClaims for data storage
- HorizontalPodAutoscalers for autoscaling
- PodDisruptionBudgets for high availability
- ServiceMonitors for Prometheus integration

## Prerequisites

- Kubernetes 1.25+
- kubectl configured with cluster access
- (Optional) Prometheus Operator for ServiceMonitor support

## Installation

### Install CRDs

```bash
make install-crds
# or
kubectl apply -f config/crd/
```

### Deploy the Operator

```bash
make deploy
# or
kubectl apply -f config/rbac/
kubectl apply -f config/manager/
```

### Verify Installation

```bash
kubectl get pods -n abi-system
kubectl get crd abiinstances.abi.io
```

## Usage

### Basic Instance

Create a minimal ABI instance:

```yaml
apiVersion: abi.io/v1
kind: AbiInstance
metadata:
  name: my-abi
  namespace: default
spec:
  replicas: 1
```

Apply it:

```bash
kubectl apply -f config/samples/abi_v1_instance_basic.yaml
```

### Production Instance

For production deployments, use a more complete configuration:

```yaml
apiVersion: abi.io/v1
kind: AbiInstance
metadata:
  name: abi-production
  namespace: abi
spec:
  replicas: 3

  image:
    repository: ghcr.io/abi/abi
    tag: v1.0.0
    pullPolicy: IfNotPresent

  features:
    ai: true
    database: true
    network: true

  resources:
    requests:
      cpu: "500m"
      memory: "512Mi"
    limits:
      cpu: "2000m"
      memory: "2Gi"

  storage:
    enabled: true
    size: "50Gi"

  highAvailability:
    enabled: true
    antiAffinity: hard
    podDisruptionBudget:
      minAvailable: 2

  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilization: 70
```

### GPU-Enabled Instance

For ML workloads requiring GPU acceleration:

```yaml
apiVersion: abi.io/v1
kind: AbiInstance
metadata:
  name: abi-gpu
spec:
  replicas: 2

  features:
    ai: true
    gpu: true

  gpu:
    backend: cuda
    count: 1
    memory: "8Gi"

  llm:
    provider: local
    model: llama-7b
```

## Custom Resource Reference

### AbiInstance Spec

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `replicas` | int | 1 | Number of ABI instance replicas |
| `image` | ImageSpec | - | Container image configuration |
| `features` | FeaturesSpec | - | Feature flags (ai, gpu, database, etc.) |
| `gpu` | GPUSpec | - | GPU configuration |
| `resources` | ResourceRequirements | - | CPU/memory requirements |
| `storage` | StorageSpec | - | Persistent storage configuration |
| `networking` | NetworkingSpec | - | Service ports and type |
| `config` | ConfigSpec | - | Application configuration |
| `llm` | LLMSpec | - | LLM provider configuration |
| `secrets` | SecretsSpec | - | Secret references for API keys |
| `highAvailability` | HASpec | - | HA configuration |
| `autoscaling` | AutoscalingSpec | - | HPA configuration |

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `ai` | true | Enable AI/LLM features |
| `gpu` | false | Enable GPU acceleration |
| `database` | true | Enable vector database |
| `network` | true | Enable distributed networking |
| `web` | true | Enable web/HTTP features |
| `profiling` | false | Enable performance profiling |

### Status Fields

| Field | Description |
|-------|-------------|
| `phase` | Current phase (Pending, Creating, Running, Updating, Failed, Terminating) |
| `replicas` | Total number of replicas |
| `readyReplicas` | Number of ready replicas |
| `health.status` | Health status (Healthy, Degraded, Unhealthy, Unknown) |
| `version` | Current ABI version running |
| `conditions` | Detailed condition information |
| `endpoints` | Service endpoint URLs |

## Operations

### Scaling

Scale manually:

```bash
kubectl scale abiinstance my-abi --replicas=5
```

Or update the spec:

```bash
kubectl patch abiinstance my-abi -p '{"spec":{"replicas":5}}'
```

### Upgrading

Update the image tag:

```bash
kubectl patch abiinstance my-abi -p '{"spec":{"image":{"tag":"v1.1.0"}}}'
```

### Monitoring

Check instance status:

```bash
kubectl get abiinstance my-abi -o wide
kubectl describe abiinstance my-abi
```

View operator logs:

```bash
kubectl logs -n abi-system -l app.kubernetes.io/name=abi-operator -f
```

### Troubleshooting

Check events:

```bash
kubectl get events --field-selector involvedObject.name=my-abi
```

Check pod status:

```bash
kubectl get pods -l app.kubernetes.io/instance=my-abi
kubectl logs -l app.kubernetes.io/instance=my-abi
```

## Uninstallation

Remove an instance:

```bash
kubectl delete abiinstance my-abi
```

Remove the operator:

```bash
make undeploy
```

## Development

### Building

```bash
make build          # Build binary
make docker-build   # Build Docker image
make docker-push    # Push Docker image
```

### Testing

```bash
make test           # Run unit tests
make lint           # Run linter
```

### Running Locally

```bash
make run            # Run operator locally (outside cluster)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                        │
│  │   ABI Operator  │                                        │
│  │   (Controller)  │                                        │
│  └────────┬────────┘                                        │
│           │ watches & reconciles                            │
│           ▼                                                 │
│  ┌─────────────────┐     creates      ┌──────────────────┐ │
│  │  AbiInstance CR │ ───────────────► │   Deployment     │ │
│  │  (Custom        │                  │   Service        │ │
│  │   Resource)     │                  │   ConfigMap      │ │
│  └─────────────────┘                  │   PVC            │ │
│                                       │   HPA            │ │
│                                       │   PDB            │ │
│                                       └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## License

This project is licensed under the same terms as the main ABI project.
