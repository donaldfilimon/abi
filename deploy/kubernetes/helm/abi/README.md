# ABI Helm Chart

A Helm chart for deploying ABI (Abbey) instances on Kubernetes.

## Prerequisites

- Kubernetes 1.25+
- Helm 3.8+
- PV provisioner support (if persistence is enabled)

## Installation

### Add the Helm repository (if published)

```bash
helm repo add abi https://abi.github.io/helm-charts
helm repo update
```

### Install from local chart

```bash
helm install my-abi ./deploy/kubernetes/helm/abi
```

### Install with custom values

```bash
helm install my-abi ./deploy/kubernetes/helm/abi -f my-values.yaml
```

### Install in a specific namespace

```bash
helm install my-abi ./deploy/kubernetes/helm/abi --namespace abi --create-namespace
```

## Configuration

See [values.yaml](values.yaml) for the full list of configurable parameters.

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image repository | `ghcr.io/abi/abi` |
| `image.tag` | Container image tag | `""` (uses appVersion) |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |

### Feature Flags

| Parameter | Description | Default |
|-----------|-------------|---------|
| `features.ai` | Enable AI/LLM features | `true` |
| `features.gpu` | Enable GPU acceleration | `false` |
| `features.database` | Enable vector database | `true` |
| `features.network` | Enable distributed networking | `true` |
| `features.web` | Enable web/HTTP features | `true` |
| `features.profiling` | Enable performance profiling | `false` |

### Resources

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resources.requests.cpu` | CPU request | `100m` |
| `resources.requests.memory` | Memory request | `256Mi` |
| `resources.limits.cpu` | CPU limit | `1000m` |
| `resources.limits.memory` | Memory limit | `1Gi` |

### Persistence

| Parameter | Description | Default |
|-----------|-------------|---------|
| `persistence.enabled` | Enable persistence | `true` |
| `persistence.size` | Storage size | `10Gi` |
| `persistence.storageClassName` | Storage class | `""` |
| `persistence.accessModes` | Access modes | `[ReadWriteOnce]` |

### Autoscaling

| Parameter | Description | Default |
|-----------|-------------|---------|
| `autoscaling.enabled` | Enable HPA | `false` |
| `autoscaling.minReplicas` | Minimum replicas | `1` |
| `autoscaling.maxReplicas` | Maximum replicas | `10` |
| `autoscaling.targetCPUUtilizationPercentage` | Target CPU utilization | `80` |

### LLM Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `llm.provider` | LLM provider | `ollama` |
| `llm.ollama.host` | Ollama service host | `http://ollama-service:11434` |
| `llm.ollama.model` | Default model | `llama3` |

### Secrets

| Parameter | Description | Default |
|-----------|-------------|---------|
| `secrets.create` | Create secrets | `false` |
| `secrets.existingSecret` | Use existing secret | `""` |
| `secrets.openaiApiKey` | OpenAI API key | `""` |
| `secrets.anthropicApiKey` | Anthropic API key | `""` |

## Examples

### Basic Installation

```bash
helm install my-abi ./deploy/kubernetes/helm/abi
```

### Production Installation

```bash
helm install my-abi ./deploy/kubernetes/helm/abi \
  --set replicaCount=3 \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=3 \
  --set autoscaling.maxReplicas=10 \
  --set podDisruptionBudget.enabled=true \
  --set podDisruptionBudget.minAvailable=2 \
  --set persistence.size=50Gi \
  --set resources.requests.cpu=500m \
  --set resources.requests.memory=512Mi \
  --set resources.limits.cpu=2000m \
  --set resources.limits.memory=2Gi
```

### With GPU Support

```bash
helm install my-abi ./deploy/kubernetes/helm/abi \
  --set features.gpu=true \
  --set gpu.backend=cuda \
  --set gpu.nvidia.enabled=true
```

### Using External Secrets

```bash
# First create the secret
kubectl create secret generic abi-api-keys \
  --from-literal=ABI_OPENAI_API_KEY=sk-xxx \
  --from-literal=ABI_ANTHROPIC_API_KEY=sk-ant-xxx

# Then reference it
helm install my-abi ./deploy/kubernetes/helm/abi \
  --set secrets.existingSecret=abi-api-keys
```

### With Ingress

```bash
helm install my-abi ./deploy/kubernetes/helm/abi \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set ingress.hosts[0].host=abi.example.com \
  --set ingress.hosts[0].paths[0].path=/ \
  --set ingress.hosts[0].paths[0].pathType=Prefix
```

## Upgrading

```bash
helm upgrade my-abi ./deploy/kubernetes/helm/abi -f my-values.yaml
```

## Uninstalling

```bash
helm uninstall my-abi
```

Note: This will not delete PersistentVolumeClaims. To delete them:

```bash
kubectl delete pvc -l app.kubernetes.io/instance=my-abi
```

## Troubleshooting

### Check pod status

```bash
kubectl get pods -l app.kubernetes.io/instance=my-abi
```

### View logs

```bash
kubectl logs -l app.kubernetes.io/instance=my-abi -f
```

### Check events

```bash
kubectl get events --sort-by=.metadata.creationTimestamp
```

## License

This chart is licensed under the same terms as the main ABI project.
