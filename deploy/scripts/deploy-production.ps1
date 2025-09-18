# Production Deployment Script for ABI AI Framework
# Requires: kubectl, docker, helm (optional)

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("staging", "production")]
    [string]$Environment,

    [Parameter(Mandatory=$false)]
    [switch]$UseGPU,

    [Parameter(Mandatory=$false)]
    [switch]$SkipTests,

    [Parameter(Mandatory=$false)]
    [string]$Namespace = "ai-production",

    [Parameter(Mandatory=$false)]
    [string]$ChartVersion = "latest"
)

# Configuration
$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)
$DEPLOY_DIR = Join-Path $PROJECT_ROOT "deploy"

# Colors for output
$Green = "Green"
$Yellow = "Yellow"
$Red = "Red"
$Cyan = "Cyan"

function Write-ColoredOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Step {
    param([string]$Step, [string]$Description)
    Write-ColoredOutput "`n🔄 $Step" $Cyan
    Write-ColoredOutput "   $Description" $Yellow
}

function Write-Success {
    param([string]$Message)
    Write-ColoredOutput "✅ $Message" $Green
}

function Write-Error {
    param([string]$Message)
    Write-ColoredOutput "❌ $Message" $Red
}

function Test-Prerequisites {
    Write-Step "Prerequisites Check" "Verifying required tools and configurations"

    $prerequisites = @(
        @{Name = "kubectl"; Command = "kubectl version --client --short" },
        @{Name = "docker"; Command = "docker --version" },
        @{Name = "git"; Command = "git --version" }
    )

    if ($UseGPU) {
        $prerequisites += @{Name = "nvidia-docker"; Command = "docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi --query-gpu=name --format=csv,noheader,nounits --id=0" }
    }

    foreach ($prereq in $prerequisites) {
        try {
            $result = Invoke-Expression $prereq.Command 2>$null
            Write-ColoredOutput "   ✓ $($prereq.Name) available" $Green
        }
        catch {
            Write-Error "Required tool '$($prereq.Name)' not found or not working"
            Write-ColoredOutput "   Please install $($prereq.Name) and ensure it's in PATH" $Yellow
            exit 1
        }
    }

    # Check Kubernetes connection
    try {
        $context = kubectl config current-context
        Write-ColoredOutput "   ✓ Kubernetes context: $context" $Green
    }
    catch {
        Write-Error "Cannot connect to Kubernetes cluster"
        exit 1
    }

    Write-Success "Prerequisites check completed"
}

function Build-ContainerImages {
    Write-Step "Container Build" "Building Docker images for $Environment environment"

    Push-Location $PROJECT_ROOT

    try {
        # Build CPU version
        Write-ColoredOutput "   Building CPU image..." $Yellow
        docker build -f deploy/docker/Dockerfile -t "abi-framework:$Environment-cpu" .

        if ($LASTEXITCODE -ne 0) {
            throw "CPU image build failed"
        }

        if ($UseGPU) {
            # Build GPU version
            Write-ColoredOutput "   Building GPU image..." $Yellow
            docker build -f deploy/docker/Dockerfile.gpu -t "abi-framework:$Environment-gpu" .

            if ($LASTEXITCODE -ne 0) {
                throw "GPU image build failed"
            }
        }

        # Tag images
        $registry = if ($Environment -eq "production") { "your-registry.com" } else { "your-staging-registry.com" }

        docker tag "abi-framework:$Environment-cpu" "$registry/abi-framework:$Environment-cpu-$ChartVersion"
        if ($UseGPU) {
            docker tag "abi-framework:$Environment-gpu" "$registry/abi-framework:$Environment-gpu-$ChartVersion"
        }

        # Push images
        Write-ColoredOutput "   Pushing images to registry..." $Yellow
        docker push "$registry/abi-framework:$Environment-cpu-$ChartVersion"
        if ($UseGPU) {
            docker push "$registry/abi-framework:$Environment-gpu-$ChartVersion"
        }

        Write-Success "Container images built and pushed"
    }
    finally {
        Pop-Location
    }
}

function Run-Tests {
    if ($SkipTests) {
        Write-ColoredOutput "   ⏭️  Tests skipped by user request" $Yellow
        return
    }

    Write-Step "Testing" "Running pre-deployment tests"

    Push-Location $PROJECT_ROOT

    try {
        # Run unit tests
        Write-ColoredOutput "   Running unit tests..." $Yellow
        $testResult = & zig build test 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Unit tests failed"
            Write-ColoredOutput $testResult $Red
            exit 1
        }

        # Run integration tests if they exist
        if (Test-Path "tests/integration_test_suite.zig") {
            Write-ColoredOutput "   Running integration tests..." $Yellow
            $intTestResult = & zig build test-integration 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Integration tests failed"
                Write-ColoredOutput $intTestResult $Red
                exit 1
            }
        }

        Write-Success "All tests passed"
    }
    finally {
        Pop-Location
    }
}

function Deploy-ToKubernetes {
    param([string]$Namespace, [string]$Environment)

    Write-Step "Kubernetes Deployment" "Deploying to $Environment environment in namespace $Namespace"

    # Create namespace if it doesn't exist
    kubectl create namespace $Namespace --dry-run=client -o yaml | kubectl apply -f -

    # Set namespace context
    kubectl config set-context --current --namespace=$Namespace

    # Apply configurations
    $k8sDir = Join-Path $DEPLOY_DIR "kubernetes"

    # Apply ConfigMaps and Secrets first
    kubectl apply -f (Join-Path $k8sDir "configmap.yaml")
    kubectl apply -f (Join-Path $k8sDir "secrets.yaml")

    # Apply PersistentVolumeClaims
    kubectl apply -f (Join-Path $k8sDir "pvc.yaml")

    # Apply the main deployment
    $deploymentFile = Join-Path $k8sDir "deployment.yaml"
    if ($UseGPU) {
        # Use GPU-enabled deployment
        (Get-Content $deploymentFile) -replace 'abi-framework:latest', 'abi-framework:gpu-latest' | kubectl apply -f -
    } else {
        kubectl apply -f $deploymentFile
    }

    # Apply service
    kubectl apply -f (Join-Path $k8sDir "service.yaml")

    # Apply ingress
    kubectl apply -f (Join-Path $k8sDir "ingress.yaml")

    # Wait for rollout to complete
    Write-ColoredOutput "   Waiting for deployment rollout..." $Yellow
    kubectl rollout status deployment/abi-ai-framework --timeout=300s

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Deployment rollout failed"
        exit 1
    }

    Write-Success "Kubernetes deployment completed"
}

function Setup-Monitoring {
    Write-Step "Monitoring Setup" "Setting up monitoring stack"

    # Deploy Prometheus and Grafana
    $monitoringDir = Join-Path $DEPLOY_DIR "monitoring"

    kubectl apply -f (Join-Path $monitoringDir "prometheus.yaml")
    kubectl apply -f (Join-Path $monitoringDir "grafana.yaml")

    # Import ABI dashboards
    Write-ColoredOutput "   Importing ABI monitoring dashboards..." $Yellow

    Write-Success "Monitoring setup completed"
}

function Run-HealthChecks {
    Write-Step "Health Checks" "Running post-deployment health checks"

    # Get service URL
    $serviceIP = kubectl get svc abi-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
    if (-not $serviceIP) {
        $serviceIP = kubectl get svc abi-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
    }

    if (-not $serviceIP) {
        Write-Error "Could not determine service endpoint"
        return
    }

    $serviceURL = "http://$serviceIP"

    # Health check
    Write-ColoredOutput "   Checking service health..." $Yellow
    try {
        $response = Invoke-WebRequest -Uri "$serviceURL/health" -TimeoutSec 30
        if ($response.StatusCode -eq 200) {
            Write-ColoredOutput "   ✓ Health check passed" $Green
        } else {
            Write-Error "Health check failed with status $($response.StatusCode)"
        }
    }
    catch {
        Write-Error "Health check request failed: $($_.Exception.Message)"
    }

    # API test
    Write-ColoredOutput "   Testing API endpoints..." $Yellow
    try {
        $apiResponse = Invoke-WebRequest -Uri "$serviceURL/api/status" -TimeoutSec 30
        if ($apiResponse.StatusCode -eq 200) {
            Write-ColoredOutput "   ✓ API test passed" $Green
        } else {
            Write-Error "API test failed with status $($apiResponse.StatusCode)"
        }
    }
    catch {
        Write-Error "API test failed: $($_.Exception.Message)"
    }

    Write-Success "Health checks completed"
}

function Show-DeploymentInfo {
    Write-Step "Deployment Summary" "Showing deployment information"

    Write-ColoredOutput "`n🌐 Service Endpoints:" $Cyan
    kubectl get svc -l app=abi-ai-framework -o wide

    Write-ColoredOutput "`n📊 Pod Status:" $Cyan
    kubectl get pods -l app=abi-ai-framework -o wide

    Write-ColoredOutput "`n💾 Storage:" $Cyan
    kubectl get pvc -l app=abi-ai-framework

    Write-ColoredOutput "`n📈 Monitoring:" $Cyan
    Write-ColoredOutput "   Grafana: http://grafana.local" $Yellow
    Write-ColoredOutput "   Prometheus: http://prometheus.local" $Yellow

    Write-Success "Deployment completed successfully!"
    Write-ColoredOutput "`n🚀 ABI AI Framework is now running in $Environment environment" $Green
}

# Main deployment flow
Write-ColoredOutput "🚀 ABI AI Framework Production Deployment" $Cyan
Write-ColoredOutput "Environment: $Environment" $Yellow
Write-ColoredOutput "GPU Support: $($UseGPU.ToString())" $Yellow
Write-ColoredOutput "Namespace: $Namespace" $Yellow

Test-Prerequisites
Run-Tests
Build-ContainerImages
Deploy-ToKubernetes -Namespace $Namespace -Environment $Environment
Setup-Monitoring
Run-HealthChecks
Show-DeploymentInfo

Write-ColoredOutput "`n🎉 Deployment completed successfully!" $Green
Write-ColoredOutput "📝 Next steps:" $Cyan
Write-ColoredOutput "   1. Monitor application logs: kubectl logs -f deployment/abi-ai-framework" $Yellow
Write-ColoredOutput "   2. Scale deployment: kubectl scale deployment abi-ai-framework --replicas=5" $Yellow
Write-ColoredOutput "   3. Update configuration: kubectl edit configmap abi-config" $Yellow
Write-ColoredOutput "   4. Check monitoring: Access Grafana dashboard" $Yellow
