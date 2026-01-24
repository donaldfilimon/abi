# =============================================================================
# ABI Operator Makefile
# =============================================================================

# Image configuration
IMG ?= abi-operator:latest
REGISTRY ?= ghcr.io/abi

# Kubernetes tools
KUBECTL ?= kubectl
KUSTOMIZE ?= kustomize

# Go configuration
GOOS ?= linux
GOARCH ?= amd64

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: fmt
fmt: ## Run go fmt against code
	go fmt ./...

.PHONY: vet
vet: ## Run go vet against code
	go vet ./...

.PHONY: lint
lint: ## Run golangci-lint against code
	golangci-lint run

.PHONY: test
test: ## Run tests
	go test ./... -coverprofile cover.out

##@ Build

.PHONY: build
build: fmt vet ## Build operator binary
	CGO_ENABLED=0 GOOS=$(GOOS) GOARCH=$(GOARCH) go build -ldflags="-w -s" -o bin/manager main.go

.PHONY: docker-build
docker-build: ## Build docker image
	docker build -t $(IMG) .

.PHONY: docker-push
docker-push: ## Push docker image
	docker push $(IMG)

.PHONY: docker-buildx
docker-buildx: ## Build and push multi-arch docker image
	docker buildx build --platform linux/amd64,linux/arm64 -t $(REGISTRY)/$(IMG) --push .

##@ Deployment

.PHONY: install-crds
install-crds: ## Install CRDs into the cluster
	$(KUBECTL) apply -f config/crd/

.PHONY: uninstall-crds
uninstall-crds: ## Uninstall CRDs from the cluster
	$(KUBECTL) delete -f config/crd/

.PHONY: deploy
deploy: install-crds ## Deploy operator to the cluster
	$(KUBECTL) apply -f config/rbac/
	$(KUBECTL) apply -f config/manager/

.PHONY: undeploy
undeploy: ## Undeploy operator from the cluster
	$(KUBECTL) delete -f config/manager/
	$(KUBECTL) delete -f config/rbac/
	$(KUBECTL) delete -f config/crd/

.PHONY: deploy-sample
deploy-sample: ## Deploy sample AbiInstance CR
	$(KUBECTL) apply -f config/samples/

##@ Tools

.PHONY: generate
generate: ## Generate code (deepcopy, etc.)
	go generate ./...

.PHONY: manifests
manifests: ## Generate CRD manifests
	controller-gen crd paths="./..." output:crd:artifacts:config=config/crd

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf bin/
	rm -f cover.out

##@ Local Development

.PHONY: run
run: ## Run operator locally (outside cluster)
	go run main.go --kubeconfig=$(HOME)/.kube/config

.PHONY: kind-load
kind-load: docker-build ## Load image into kind cluster
	kind load docker-image $(IMG)
