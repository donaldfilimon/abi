# WDBX-AI Makefile

# Variables
PROJECT_NAME := wdbx-ai
VERSION := $(shell git describe --tags --always 2>/dev/null || echo "dev")
BUILD_DIR := zig-out
RELEASE_DIR := release
DOCKER_IMAGE := wdbx/wdbx-ai
ZIG := zig

# Colors
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# Default target
.DEFAULT_GOAL := help

## help: Show this help message
.PHONY: help
help:
	@echo "WDBX-AI Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  ${GREEN}%-15s${NC} %s\n", $$1, $$2 } /^##@/ { printf "\n${YELLOW}%s${NC}\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

## build: Build the project in debug mode
.PHONY: build
build:
	@echo "Building $(PROJECT_NAME)..."
	@$(ZIG) build

## release: Build the project in release mode
.PHONY: release
release:
	@echo "Building $(PROJECT_NAME) (release)..."
	@$(ZIG) build -Doptimize=ReleaseFast

## test: Run all tests
.PHONY: test
test:
	@echo "Running tests..."
	@$(ZIG) build test

## test-core: Run core tests
.PHONY: test-core
test-core:
	@echo "Running core tests..."
	@$(ZIG) build test-core

## test-integration: Run integration tests
.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	@$(ZIG) build test-integration

## bench: Run benchmarks
.PHONY: bench
bench:
	@echo "Running benchmarks..."
	@$(ZIG) build bench

## fmt: Format source code
.PHONY: fmt
fmt:
	@echo "Formatting code..."
	@$(ZIG) build fmt

## fmt-check: Check code formatting
.PHONY: fmt-check
fmt-check:
	@echo "Checking code format..."
	@$(ZIG) build fmt-check

## clean: Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR) $(RELEASE_DIR) zig-cache
	@echo "Clean complete"

## docs: Generate documentation
.PHONY: docs
docs:
	@echo "Generating documentation..."
	@$(ZIG) build docs

##@ Docker

## docker-build: Build Docker image
.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	@docker build -t $(DOCKER_IMAGE):$(VERSION) .
	@docker tag $(DOCKER_IMAGE):$(VERSION) $(DOCKER_IMAGE):latest

## docker-push: Push Docker image
.PHONY: docker-push
docker-push:
	@echo "Pushing Docker image..."
	@docker push $(DOCKER_IMAGE):$(VERSION)
	@docker push $(DOCKER_IMAGE):latest

## docker-run: Run Docker container
.PHONY: docker-run
docker-run:
	@echo "Running Docker container..."
	@docker run -d \
		--name $(PROJECT_NAME) \
		-p 8080:8080 \
		-v $(PWD)/data:/data \
		-v $(PWD)/config:/config:ro \
		$(DOCKER_IMAGE):latest

## docker-compose-up: Start services with docker-compose
.PHONY: docker-compose-up
docker-compose-up:
	@echo "Starting services..."
	@docker-compose up -d

## docker-compose-down: Stop services
.PHONY: docker-compose-down
docker-compose-down:
	@echo "Stopping services..."
	@docker-compose down

##@ Deployment

## install: Install WDBX-AI (requires root)
.PHONY: install
install: release
	@echo "Installing $(PROJECT_NAME)..."
	@sudo ./scripts/deploy.sh install

## uninstall: Uninstall WDBX-AI (requires root)
.PHONY: uninstall
uninstall:
	@echo "Uninstalling $(PROJECT_NAME)..."
	@sudo ./scripts/deploy.sh uninstall

## service-start: Start WDBX-AI service
.PHONY: service-start
service-start:
	@sudo systemctl start wdbx-ai

## service-stop: Stop WDBX-AI service
.PHONY: service-stop
service-stop:
	@sudo systemctl stop wdbx-ai

## service-status: Show service status
.PHONY: service-status
service-status:
	@sudo systemctl status wdbx-ai

##@ Kubernetes

## k8s-deploy: Deploy to Kubernetes
.PHONY: k8s-deploy
k8s-deploy:
	@echo "Deploying to Kubernetes..."
	@kubectl apply -f deploy/kubernetes/

## k8s-delete: Delete from Kubernetes
.PHONY: k8s-delete
k8s-delete:
	@echo "Deleting from Kubernetes..."
	@kubectl delete -f deploy/kubernetes/

## k8s-logs: Show Kubernetes logs
.PHONY: k8s-logs
k8s-logs:
	@kubectl logs -f deployment/wdbx-ai -n wdbx

##@ Utilities

## run: Run the application
.PHONY: run
run: build
	@echo "Running $(PROJECT_NAME)..."
	@$(BUILD_DIR)/bin/wdbx

## cli: Run the CLI
.PHONY: cli
cli: build
	@$(BUILD_DIR)/bin/wdbx-cli

## repl: Start interactive REPL
.PHONY: repl
repl: build
	@$(BUILD_DIR)/bin/wdbx

## version: Show version
.PHONY: version
version:
	@echo "$(PROJECT_NAME) version $(VERSION)"

## deps: Check dependencies
.PHONY: deps
deps:
	@echo "Checking dependencies..."
	@command -v $(ZIG) >/dev/null 2>&1 || { echo "Zig is not installed"; exit 1; }
	@echo "Zig version: $$($(ZIG) version)"
	@command -v docker >/dev/null 2>&1 || echo "Docker is not installed (optional)"
	@command -v kubectl >/dev/null 2>&1 || echo "kubectl is not installed (optional)"

## release-tarball: Create release tarball
.PHONY: release-tarball
release-tarball: release
	@echo "Creating release tarball..."
	@./scripts/build.sh release $(VERSION)

## ci: Run CI checks locally
.PHONY: ci
ci: fmt-check test bench
	@echo "CI checks passed!"

## all: Build everything
.PHONY: all
all: clean fmt release test docs
	@echo "Build complete!"

# Include local overrides if present
-include Makefile.local