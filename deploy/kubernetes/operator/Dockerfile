# =============================================================================
# ABI Operator Dockerfile
# =============================================================================
# Multi-stage build for minimal production image

# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /workspace

# Copy go module files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the operator binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s -X main.Version=${VERSION:-dev}" \
    -o manager main.go

# Runtime stage
FROM gcr.io/distroless/static:nonroot

LABEL org.opencontainers.image.title="ABI Operator"
LABEL org.opencontainers.image.description="Kubernetes operator for managing ABI instances"
LABEL org.opencontainers.image.source="https://github.com/abi/abi"
LABEL org.opencontainers.image.vendor="ABI Project"

WORKDIR /

# Copy the binary from builder
COPY --from=builder /workspace/manager .

# Run as non-root user
USER 65532:65532

ENTRYPOINT ["/manager"]
