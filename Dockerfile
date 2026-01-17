# =============================================================================
# ABI - AI Agent System Dockerfile
# Multi-stage build for production deployment
# =============================================================================

# Stage 1: Build environment
FROM alpine:3.19 AS builder

# Install build dependencies
RUN apk add --no-cache \
    curl \
    tar \
    xz \
    build-base \
    linux-headers \
    openssl-dev \
    openssl-libs-static \
    zlib-dev \
    zlib-static

# Install Zig 0.16.x
ARG ZIG_VERSION=0.16.0
RUN curl -L "https://ziglang.org/download/${ZIG_VERSION}/zig-linux-x86_64-${ZIG_VERSION}.tar.xz" \
    | tar -xJ -C /opt && \
    ln -s /opt/zig-linux-x86_64-${ZIG_VERSION}/zig /usr/local/bin/zig

WORKDIR /build

# Copy source files
COPY . .

# Build with production optimizations
ARG ENABLE_AI=true
ARG ENABLE_GPU=false
ARG ENABLE_WEB=true
ARG ENABLE_DATABASE=true
ARG ENABLE_NETWORK=false
ARG ENABLE_PROFILING=true

RUN zig build \
    -Doptimize=ReleaseSafe \
    -Denable-ai=${ENABLE_AI} \
    -Denable-gpu=${ENABLE_GPU} \
    -Denable-web=${ENABLE_WEB} \
    -Denable-database=${ENABLE_DATABASE} \
    -Denable-network=${ENABLE_NETWORK} \
    -Denable-profiling=${ENABLE_PROFILING}

# Stage 2: Runtime environment
FROM alpine:3.19 AS runtime

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    libstdc++ \
    libgcc \
    tini

# Security: Create non-root user
RUN addgroup -g 1000 abbey && \
    adduser -u 1000 -G abbey -s /bin/sh -D abbey

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R abbey:abbey /app

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/zig-out/bin/abi /app/abi

# Copy configuration files if they exist
COPY --from=builder /build/config* /app/config/ 2>/dev/null || true

# Switch to non-root user
USER abbey

# Expose ports
# 8080: HTTP API
# 9090: Metrics endpoint
# 50051: gRPC (if enabled)
EXPOSE 8080 9090 50051

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/app/abi", "health"] || exit 1

# Use tini for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]

# Default command
CMD ["/app/abi", "serve", "--host", "0.0.0.0", "--port", "8080", "--metrics-port", "9090"]
