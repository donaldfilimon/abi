# WDBX Refactored - Production Docker Image
# Multi-stage build for optimal size and security

# Build stage
FROM alpine:3.19 AS builder

# Install build dependencies
RUN apk add --no-cache \
    curl \
    xz \
    gcc \
    musl-dev \
    make

# Install Zig
ARG ZIG_VERSION=0.15.1
RUN curl -L https://ziglang.org/download/${ZIG_VERSION}/zig-linux-x86_64-${ZIG_VERSION}.tar.xz | tar -xJ -C /opt
ENV PATH="/opt/zig-linux-x86_64-${ZIG_VERSION}:$PATH"

# Set up build directory
WORKDIR /build

# Copy source code
COPY . .

# Build the application
RUN zig build -DReleaseFast -Dstrip -Dtarget=x86_64-linux-musl

# Runtime stage
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    numactl \
    hwloc

# Create non-root user
RUN addgroup -g 1000 wdbx && \
    adduser -u 1000 -G wdbx -D wdbx

# Copy binary from build stage
COPY --from=builder /build/zig-out/bin/wdbx /usr/local/bin/
COPY --from=builder /build/zig-out/bin/wdbx-benchmark /usr/local/bin/

# Create data directory
RUN mkdir -p /var/lib/wdbx && \
    chown -R wdbx:wdbx /var/lib/wdbx

# Copy configuration
COPY deploy/config/production.yaml /etc/wdbx/config.yaml

# Environment variables for production
ENV WDBX_DATA_DIR=/var/lib/wdbx \
    WDBX_CONFIG=/etc/wdbx/config.yaml \
    WDBX_LOG_LEVEL=info \
    WDBX_ENABLE_METRICS=true \
    WDBX_METRICS_PORT=9090 \
    WDBX_API_PORT=8080 \
    WDBX_ADMIN_PORT=8081

# Expose ports
EXPOSE 8080 8081 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8081/health || exit 1

# Switch to non-root user
USER wdbx

# Set working directory
WORKDIR /var/lib/wdbx

# Entry point with NUMA optimization
ENTRYPOINT ["numactl", "--interleave=all", "/usr/local/bin/wdbx"]

# Default command
CMD ["serve"]