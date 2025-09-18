# Multi-stage Docker build for ABI AI Framework
# Stage 1: Build stage
FROM ziglang/zig:0.16.0-dev AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy build files
COPY build.zig build.zig.zon ./

# Copy source code
COPY src/ src/

# Build optimized release binary
RUN zig build -Doptimize=ReleaseFast

# Stage 2: Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r abi && useradd -r -g abi abi

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/zig-out/bin/abi /app/abi

# Copy configuration and data directories
COPY --from=builder /app/.wdbx-config /app/.wdbx-config
COPY --from=builder /app/deploy/config/ /app/config/

# Create data directories
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R abi:abi /app

# Switch to non-root user
USER abi

# Expose ports
EXPOSE 8080 8443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["./abi", "server", "start", "--port", "8080", "--host", "0.0.0.0"]
