# ABI Framework Docker Image
#
# Multi-stage build for minimal image size.
#
# Build:
#   docker build -t abi:latest .
#   docker build --build-arg ENABLE_GPU=true -t abi:gpu .
#
# Run:
#   docker run -it abi:latest --help
#   docker run -v $(pwd)/data:/data abi:latest db stats

# ============================================================================
# Stage 1: Build
# ============================================================================
FROM alpine:3.21 AS builder

# Build arguments for feature flags
ARG ENABLE_GPU=false
ARG ENABLE_AI=true
ARG ENABLE_DATABASE=true
ARG ENABLE_NETWORK=true
ARG GPU_BACKEND=none
ARG OPTIMIZE=ReleaseFast

# Install Zig 0.16 and build dependencies
RUN apk add --no-cache \
    curl \
    tar \
    xz \
    libc-dev \
    && curl -L https://ziglang.org/download/0.16.0/zig-linux-x86_64-0.16.0.tar.xz | tar xJ \
    && mv zig-linux-x86_64-0.16.0 /opt/zig

ENV PATH="/opt/zig:$PATH"

# Copy source code
WORKDIR /build
COPY . .

# Build ABI with specified features
RUN zig build \
    -Doptimize=${OPTIMIZE} \
    -Denable-gpu=${ENABLE_GPU} \
    -Denable-ai=${ENABLE_AI} \
    -Denable-database=${ENABLE_DATABASE} \
    -Denable-network=${ENABLE_NETWORK} \
    -Dgpu-backend=${GPU_BACKEND} \
    --prefix=/install

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM alpine:3.21 AS runtime

# Runtime dependencies
RUN apk add --no-cache \
    libstdc++ \
    ca-certificates

# Create non-root user
RUN adduser -D -u 1000 abi
USER abi
WORKDIR /home/abi

# Copy built binary
COPY --from=builder /install/bin/abi /usr/local/bin/abi

# Default data directory
VOLUME /data
ENV ABI_DATA_DIR=/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /usr/local/bin/abi system-info > /dev/null || exit 1

# Default command
ENTRYPOINT ["/usr/local/bin/abi"]
CMD ["--help"]

# ============================================================================
# Labels
# ============================================================================
LABEL org.opencontainers.image.title="ABI Framework"
LABEL org.opencontainers.image.description="High-performance AI & Vector Database Framework"
LABEL org.opencontainers.image.version="0.4.0"
LABEL org.opencontainers.image.source="https://github.com/donaldthai/abi"
LABEL org.opencontainers.image.licenses="MIT"

# ============================================================================
# GPU Stage (optional)
# ============================================================================
FROM nvidia/cuda:12.3-runtime-ubuntu22.04 AS gpu-runtime

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 abi
USER abi
WORKDIR /home/abi

# Copy built binary from builder
COPY --from=builder /install/bin/abi /usr/local/bin/abi

# Default data directory
VOLUME /data
ENV ABI_DATA_DIR=/data
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /usr/local/bin/abi system-info > /dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/abi"]
CMD ["--help"]
