# WDBX-AI Docker Image
# Multi-stage build for optimal size

# Build stage
FROM alpine:3.19 AS builder

# Install build dependencies
RUN apk add --no-cache \
    curl \
    xz \
    tar

# Install Zig
ARG ZIG_VERSION=0.15.1
RUN curl -L https://ziglang.org/download/${ZIG_VERSION}/zig-linux-x86_64-${ZIG_VERSION}.tar.xz | tar -xJ && \
    mv zig-linux-x86_64-${ZIG_VERSION} /usr/local/zig && \
    ln -s /usr/local/zig/zig /usr/local/bin/zig

# Copy source code
WORKDIR /build
COPY . .

# Build the project
RUN zig build -Doptimize=ReleaseFast

# Runtime stage
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    libstdc++ \
    libc6-compat

# Create user and directories
RUN addgroup -g 1000 wdbx && \
    adduser -u 1000 -G wdbx -s /bin/sh -D wdbx && \
    mkdir -p /opt/wdbx-ai /data /config /logs && \
    chown -R wdbx:wdbx /opt/wdbx-ai /data /config /logs

# Copy binaries from build stage
COPY --from=builder /build/zig-out/bin/wdbx /opt/wdbx-ai/
COPY --from=builder /build/zig-out/bin/wdbx-cli /opt/wdbx-ai/

# Copy default configuration
COPY --from=builder /build/config/default.toml /config/config.toml

# Make binaries executable
RUN chmod +x /opt/wdbx-ai/wdbx /opt/wdbx-ai/wdbx-cli && \
    ln -s /opt/wdbx-ai/wdbx /usr/local/bin/wdbx && \
    ln -s /opt/wdbx-ai/wdbx-cli /usr/local/bin/wdbx-cli

# Switch to non-root user
USER wdbx

# Expose ports
EXPOSE 8080

# Set environment variables
ENV WDBX_DATABASE_PATH=/data/wdbx.db \
    WDBX_CONFIG_PATH=/config/config.toml \
    WDBX_LOG_PATH=/logs/wdbx.log

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wdbx-cli health || exit 1

# Volume mount points
VOLUME ["/data", "/config", "/logs"]

# Default command
CMD ["wdbx", "serve", "--config", "/config/config.toml"]