FROM ziglang/zig:0.16.0 AS build

WORKDIR /app

# Copy the Zig source and compile it into a ReleaseFast binary
COPY ../zig/metrics.zig .
RUN zig build-exe metrics.zig -O ReleaseFast -lc

# Minimal runtime image
FROM gcr.io/distroless/cc

# Copy the compiled binary
COPY --from=build /app/metrics /metrics

# Expose the metrics port
EXPOSE 9100

# Run the exporter
ENTRYPOINT ["/metrics"]
