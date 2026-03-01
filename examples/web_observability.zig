//! Web Server + Observability Cross-Module Integration Example
//!
//! Demonstrates how the web server and observability subsystems integrate:
//! - HTTP MetricsMiddleware for per-request instrumentation
//! - Server configuration with middleware pipeline
//! - Simulated request/response cycle with latency tracking
//! - Prometheus exposition format output
//!
//! Run with: `zig build run-web-observability`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Web Server + Observability Integration Example ===\n\n", .{});

    // ── Step 1: Server Configuration ───────────────────────────────────────
    std.debug.print("--- Step 1: HTTP Server Config ---\n", .{});

    const server_config = abi.features.web.server.ServerConfig{
        .host = "0.0.0.0",
        .port = 8080,
        .max_connections = 2048,
        .read_timeout_ms = 30_000,
        .write_timeout_ms = 30_000,
        .keep_alive = true,
        .keep_alive_timeout_ms = 5_000,
        .worker_threads = 4,
    };

    std.debug.print("  Bind:             {s}:{d}\n", .{ server_config.host, server_config.port });
    std.debug.print("  Max connections:  {d}\n", .{server_config.max_connections});
    std.debug.print("  Workers:          {d}\n", .{server_config.worker_threads});
    std.debug.print("  Keep-alive:       {} ({d} ms timeout)\n", .{
        server_config.keep_alive,
        server_config.keep_alive_timeout_ms,
    });

    // ── Step 2: Metrics Middleware ──────────────────────────────────────────
    std.debug.print("\n--- Step 2: MetricsMiddleware Setup ---\n", .{});

    var metrics = abi.features.web.middleware.MetricsMiddleware.init();

    std.debug.print("  MetricsMiddleware initialized (lock-free atomic counters)\n", .{});
    std.debug.print("  Histogram buckets: {d}\n", .{abi.features.web.middleware.observability.BUCKET_COUNT});

    // Print bucket boundaries
    std.debug.print("  Bucket bounds (us): ", .{});
    for (abi.features.web.middleware.observability.bucket_bounds_us, 0..) |bound, i| {
        if (i > 0) std.debug.print(", ", .{});
        if (bound == std.math.maxInt(u64)) {
            std.debug.print("+Inf", .{});
        } else {
            std.debug.print("{d}", .{bound});
        }
    }
    std.debug.print("\n", .{});

    // ── Step 3: Simulate Request Traffic ───────────────────────────────────
    std.debug.print("\n--- Step 3: Simulated Request Traffic ---\n", .{});

    // Simulate a mix of requests with different status codes
    const sim_requests = [_]struct { path: []const u8, status: u16 }{
        .{ .path = "GET /api/vectors", .status = 200 },
        .{ .path = "GET /api/vectors/42", .status = 200 },
        .{ .path = "POST /api/vectors", .status = 201 },
        .{ .path = "GET /api/vectors/999", .status = 404 },
        .{ .path = "GET /api/search?q=test", .status = 200 },
        .{ .path = "POST /api/vectors", .status = 201 },
        .{ .path = "DELETE /api/vectors/1", .status = 204 },
        .{ .path = "GET /api/vectors", .status = 200 },
        .{ .path = "POST /api/auth/login", .status = 401 },
        .{ .path = "GET /api/health", .status = 200 },
        .{ .path = "POST /api/bulk-insert", .status = 500 },
        .{ .path = "GET /metrics", .status = 200 },
    };

    for (sim_requests) |req| {
        // Start request timing
        const req_metrics = metrics.processRequest();

        // Simulate some processing time (busy loop ~1-10 us)
        var dummy: u64 = 0;
        for (0..100) |j| dummy +%= j;
        std.mem.doNotOptimizeAway(&dummy);

        // Record the response
        metrics.recordResponse(req_metrics, req.status);

        std.debug.print("  {s: <30} -> {d}\n", .{ req.path, req.status });
    }

    // ── Step 4: Metrics Snapshot ───────────────────────────────────────────
    std.debug.print("\n--- Step 4: Metrics Snapshot ---\n", .{});

    const snap = metrics.getSnapshot();
    std.debug.print("  Total requests:    {d}\n", .{snap.total_requests});
    std.debug.print("  Total errors:      {d}\n", .{snap.total_errors});
    std.debug.print("  Active requests:   {d}\n", .{snap.active_requests});

    std.debug.print("  Status classes:\n", .{});
    const class_labels = [_][]const u8{ "1xx", "2xx", "3xx", "4xx", "5xx", "other" };
    for (class_labels, 0..) |label, i| {
        if (snap.status_counts[i] > 0) {
            std.debug.print("    {s}: {d}\n", .{ label, snap.status_counts[i] });
        }
    }

    std.debug.print("  Latency histogram:\n", .{});
    var cumulative: u64 = 0;
    for (abi.features.web.middleware.observability.bucket_bounds_us, 0..) |bound, i| {
        cumulative += snap.request_durations_us[i];
        if (bound == std.math.maxInt(u64)) {
            std.debug.print("    <= +Inf us: {d}\n", .{cumulative});
        } else {
            std.debug.print("    <= {d: >7} us: {d}\n", .{ bound, cumulative });
        }
    }

    // ── Step 5: Prometheus Exposition Format ───────────────────────────────
    std.debug.print("\n--- Step 5: Prometheus Output ---\n", .{});

    const prom_output = metrics.formatPrometheus(allocator) catch |err| {
        std.debug.print("  Failed to format Prometheus output: {t}\n", .{err});
        return;
    };
    defer allocator.free(prom_output);

    // Print first N bytes of Prometheus output as a preview
    const preview_len = @min(prom_output.len, 600);
    std.debug.print("{s}", .{prom_output[0..preview_len]});
    if (prom_output.len > preview_len) {
        std.debug.print("  ... ({d} more bytes)\n", .{prom_output.len - preview_len});
    }

    // ── Summary ────────────────────────────────────────────────────────────
    std.debug.print("\n--- Integration Summary ---\n", .{});
    std.debug.print("  A production web service combines:\n", .{});
    std.debug.print("  • web.server.ServerConfig — HTTP server configuration\n", .{});
    std.debug.print("  • web.middleware.MetricsMiddleware — per-request instrumentation\n", .{});
    std.debug.print("  • formatPrometheus() — /metrics endpoint for scraping\n", .{});
    std.debug.print("  • web.middleware.defaultChain() — logging + CORS + errors\n", .{});

    std.debug.print("\n=== Web + Observability Integration Complete ===\n", .{});
}
