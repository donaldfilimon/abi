//! Observability Example
//!
//! Demonstrates the observability subsystem:
//! - Metrics: Counters, gauges, histograms
//! - Tracing: Distributed spans
//!
//! Run with: zig build run-observability

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== ABI Observability Demo ===\n\n", .{});

    // Check if profiling is enabled
    if (!build_options.enable_profiling) {
        std.debug.print("Note: Profiling feature is disabled. Enable with -Denable-profiling=true\n\n", .{});
        return;
    }

    // Demo: Metrics
    try demoMetrics(allocator);

    std.debug.print("\n=== Demo Complete ===\n", .{});
}

/// Demonstrate Prometheus-style metrics
fn demoMetrics(allocator: std.mem.Allocator) !void {
    std.debug.print("--- Metrics Demo ---\n", .{});

    const obs = abi.features.observability;

    // Create a metrics collector
    var collector = obs.MetricsCollector.init(allocator);
    defer collector.deinit();

    // Register metrics
    const requests = try collector.registerCounter("http_requests_total");
    const active_conns = try collector.registerGauge("active_connections");
    const latency_hist = try collector.registerHistogram("request_latency_ms", &[_]u64{ 10, 50, 100, 500, 1000 });
    // Note: Histogram memory is managed by the collector

    // Simulate some metrics
    std.debug.print("Simulating application metrics:\n", .{});

    // Increment counter
    for (0..10) |_| {
        requests.inc(1);
    }
    std.debug.print("  http_requests_total: {d}\n", .{requests.get()});

    // Update gauge
    active_conns.set(42);
    std.debug.print("  active_connections: {d}\n", .{active_conns.get()});

    // Record latency samples
    const latencies = [_]u64{ 5, 12, 8, 25, 3, 15, 7, 120, 500 };
    for (latencies) |lat| {
        latency_hist.record(lat);
    }
    std.debug.print("  request_latency_ms: {d} samples recorded\n", .{latencies.len});

    // Show histogram buckets
    std.debug.print("\nLatency histogram buckets:\n", .{});
    for (latency_hist.bounds, 0..) |bound, i| {
        std.debug.print("  <= {d}ms: {d}\n", .{ bound, latency_hist.buckets[i] });
    }
    std.debug.print("  > {d}ms: {d}\n", .{ latency_hist.bounds[latency_hist.bounds.len - 1], latency_hist.buckets[latency_hist.buckets.len - 1] });
    std.debug.print("\n", .{});
}
