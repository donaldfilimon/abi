# Metrics Module Consolidation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify 4 scattered metrics.zig files into a central observability module with shared primitives and domain-specific extensions.

**Architecture:** Create `src/observability/metrics/` with core primitives (Counter, Gauge, Histogram), then refactor domain-specific metrics (GPU, AI eval, personas) to use these shared types. Preserve Prometheus export capability.

**Tech Stack:** Zig 0.16, std.Thread.Mutex for thread-safety, std.ArrayList for dynamic storage

---

## Current State Analysis

| File | Size | Key Types | Thread-Safe | Prometheus |
|------|------|-----------|-------------|------------|
| `src/gpu/metrics.zig` | 15KB | MetricType, DataPoint, KernelMetrics | No | No |
| `src/gpu/mega/metrics.zig` | 14KB | Counter, Gauge, HistogramValue, MetricsExporter | Yes | Yes |
| `src/ai/eval/metrics.zig` | 15KB | TokenMetrics, TextStatistics, F1Score | No | No |
| `src/ai/personas/metrics.zig` | 14KB | LatencyPercentiles, LatencyWindow, PersonaMetricSet | Yes | No |

**Shared patterns to consolidate:**
- Counter (increment, add) - in mega/metrics and personas/metrics
- Gauge (set, inc, dec, add) - in mega/metrics
- Histogram with buckets - in mega/metrics (fixed buckets) and personas/metrics (sliding window)
- Percentile calculation - in both histogram implementations

---

## Task 1: Create Core Metrics Primitives

**Files:**
- Create: `src/observability/metrics/primitives.zig`
- Create: `src/observability/metrics/mod.zig`

**Step 1: Create the primitives file with Counter type**

```zig
//! Core Metrics Primitives
//!
//! Thread-safe metric types for counters, gauges, and histograms.
//! Used as building blocks for domain-specific metrics.

const std = @import("std");

/// Monotonically increasing counter.
pub const Counter = struct {
    value: u64 = 0,

    pub fn inc(self: *Counter) void {
        _ = @atomicRmw(u64, &self.value, .Add, 1, .monotonic);
    }

    pub fn add(self: *Counter, n: u64) void {
        _ = @atomicRmw(u64, &self.value, .Add, n, .monotonic);
    }

    pub fn get(self: *const Counter) u64 {
        return @atomicLoad(u64, &self.value, .monotonic);
    }

    pub fn reset(self: *Counter) void {
        @atomicStore(u64, &self.value, 0, .monotonic);
    }
};

/// Value that can increase or decrease.
pub const Gauge = struct {
    value: i64 = 0,

    pub fn set(self: *Gauge, v: i64) void {
        @atomicStore(i64, &self.value, v, .monotonic);
    }

    pub fn inc(self: *Gauge) void {
        _ = @atomicRmw(i64, &self.value, .Add, 1, .monotonic);
    }

    pub fn dec(self: *Gauge) void {
        _ = @atomicRmw(i64, &self.value, .Sub, 1, .monotonic);
    }

    pub fn add(self: *Gauge, v: i64) void {
        _ = @atomicRmw(i64, &self.value, .Add, v, .monotonic);
    }

    pub fn get(self: *const Gauge) i64 {
        return @atomicLoad(i64, &self.value, .monotonic);
    }
};

/// Float gauge for non-integer measurements (requires mutex).
pub const FloatGauge = struct {
    value: f64 = 0,
    mutex: std.Thread.Mutex = .{},

    pub fn set(self: *FloatGauge, v: f64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.value = v;
    }

    pub fn add(self: *FloatGauge, v: f64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.value += v;
    }

    pub fn get(self: *FloatGauge) f64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.value;
    }
};
```

**Step 2: Run `zig fmt src/observability/metrics/primitives.zig`**

**Step 3: Add histogram to primitives.zig**

Append after FloatGauge:

```zig
/// Standard latency buckets in milliseconds.
pub const default_latency_buckets = [_]f64{ 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000 };

/// Histogram with configurable buckets.
pub fn Histogram(comptime bucket_count: usize) type {
    return struct {
        const Self = @This();

        buckets: [bucket_count]u64 = [_]u64{0} ** bucket_count,
        bucket_bounds: [bucket_count]f64,
        sum: f64 = 0,
        count: u64 = 0,
        mutex: std.Thread.Mutex = .{},

        pub fn init(bounds: [bucket_count]f64) Self {
            return .{ .bucket_bounds = bounds };
        }

        pub fn initDefault() Self {
            comptime {
                if (bucket_count != default_latency_buckets.len) {
                    @compileError("Use Histogram(14) for default latency buckets");
                }
            }
            return .{ .bucket_bounds = default_latency_buckets };
        }

        pub fn observe(self: *Self, value: f64) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            self.sum += value;
            self.count += 1;
            for (&self.buckets, 0..) |*bucket, i| {
                if (value <= self.bucket_bounds[i]) {
                    bucket.* += 1;
                }
            }
        }

        pub fn mean(self: *Self) f64 {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.count == 0) return 0;
            return self.sum / @as(f64, @floatFromInt(self.count));
        }

        pub fn percentile(self: *Self, p: f64) f64 {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.count == 0) return 0;

            const target = @as(f64, @floatFromInt(self.count)) * p;
            var cumulative: u64 = 0;
            for (self.buckets, 0..) |bucket, i| {
                cumulative += bucket;
                if (@as(f64, @floatFromInt(cumulative)) >= target) {
                    return self.bucket_bounds[i];
                }
            }
            return self.bucket_bounds[bucket_count - 1];
        }

        pub fn getCount(self: *Self) u64 {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.count;
        }

        pub fn reset(self: *Self) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.buckets = [_]u64{0} ** bucket_count;
            self.sum = 0;
            self.count = 0;
        }
    };
}

/// Standard latency histogram with default buckets.
pub const LatencyHistogram = Histogram(default_latency_buckets.len);
```

**Step 4: Create mod.zig entry point**

```zig
//! Observability Metrics Module
//!
//! Provides core metric primitives and domain-specific metric types.

pub const primitives = @import("primitives.zig");

// Re-export core types
pub const Counter = primitives.Counter;
pub const Gauge = primitives.Gauge;
pub const FloatGauge = primitives.FloatGauge;
pub const Histogram = primitives.Histogram;
pub const LatencyHistogram = primitives.LatencyHistogram;
pub const default_latency_buckets = primitives.default_latency_buckets;

test {
    _ = primitives;
}
```

**Step 5: Run tests**

Run: `zig test src/observability/metrics/primitives.zig`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/observability/metrics/
git commit -m "feat(observability): add core metrics primitives

Add thread-safe Counter, Gauge, FloatGauge, and Histogram types
as foundation for consolidated metrics system.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Prometheus Export Support

**Files:**
- Create: `src/observability/metrics/prometheus.zig`
- Modify: `src/observability/metrics/mod.zig`

**Step 1: Create prometheus.zig with MetricWriter**

```zig
//! Prometheus Text Format Export
//!
//! Writes metrics in Prometheus exposition format.

const std = @import("std");
const primitives = @import("primitives.zig");

pub const MetricWriter = struct {
    output: std.ArrayList(u8),

    pub fn init(allocator: std.mem.Allocator) MetricWriter {
        return .{ .output = std.ArrayList(u8).init(allocator) };
    }

    pub fn deinit(self: *MetricWriter) void {
        self.output.deinit();
    }

    pub fn writeCounter(self: *MetricWriter, name: []const u8, help: []const u8, value: u64, labels: ?[]const u8) !void {
        const writer = self.output.writer();
        try writer.print("# HELP {s} {s}\n", .{ name, help });
        try writer.print("# TYPE {s} counter\n", .{name});
        if (labels) |l| {
            try writer.print("{s}{{{s}}} {d}\n\n", .{ name, l, value });
        } else {
            try writer.print("{s} {d}\n\n", .{ name, value });
        }
    }

    pub fn writeGauge(self: *MetricWriter, name: []const u8, help: []const u8, value: anytype, labels: ?[]const u8) !void {
        const writer = self.output.writer();
        try writer.print("# HELP {s} {s}\n", .{ name, help });
        try writer.print("# TYPE {s} gauge\n", .{name});
        if (labels) |l| {
            try writer.print("{s}{{{s}}} {d}\n\n", .{ name, l, value });
        } else {
            try writer.print("{s} {d}\n\n", .{ name, value });
        }
    }

    pub fn writeHistogram(
        self: *MetricWriter,
        name: []const u8,
        help: []const u8,
        comptime bucket_count: usize,
        histogram: *primitives.Histogram(bucket_count),
        labels: ?[]const u8,
    ) !void {
        histogram.mutex.lock();
        defer histogram.mutex.unlock();

        const writer = self.output.writer();
        try writer.print("# HELP {s} {s}\n", .{ name, help });
        try writer.print("# TYPE {s} histogram\n", .{name});

        const label_prefix = if (labels) |l| l else "";
        const comma = if (labels != null) "," else "";

        var cumulative: u64 = 0;
        for (histogram.buckets, 0..) |bucket, i| {
            cumulative += bucket;
            try writer.print("{s}_bucket{{{s}{s}le=\"{d}\"}} {d}\n", .{
                name, label_prefix, comma, histogram.bucket_bounds[i], cumulative,
            });
        }
        try writer.print("{s}_bucket{{{s}{s}le=\"+Inf\"}} {d}\n", .{
            name, label_prefix, comma, histogram.count,
        });
        try writer.print("{s}_sum{{{s}}} {d}\n", .{ name, label_prefix, histogram.sum });
        try writer.print("{s}_count{{{s}}} {d}\n\n", .{ name, label_prefix, histogram.count });
    }

    pub fn finish(self: *MetricWriter) ![]u8 {
        return try self.output.toOwnedSlice();
    }
};

test "prometheus counter export" {
    const allocator = std.testing.allocator;
    var writer = MetricWriter.init(allocator);
    defer writer.deinit();

    try writer.writeCounter("requests_total", "Total requests", 42, null);
    const output = try writer.finish();
    defer allocator.free(output);

    try std.testing.expect(std.mem.indexOf(u8, output, "requests_total 42") != null);
}
```

**Step 2: Update mod.zig to include prometheus**

Add to mod.zig:

```zig
pub const prometheus = @import("prometheus.zig");
pub const MetricWriter = prometheus.MetricWriter;
```

And update test block:

```zig
test {
    _ = primitives;
    _ = prometheus;
}
```

**Step 3: Run tests**

Run: `zig test src/observability/metrics/mod.zig`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/observability/metrics/prometheus.zig src/observability/metrics/mod.zig
git commit -m "feat(observability): add Prometheus text format export

Add MetricWriter for exporting Counter, Gauge, and Histogram
metrics in Prometheus exposition format.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Sliding Window Histogram

**Files:**
- Create: `src/observability/metrics/sliding_window.zig`
- Modify: `src/observability/metrics/mod.zig`

**Step 1: Create sliding_window.zig**

This captures the LatencyWindow pattern from ai/personas/metrics.zig:

```zig
//! Sliding Window Metrics
//!
//! Time-windowed metrics that automatically expire old samples.

const std = @import("std");

/// A sample with timestamp.
pub const TimestampedSample = struct {
    value: f64,
    timestamp_ms: i64,
};

/// Sliding window for latency tracking with automatic expiration.
pub fn SlidingWindow(comptime max_samples: usize) type {
    return struct {
        const Self = @This();

        samples: [max_samples]TimestampedSample = undefined,
        count: usize = 0,
        head: usize = 0,
        window_ms: i64,
        mutex: std.Thread.Mutex = .{},

        pub fn init(window_ms: i64) Self {
            return .{ .window_ms = window_ms };
        }

        pub fn record(self: *Self, value: f64, now_ms: i64) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            // Add new sample
            self.samples[self.head] = .{ .value = value, .timestamp_ms = now_ms };
            self.head = (self.head + 1) % max_samples;
            if (self.count < max_samples) self.count += 1;
        }

        pub fn percentile(self: *Self, p: f64, now_ms: i64) f64 {
            self.mutex.lock();
            defer self.mutex.unlock();

            const cutoff = now_ms - self.window_ms;

            // Collect valid samples
            var valid: [max_samples]f64 = undefined;
            var valid_count: usize = 0;

            for (0..self.count) |i| {
                const idx = (self.head + max_samples - 1 - i) % max_samples;
                if (self.samples[idx].timestamp_ms >= cutoff) {
                    valid[valid_count] = self.samples[idx].value;
                    valid_count += 1;
                }
            }

            if (valid_count == 0) return 0;

            // Sort for percentile calculation
            std.mem.sort(f64, valid[0..valid_count], {}, std.sort.asc(f64));

            const index = @as(usize, @intFromFloat(@as(f64, @floatFromInt(valid_count - 1)) * p));
            return valid[index];
        }

        pub fn mean(self: *Self, now_ms: i64) f64 {
            self.mutex.lock();
            defer self.mutex.unlock();

            const cutoff = now_ms - self.window_ms;
            var sum: f64 = 0;
            var count: usize = 0;

            for (0..self.count) |i| {
                const idx = (self.head + max_samples - 1 - i) % max_samples;
                if (self.samples[idx].timestamp_ms >= cutoff) {
                    sum += self.samples[idx].value;
                    count += 1;
                }
            }

            if (count == 0) return 0;
            return sum / @as(f64, @floatFromInt(count));
        }

        pub fn validCount(self: *Self, now_ms: i64) usize {
            self.mutex.lock();
            defer self.mutex.unlock();

            const cutoff = now_ms - self.window_ms;
            var count: usize = 0;

            for (0..self.count) |i| {
                const idx = (self.head + max_samples - 1 - i) % max_samples;
                if (self.samples[idx].timestamp_ms >= cutoff) {
                    count += 1;
                }
            }

            return count;
        }
    };
}

/// Standard 1000-sample sliding window.
pub const StandardWindow = SlidingWindow(1000);

test "sliding window percentile" {
    var window = SlidingWindow(100).init(60000); // 1 minute window

    // Record some values
    window.record(10, 1000);
    window.record(20, 2000);
    window.record(30, 3000);
    window.record(40, 4000);
    window.record(50, 5000);

    // P50 should be around 30
    const p50 = window.percentile(0.5, 10000);
    try std.testing.expect(p50 >= 20 and p50 <= 40);
}
```

**Step 2: Update mod.zig**

Add:

```zig
pub const sliding_window = @import("sliding_window.zig");
pub const SlidingWindow = sliding_window.SlidingWindow;
pub const StandardWindow = sliding_window.StandardWindow;
pub const TimestampedSample = sliding_window.TimestampedSample;
```

Update test block:

```zig
test {
    _ = primitives;
    _ = prometheus;
    _ = sliding_window;
}
```

**Step 3: Run tests**

Run: `zig test src/observability/metrics/mod.zig`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/observability/metrics/sliding_window.zig src/observability/metrics/mod.zig
git commit -m "feat(observability): add sliding window metrics

Add time-windowed metrics with automatic sample expiration
for latency percentile tracking.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Refactor GPU Mega Metrics

**Files:**
- Modify: `src/gpu/mega/metrics.zig`

**Step 1: Update imports to use new primitives**

Replace the existing Counter, Gauge, and HistogramValue implementations with imports:

```zig
//! GPU Metrics and Prometheus Export
//!
//! Provides metrics collection for GPU workloads with Prometheus-compatible
//! export. Uses shared observability primitives.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const obs = @import("../../observability/metrics/mod.zig");

// Re-export for API compatibility
pub const Counter = obs.Counter;
pub const Gauge = obs.Gauge;
pub const HistogramValue = obs.LatencyHistogram;
pub const default_latency_buckets = obs.default_latency_buckets;
```

**Step 2: Update BackendMetrics to use new types**

The BackendMetrics struct should now use the imported types (the API is compatible):

```zig
/// Per-backend metrics.
pub const BackendMetrics = struct {
    workload_count: Counter = .{},
    workload_success: Counter = .{},
    workload_failure: Counter = .{},
    failover_count: Counter = .{},
    latency_histogram: HistogramValue = HistogramValue.initDefault(),
    active_workloads: obs.FloatGauge = .{},
    energy_wh: obs.FloatGauge = .{},

    /// Record a completed workload.
    pub fn recordWorkload(self: *BackendMetrics, latency_ms: f64, success: bool) void {
        self.workload_count.inc();
        if (success) {
            self.workload_success.inc();
        } else {
            self.workload_failure.inc();
        }
        self.latency_histogram.observe(latency_ms);
    }

    /// Record a failover event.
    pub fn recordFailover(self: *BackendMetrics) void {
        self.failover_count.inc();
    }
};
```

**Step 3: Update MetricsExporter to use MetricWriter**

Replace the manual Prometheus formatting with MetricWriter usage in `exportPrometheus`:

```zig
/// Export metrics in Prometheus text format.
pub fn exportPrometheus(self: *MetricsExporter, allocator: std.mem.Allocator) ![]u8 {
    // Snapshot under lock
    const snapshot = blk: {
        self.mutex.lock();
        defer self.mutex.unlock();

        var cloned = std.AutoHashMap(backend_mod.Backend, BackendMetrics).init(allocator);
        errdefer cloned.deinit();

        var iter = self.metrics.iterator();
        while (iter.next()) |entry| {
            try cloned.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        break :blk .{
            .metrics = cloned,
            .global_workload_count = self.global_workload_count.get(),
            .global_failover_count = self.global_failover_count.get(),
        };
    };
    defer snapshot.metrics.deinit();

    var writer = obs.MetricWriter.init(allocator);
    defer writer.deinit();

    // Global metrics
    try writer.writeCounter("gpu_mega_workload_total", "Total workloads processed", snapshot.global_workload_count, null);
    try writer.writeCounter("gpu_mega_failover_total", "Total failover events", snapshot.global_failover_count, null);

    // Per-backend metrics
    var iter = snapshot.metrics.iterator();
    while (iter.next()) |entry| {
        const backend_name = @tagName(entry.key_ptr.*);
        const m = entry.value_ptr;
        const labels = std.fmt.allocPrint(allocator, "backend=\"{s}\"", .{backend_name}) catch continue;
        defer allocator.free(labels);

        try writer.writeCounter("gpu_mega_backend_workload_total", "Workloads per backend", m.workload_count.get(), labels);
        try writer.writeCounter("gpu_mega_backend_success_total", "Successful workloads per backend", m.workload_success.get(), labels);
        try writer.writeCounter("gpu_mega_backend_failure_total", "Failed workloads per backend", m.workload_failure.get(), labels);
        try writer.writeCounter("gpu_mega_backend_failover_total", "Failover events per backend", m.failover_count.get(), labels);
    }

    return try writer.finish();
}
```

**Step 4: Run tests**

Run: `zig test src/gpu/mega/metrics.zig`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/gpu/mega/metrics.zig
git commit -m "refactor(gpu): use shared observability primitives

Migrate gpu/mega/metrics.zig to use centralized Counter, Gauge,
Histogram, and MetricWriter from observability module.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Refactor AI Personas Metrics

**Files:**
- Modify: `src/ai/personas/metrics.zig`

**Step 1: Update imports**

```zig
//! Persona Metrics
//!
//! Performance tracking for AI personas using shared observability primitives.

const std = @import("std");
const obs = @import("../../observability/metrics/mod.zig");

// Use shared types
pub const LatencyWindow = obs.SlidingWindow(1000);
```

**Step 2: Update LatencyPercentiles to use shared window**

```zig
pub const LatencyPercentiles = struct {
    window: LatencyWindow,

    pub fn init(window_ms: i64) LatencyPercentiles {
        return .{ .window = LatencyWindow.init(window_ms) };
    }

    pub fn record(self: *LatencyPercentiles, latency_ms: f64, now_ms: i64) void {
        self.window.record(latency_ms, now_ms);
    }

    pub fn p50(self: *LatencyPercentiles, now_ms: i64) f64 {
        return self.window.percentile(0.5, now_ms);
    }

    pub fn p90(self: *LatencyPercentiles, now_ms: i64) f64 {
        return self.window.percentile(0.9, now_ms);
    }

    pub fn p99(self: *LatencyPercentiles, now_ms: i64) f64 {
        return self.window.percentile(0.99, now_ms);
    }
};
```

**Step 3: Update PersonaMetricSet to use shared Counter**

```zig
pub const PersonaMetricSet = struct {
    request_count: obs.Counter = .{},
    success_count: obs.Counter = .{},
    failure_count: obs.Counter = .{},
    latency: LatencyPercentiles,

    pub fn init() PersonaMetricSet {
        return .{
            .latency = LatencyPercentiles.init(300000), // 5 minute window
        };
    }

    pub fn recordRequest(self: *PersonaMetricSet, latency_ms: f64, success: bool, now_ms: i64) void {
        self.request_count.inc();
        if (success) {
            self.success_count.inc();
        } else {
            self.failure_count.inc();
        }
        self.latency.record(latency_ms, now_ms);
    }
};
```

**Step 4: Run tests**

Run: `zig test src/ai/personas/metrics.zig`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/ai/personas/metrics.zig
git commit -m "refactor(ai): use shared observability primitives in personas

Migrate personas/metrics.zig to use centralized Counter and
SlidingWindow from observability module.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update Observability Module Entry Point

**Files:**
- Modify: `src/observability/mod.zig`

**Step 1: Add metrics module to observability**

```zig
// In src/observability/mod.zig, add:
pub const metrics = @import("metrics/mod.zig");
```

**Step 2: Verify full build**

Run: `zig build`
Expected: Build succeeds

**Step 3: Run all tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/observability/mod.zig
git commit -m "feat(observability): integrate metrics module

Wire centralized metrics into main observability module entry point.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification Checklist

After all tasks:

```bash
# Format check
zig fmt --check .

# Full test suite
zig build test --summary all

# Build check
zig build

# Verify imports work
zig build run -- system-info
```

## Future Work (Not in This Plan)

- Migrate `src/gpu/metrics.zig` (requires more invasive changes to GPU module)
- Migrate `src/ai/eval/metrics.zig` (F1/precision/recall are domain-specific)
- Add OpenTelemetry export format
- Add metrics aggregation server
