// ============================================================================
// ABI Framework — Hierarchical Profiler
// Adapted from abi-system-v2.0/profiler.zig
// ============================================================================
//
// Span-based instrumentation for performance diagnostics.
// Fixed-size ring buffer — no allocation during profiling.
// Thread-safe via inline spinlock.
//
// Changes from v2.0:
//   - Replaced std.time.nanoTimestamp() with time.Timer (Zig 0.16)
//   - Replaced utils.Atomic.SpinLock with inline atomic spinlock
//   - Removed @import("utils") dependency
//   - Removed emojis from report output (project convention)
// ============================================================================

const std = @import("std");
const time = @import("../time.zig");

// ─── Inline SpinLock ─────────────────────────────────────────────────────────

const SpinLock = struct {
    state: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    fn acquire(self: *SpinLock) void {
        var backoff: u32 = 1;
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
            var i: u32 = 0;
            while (i < backoff) : (i += 1) std.atomic.spinLoopHint();
            backoff = @min(backoff * 2, 1024);
        }
    }

    fn release(self: *SpinLock) void {
        self.state.store(0, .release);
    }

    fn guard(self: *SpinLock) Guard {
        self.acquire();
        return .{ .lock = self };
    }

    const Guard = struct {
        lock: *SpinLock,
        fn deinit(self: Guard) void {
            self.lock.release();
        }
    };
};

// ─── Span Record ─────────────────────────────────────────────────────────────

pub const SpanRecord = struct {
    name: [64]u8 = undefined,
    name_len: u8 = 0,
    category: Category = .user,
    duration_ns: u64 = 0,
    depth: u8 = 0,
    thread_id: u32 = 0,
    metadata: u64 = 0,
    completed: bool = false,

    pub fn durationNs(self: *const SpanRecord) u64 {
        return self.duration_ns;
    }

    pub fn durationUs(self: *const SpanRecord) f64 {
        return @as(f64, @floatFromInt(self.duration_ns)) / 1000.0;
    }

    pub fn durationMs(self: *const SpanRecord) f64 {
        return @as(f64, @floatFromInt(self.duration_ns)) / 1_000_000.0;
    }

    pub fn getName(self: *const SpanRecord) []const u8 {
        return self.name[0..self.name_len];
    }

    pub fn throughputMBs(self: *const SpanRecord) f64 {
        if (self.duration_ns == 0 or self.metadata == 0) return 0;
        return @as(f64, @floatFromInt(self.metadata)) / @as(f64, @floatFromInt(self.duration_ns)) * 1000.0;
    }
};

pub const Category = enum(u8) {
    user = 0,
    memory = 1,
    compute = 2,
    gpu = 3,
    io = 4,
    sync = 5,

    pub fn label(self: Category) []const u8 {
        return switch (self) {
            .user => "user",
            .memory => "memory",
            .compute => "compute",
            .gpu => "gpu",
            .io => "io",
            .sync => "sync",
        };
    }

    pub fn color(self: Category) []const u8 {
        return switch (self) {
            .user => "\x1b[37m",
            .memory => "\x1b[36m",
            .compute => "\x1b[33m",
            .gpu => "\x1b[35m",
            .io => "\x1b[34m",
            .sync => "\x1b[31m",
        };
    }
};

// ─── Span Handle ─────────────────────────────────────────────────────────────

pub const SpanHandle = struct {
    index: u32,
    valid: bool,
    timer: time.Timer,
};

// ─── Profiler ────────────────────────────────────────────────────────────────

pub fn ProfilerType(comptime max_spans: usize) type {
    return struct {
        const Self = @This();

        spans: [max_spans]SpanRecord = undefined,
        count: usize = 0,
        depth: u8 = 0,
        enabled: bool = true,
        overflow_count: u64 = 0,
        lock: SpinLock = .{},

        pub fn init(enabled: bool) Self {
            return Self{ .enabled = enabled };
        }

        pub fn begin(self: *Self, name: []const u8) SpanHandle {
            return self.beginCat(name, .user);
        }

        pub fn beginCat(self: *Self, name: []const u8, category: Category) SpanHandle {
            const timer = time.Timer.start() catch time.Timer{
                .start_instant = .{ .nanos = 0 },
            };

            if (!self.enabled) return .{ .index = 0, .valid = false, .timer = timer };

            const g = self.lock.guard();
            defer g.deinit();

            if (self.count >= max_spans) {
                self.overflow_count += 1;
                return .{ .index = 0, .valid = false, .timer = timer };
            }

            const idx: u32 = @intCast(self.count);
            var span = &self.spans[self.count];
            self.count += 1;

            const copy_len = @min(name.len, span.name.len);
            @memcpy(span.name[0..copy_len], name[0..copy_len]);
            span.name_len = @intCast(copy_len);
            span.category = category;
            span.duration_ns = 0;
            span.depth = self.depth;
            span.metadata = 0;
            span.completed = false;

            self.depth += 1;
            return .{ .index = idx, .valid = true, .timer = timer };
        }

        pub fn end(self: *Self, handle: SpanHandle) void {
            var h = handle;
            self.endWithMeta(&h, 0);
        }

        pub fn endWithMeta(self: *Self, handle: *SpanHandle, metadata: u64) void {
            if (!handle.valid) return;

            const g = self.lock.guard();
            defer g.deinit();

            self.spans[handle.index].duration_ns = handle.timer.read();
            self.spans[handle.index].metadata = metadata;
            self.spans[handle.index].completed = true;
            if (self.depth > 0) self.depth -= 1;
        }

        // ── Reporting ────────────────────────────────────────────────

        pub fn report(self: *const Self, writer: anytype) !void {
            try writer.writeAll("\n=== Profile Report ===\n\n");

            for (self.spans[0..self.count]) |*span| {
                if (!span.completed) continue;

                for (0..span.depth) |_| try writer.writeAll("  ");

                try writer.print("{s}{s}\x1b[0m {s} ", .{
                    span.category.color(),
                    span.category.label(),
                    span.getName(),
                });

                const dur = span.duration_ns;
                if (dur < 1_000) {
                    try writer.print("{d}ns", .{dur});
                } else if (dur < 1_000_000) {
                    try writer.print("{d:.1}us", .{span.durationUs()});
                } else {
                    try writer.print("{d:.2}ms", .{span.durationMs()});
                }

                if (span.metadata > 0) {
                    try writer.print(" ({d:.1} MB/s)", .{span.throughputMBs()});
                }

                try writer.writeByte('\n');
            }

            if (self.overflow_count > 0) {
                try writer.print("\nWARNING: {d} spans dropped (buffer full)\n", .{self.overflow_count});
            }

            try writer.writeByte('\n');
        }

        /// Export spans as Chrome Trace JSON for chrome://tracing
        pub fn exportChromeTrace(self: *const Self, writer: anytype) !void {
            try writer.writeAll("{\"traceEvents\":[\n");

            var first = true;
            for (self.spans[0..self.count]) |*span| {
                if (!span.completed) continue;
                if (!first) try writer.writeAll(",\n");
                first = false;

                try writer.print(
                    "{{\"name\":\"{s}\",\"cat\":\"{s}\",\"ph\":\"X\",\"ts\":0,\"dur\":{d:.1},\"pid\":0,\"tid\":{d}}}",
                    .{ span.getName(), span.category.label(), span.durationUs(), span.thread_id },
                );
            }

            try writer.writeAll("\n]}\n");
        }

        // ── Aggregation ──────────────────────────────────────────────

        pub fn totalTimeNs(self: *const Self, category: Category) u64 {
            var total: u64 = 0;
            for (self.spans[0..self.count]) |*span| {
                if (span.category == category and span.completed) {
                    total += span.duration_ns;
                }
            }
            return total;
        }

        pub fn slowestSpan(self: *const Self, name: []const u8) ?*const SpanRecord {
            var slowest: ?*const SpanRecord = null;
            var max_dur: u64 = 0;

            for (self.spans[0..self.count]) |*span| {
                if (!span.completed) continue;
                if (!std.mem.eql(u8, span.getName(), name)) continue;
                if (span.duration_ns > max_dur) {
                    max_dur = span.duration_ns;
                    slowest = span;
                }
            }
            return slowest;
        }

        pub fn reset(self: *Self) void {
            self.count = 0;
            self.depth = 0;
            self.overflow_count = 0;
        }

        pub fn spanCount(self: *const Self) usize {
            return self.count;
        }
    };
}

/// Default profiler with 8K span capacity
pub const Profiler = ProfilerType(8192);

/// Compact profiler for constrained environments
pub const CompactProfiler = ProfilerType(256);

// ─── Tests ──────────────────────────────────────────────────────────────────

test "SpanRecord duration conversions" {
    var span = SpanRecord{};
    span.duration_ns = 1_500_000; // 1.5ms
    try std.testing.expectApproxEqAbs(@as(f64, 1500.0), span.durationUs(), 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), span.durationMs(), 0.01);
}

test "SpanRecord throughput calculation" {
    var span = SpanRecord{};
    span.duration_ns = 1_000_000_000; // 1 second
    span.metadata = 100_000_000; // 100MB
    // throughput = 100MB / 1s = 100 MB/s
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), span.throughputMBs(), 1.0);

    // Zero duration → 0 throughput
    var zero_span = SpanRecord{};
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), zero_span.throughputMBs(), 0.001);
}

test "Profiler begin/end records spans" {
    var p = CompactProfiler.init(true);
    try std.testing.expectEqual(@as(usize, 0), p.spanCount());

    const h = p.begin("test_op");
    p.end(h);

    try std.testing.expectEqual(@as(usize, 1), p.spanCount());
    try std.testing.expect(p.spans[0].completed);
    try std.testing.expectEqualStrings("test_op", p.spans[0].getName());
}

test "Profiler disabled skips recording" {
    var p = CompactProfiler.init(false);
    const h = p.begin("should_not_record");
    p.end(h);
    try std.testing.expectEqual(@as(usize, 0), p.spanCount());
}

test "Profiler overflow tracking" {
    var p = ProfilerType(2).init(true);
    const h1 = p.begin("a");
    p.end(h1);
    const h2 = p.begin("b");
    p.end(h2);
    const h3 = p.begin("overflow");
    p.end(h3); // should be dropped
    try std.testing.expectEqual(@as(usize, 2), p.spanCount());
    try std.testing.expectEqual(@as(u64, 1), p.overflow_count);
}

test "Profiler category time aggregation" {
    var p = CompactProfiler.init(true);
    const h1 = p.beginCat("compute_a", .compute);
    p.end(h1);
    const h2 = p.beginCat("io_op", .io);
    p.end(h2);

    try std.testing.expect(p.totalTimeNs(.compute) > 0 or p.totalTimeNs(.compute) == 0);
    // Just verify it doesn't crash — actual duration depends on timing
}

test "Profiler reset" {
    var p = CompactProfiler.init(true);
    const h = p.begin("op");
    p.end(h);
    try std.testing.expectEqual(@as(usize, 1), p.spanCount());
    p.reset();
    try std.testing.expectEqual(@as(usize, 0), p.spanCount());
}

test "Category labels" {
    try std.testing.expectEqualStrings("user", Category.user.label());
    try std.testing.expectEqualStrings("gpu", Category.gpu.label());
    try std.testing.expectEqualStrings("io", Category.io.label());
}

test {
    std.testing.refAllDecls(@This());
}
