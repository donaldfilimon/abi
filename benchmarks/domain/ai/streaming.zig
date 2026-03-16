//! Streaming Inference Benchmarks
//!
//! Measures performance characteristics of the streaming inference pipeline:
//!
//! - **TTFT (Time To First Token)**: Latency until first token arrives
//! - **Inter-token latency**: Time between consecutive tokens (P50/P90/P99)
//! - **Throughput**: Tokens per second generation rate
//! - **SSE encoding overhead**: Cost of Server-Sent Events formatting
//!
//! Supports both mock token generation (for overhead isolation) and real
//! model inference (for end-to-end measurements).
//!
//! ## Usage
//!
//! ```zig
//! const streaming = @import("streaming.zig");
//!
//! // Run with default config
//! try streaming.runStreamingBenchmarks(allocator, .standard);
//!
//! // Run with custom config
//! try streaming.runStreamingBenchmarks(allocator, .comprehensive);
//! ```

const std = @import("std");
const abi = @import("abi");
const framework = @import("../../system/framework.zig");

/// Configuration for streaming benchmarks
pub const StreamingBenchConfig = struct {
    /// Number of tokens to generate per run
    tokens_per_run: []const usize = &.{ 32, 64, 128, 256, 512 },
    /// Number of warmup iterations
    warmup_iterations: usize = 10,
    /// Number of benchmark iterations per configuration
    iterations: usize = 100,
    /// Token generation patterns to test
    patterns: []const GenerationPattern = &.{ .constant_rate, .variable_rate, .burst, .warmup },
    /// Simulated token delay range (nanoseconds) for mock generator
    min_token_delay_ns: u64 = 1_000_000, // 1ms
    max_token_delay_ns: u64 = 50_000_000, // 50ms
    /// Whether to benchmark SSE encoding overhead
    bench_sse_encoding: bool = true,
    /// Whether to benchmark WebSocket framing overhead
    bench_ws_framing: bool = true,
    /// Random seed for reproducibility
    seed: u64 = 42,

    /// Quick configuration for CI
    pub const quick = StreamingBenchConfig{
        .tokens_per_run = &.{ 32, 64 },
        .warmup_iterations = 5,
        .iterations = 50,
        .patterns = &.{ .constant_rate, .variable_rate },
        .min_token_delay_ns = 1_000_000,
        .max_token_delay_ns = 10_000_000,
    };

    /// Standard configuration for development
    pub const standard = StreamingBenchConfig{
        .tokens_per_run = &.{ 32, 64, 128, 256 },
        .warmup_iterations = 10,
        .iterations = 100,
        .patterns = &.{ .constant_rate, .variable_rate, .burst },
        .min_token_delay_ns = 1_000_000,
        .max_token_delay_ns = 30_000_000,
    };

    /// Comprehensive configuration for full benchmarking
    pub const comprehensive = StreamingBenchConfig{
        .tokens_per_run = &.{ 32, 64, 128, 256, 512, 1024 },
        .warmup_iterations = 20,
        .iterations = 200,
        .patterns = &.{ .constant_rate, .variable_rate, .burst, .warmup },
        .min_token_delay_ns = 500_000,
        .max_token_delay_ns = 100_000_000,
        .bench_sse_encoding = true,
        .bench_ws_framing = true,
    };
};

/// Token generation pattern for mock generator
pub const GenerationPattern = enum {
    /// Constant delay between tokens
    constant_rate,
    /// Variable delay (uniform random within range)
    variable_rate,
    /// Burst: fast tokens followed by pause
    burst,
    /// Warmup: slow start, then faster
    warmup,

    pub fn name(self: GenerationPattern) []const u8 {
        return switch (self) {
            .constant_rate => "constant_rate",
            .variable_rate => "variable_rate",
            .burst => "burst",
            .warmup => "warmup",
        };
    }
};

/// Results from streaming benchmark
pub const StreamingBenchResult = struct {
    /// Configuration used
    pattern: GenerationPattern,
    tokens_generated: usize,
    iterations: usize,

    /// Time to first token statistics (nanoseconds)
    ttft_min_ns: u64,
    ttft_max_ns: u64,
    ttft_mean_ns: f64,
    ttft_p50_ns: u64,
    ttft_p90_ns: u64,
    ttft_p99_ns: u64,

    /// Inter-token latency statistics (nanoseconds)
    itl_min_ns: u64,
    itl_max_ns: u64,
    itl_mean_ns: f64,
    itl_p50_ns: u64,
    itl_p90_ns: u64,
    itl_p99_ns: u64,

    /// Throughput (tokens per second)
    throughput_mean: f64,
    throughput_p50: f64,
    throughput_p99: f64,

    /// Total generation time (nanoseconds)
    total_time_ns: u64,

    pub fn format(
        self: StreamingBenchResult,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print(
            "StreamingBenchResult{{ pattern={s}, tokens={d}, ttft_p50={d}ms, itl_p50={d}ms, throughput={d:.1} tok/s }}",
            .{
                self.pattern.name(),
                self.tokens_generated,
                self.ttft_p50_ns / 1_000_000,
                self.itl_p50_ns / 1_000_000,
                self.throughput_mean,
            },
        );
    }
};

/// SSE encoding overhead result
pub const SseOverheadResult = struct {
    /// Tokens encoded
    tokens_encoded: usize,
    /// Mean encoding time per token (nanoseconds)
    encode_time_mean_ns: f64,
    /// Total bytes written
    bytes_written: usize,
    /// Overhead ratio (encoded size / raw size)
    overhead_ratio: f64,
};

/// WebSocket framing overhead result
pub const WsFramingResult = struct {
    /// Messages framed
    messages_framed: usize,
    /// Mean framing time per message (nanoseconds)
    frame_time_mean_ns: f64,
    /// Total bytes with framing
    bytes_with_framing: usize,
    /// Overhead ratio
    overhead_ratio: f64,
};

/// Mock token generator for benchmarking
pub const MockTokenGenerator = struct {
    allocator: std.mem.Allocator,
    pattern: GenerationPattern,
    prng: std.Random.Xoroshiro128,
    min_delay_ns: u64,
    max_delay_ns: u64,
    tokens_generated: usize,
    total_tokens: usize,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        pattern: GenerationPattern,
        total_tokens: usize,
        seed: u64,
        min_delay_ns: u64,
        max_delay_ns: u64,
    ) Self {
        return .{
            .allocator = allocator,
            .pattern = pattern,
            .prng = std.Random.Xoroshiro128.init(seed),
            .min_delay_ns = min_delay_ns,
            .max_delay_ns = max_delay_ns,
            .tokens_generated = 0,
            .total_tokens = total_tokens,
        };
    }

    /// Generate next token with simulated delay
    /// Returns null when generation is complete
    pub fn next(self: *Self) ?MockToken {
        if (self.tokens_generated >= self.total_tokens) {
            return null;
        }

        const delay = self.calculateDelay();
        self.simulateDelay(delay);

        const token = MockToken{
            .id = @intCast(self.tokens_generated),
            .text = self.generateTokenText(),
            .delay_ns = delay,
        };

        self.tokens_generated += 1;
        return token;
    }

    fn calculateDelay(self: *Self) u64 {
        const rand = self.prng.random();
        const progress = @as(f64, @floatFromInt(self.tokens_generated)) /
            @as(f64, @floatFromInt(self.total_tokens));

        return switch (self.pattern) {
            .constant_rate => (self.min_delay_ns + self.max_delay_ns) / 2,

            .variable_rate => blk: {
                const range = self.max_delay_ns - self.min_delay_ns;
                const random_offset = rand.uintLessThan(u64, range + 1);
                break :blk self.min_delay_ns + random_offset;
            },

            .burst => blk: {
                // Every 8 tokens, insert a longer pause
                if (self.tokens_generated % 8 == 7) {
                    break :blk self.max_delay_ns;
                }
                break :blk self.min_delay_ns;
            },

            .warmup => blk: {
                // Start slow, get faster
                const slowdown = 1.0 - progress;
                const range = @as(f64, @floatFromInt(self.max_delay_ns - self.min_delay_ns));
                const delay_offset = @as(u64, @intFromFloat(range * slowdown));
                break :blk self.min_delay_ns + delay_offset;
            },
        };
    }

    fn simulateDelay(_: *Self, delay_ns: u64) void {
        var timer = abi.services.shared.time.Timer.start() catch return;

        if (delay_ns < 1_000_000) {
            // For sub-millisecond delays, use tight busy-wait
            while (timer.read() < delay_ns) {
                std.atomic.spinLoopHint();
            }
        } else {
            // For longer delays, yield periodically to reduce CPU load
            while (timer.read() < delay_ns) {
                var yield_count: usize = 0;
                while (yield_count < 100) : (yield_count += 1) {
                    std.atomic.spinLoopHint();
                }
            }
        }
    }

    fn generateTokenText(self: *Self) []const u8 {
        // Generate realistic token lengths (1-8 chars typically)
        const sample_tokens = [_][]const u8{
            " the",
            " a",
            " is",
            " and",
            " to",
            " of",
            " in",
            " that",
            " it",
            " for",
            " was",
            " on",
            " are",
            " as",
            " with",
            ".",
            ",",
            "!",
            "?",
            "\n",
            " Hello",
            " world",
            " AI",
            " model",
            " token",
        };

        const idx = self.prng.random().uintLessThan(usize, sample_tokens.len);
        return sample_tokens[idx];
    }

    pub fn reset(self: *Self, seed: u64) void {
        self.prng = std.Random.Xoroshiro128.init(seed);
        self.tokens_generated = 0;
    }
};

/// Mock token structure
pub const MockToken = struct {
    id: u32,
    text: []const u8,
    delay_ns: u64,
};

/// Run streaming benchmarks with the given configuration preset
pub fn runStreamingBenchmarks(allocator: std.mem.Allocator, preset: ConfigPreset) !void {
    const config = switch (preset) {
        .quick => StreamingBenchConfig.quick,
        .standard => StreamingBenchConfig.standard,
        .comprehensive => StreamingBenchConfig.comprehensive,
    };

    try runStreamingBenchmarksWithConfig(allocator, config);
}

/// Configuration preset enum
pub const ConfigPreset = enum {
    quick,
    standard,
    comprehensive,
};

/// Run streaming benchmarks with custom configuration
pub fn runStreamingBenchmarksWithConfig(allocator: std.mem.Allocator, config: StreamingBenchConfig) !void {
    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    STREAMING INFERENCE BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n", .{});

    // Run token generation benchmarks for each pattern and token count
    for (config.patterns) |pattern| {
        std.debug.print("\n[Pattern: {s}]\n", .{pattern.name()});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});

        for (config.tokens_per_run) |token_count| {
            const result = try benchmarkTokenGeneration(
                allocator,
                pattern,
                token_count,
                config.iterations,
                config.warmup_iterations,
                config.seed,
                config.min_token_delay_ns,
                config.max_token_delay_ns,
            );

            printTokenGenerationResult(result);
        }
    }

    // Run SSE encoding overhead benchmark
    if (config.bench_sse_encoding) {
        std.debug.print("\n[SSE Encoding Overhead]\n", .{});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});

        for (config.tokens_per_run) |token_count| {
            const result = try benchmarkSseEncoding(allocator, token_count, config.iterations);
            printSseOverheadResult(result);
        }
    }

    // Run WebSocket framing overhead benchmark
    if (config.bench_ws_framing) {
        std.debug.print("\n[WebSocket Framing Overhead]\n", .{});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});

        for (config.tokens_per_run) |token_count| {
            const result = try benchmarkWsFraming(allocator, token_count, config.iterations);
            printWsFramingResult(result);
        }
    }

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("                    STREAMING BENCHMARKS COMPLETE\n", .{});
    std.debug.print("================================================================================\n", .{});
}

/// Benchmark token generation for a specific pattern
fn benchmarkTokenGeneration(
    allocator: std.mem.Allocator,
    pattern: GenerationPattern,
    token_count: usize,
    iterations: usize,
    warmup_iterations: usize,
    seed: u64,
    min_delay_ns: u64,
    max_delay_ns: u64,
) !StreamingBenchResult {
    var ttft_samples = std.ArrayListUnmanaged(u64).empty;
    defer ttft_samples.deinit(allocator);

    var itl_samples = std.ArrayListUnmanaged(u64).empty;
    defer itl_samples.deinit(allocator);

    var throughput_samples = std.ArrayListUnmanaged(f64).empty;
    defer throughput_samples.deinit(allocator);

    var total_time: u64 = 0;

    // Warmup phase
    {
        var warmup_gen = MockTokenGenerator.init(
            allocator,
            pattern,
            token_count,
            seed,
            min_delay_ns,
            max_delay_ns,
        );

        for (0..warmup_iterations) |w| {
            warmup_gen.reset(seed +% w);
            while (warmup_gen.next()) |token| {
                std.mem.doNotOptimizeAway(&token);
            }
        }
    }

    // Benchmark iterations
    for (0..iterations) |iter| {
        var gen = MockTokenGenerator.init(
            allocator,
            pattern,
            token_count,
            seed +% iter,
            min_delay_ns,
            max_delay_ns,
        );

        var run_timer = abi.services.shared.time.Timer.start() catch continue;
        var prev_token_time: u64 = 0;
        var first_token = true;

        while (gen.next()) |token| {
            const current_time = run_timer.read();

            if (first_token) {
                try ttft_samples.append(allocator, current_time);
                first_token = false;
            } else {
                const itl = current_time - prev_token_time;
                try itl_samples.append(allocator, itl);
            }

            prev_token_time = current_time;
            std.mem.doNotOptimizeAway(&token);
        }

        const run_time = run_timer.read();
        total_time += run_time;

        // Calculate throughput for this run
        const throughput = @as(f64, @floatFromInt(token_count)) /
            (@as(f64, @floatFromInt(run_time)) / 1_000_000_000.0);
        try throughput_samples.append(allocator, throughput);
    }

    // Calculate statistics
    const ttft_stats = calculateStats(allocator, ttft_samples.items) catch StatResult{};
    const itl_stats = calculateStats(allocator, itl_samples.items) catch StatResult{};
    const throughput_stats = calculateFloatStats(allocator, throughput_samples.items) catch FloatStatResult{};

    return StreamingBenchResult{
        .pattern = pattern,
        .tokens_generated = token_count,
        .iterations = iterations,
        .ttft_min_ns = ttft_stats.min,
        .ttft_max_ns = ttft_stats.max,
        .ttft_mean_ns = ttft_stats.mean,
        .ttft_p50_ns = ttft_stats.p50,
        .ttft_p90_ns = ttft_stats.p90,
        .ttft_p99_ns = ttft_stats.p99,
        .itl_min_ns = itl_stats.min,
        .itl_max_ns = itl_stats.max,
        .itl_mean_ns = itl_stats.mean,
        .itl_p50_ns = itl_stats.p50,
        .itl_p90_ns = itl_stats.p90,
        .itl_p99_ns = itl_stats.p99,
        .throughput_mean = throughput_stats.mean,
        .throughput_p50 = throughput_stats.p50,
        .throughput_p99 = throughput_stats.p99,
        .total_time_ns = total_time,
    };
}

/// Benchmark SSE encoding overhead
fn benchmarkSseEncoding(allocator: std.mem.Allocator, token_count: usize, iterations: usize) !SseOverheadResult {
    // Sample tokens for encoding
    const sample_tokens = [_][]const u8{
        " the", " a", " is", " and", " to", " of", " in", " that", " it", " for",
    };

    var total_encode_time: u64 = 0;
    var total_raw_bytes: usize = 0;
    var total_encoded_bytes: usize = 0;

    var buffer = std.ArrayListUnmanaged(u8).empty;
    defer buffer.deinit(allocator);

    for (0..iterations) |_| {
        buffer.clearRetainingCapacity();

        const timer = abi.services.shared.time.Timer.start() catch continue;

        for (0..token_count) |i| {
            const token = sample_tokens[i % sample_tokens.len];
            total_raw_bytes += token.len;

            // SSE format: data: {"token":"<text>","index":<n>}\n\n
            try buffer.appendSlice(allocator, "data: {\"token\":\"");
            try buffer.appendSlice(allocator, token);
            try buffer.appendSlice(allocator, "\",\"index\":");

            var idx_buf: [16]u8 = undefined;
            const idx_str = std.fmt.bufPrint(&idx_buf, "{d}", .{i}) catch "0";
            try buffer.appendSlice(allocator, idx_str);
            try buffer.appendSlice(allocator, "}\n\n");
        }

        var t = timer;
        total_encode_time += t.read();
        total_encoded_bytes += buffer.items.len;
    }

    const tokens_total = token_count * iterations;
    const mean_encode_time = @as(f64, @floatFromInt(total_encode_time)) /
        @as(f64, @floatFromInt(tokens_total));

    return SseOverheadResult{
        .tokens_encoded = tokens_total,
        .encode_time_mean_ns = mean_encode_time,
        .bytes_written = total_encoded_bytes,
        .overhead_ratio = @as(f64, @floatFromInt(total_encoded_bytes)) /
            @as(f64, @floatFromInt(total_raw_bytes)),
    };
}

/// Benchmark WebSocket framing overhead
fn benchmarkWsFraming(allocator: std.mem.Allocator, message_count: usize, iterations: usize) !WsFramingResult {
    // Sample messages
    const sample_messages = [_][]const u8{
        "{\"type\":\"token\",\"text\":\" the\"}",
        "{\"type\":\"token\",\"text\":\" a\"}",
        "{\"type\":\"token\",\"text\":\" is\"}",
        "{\"type\":\"start\",\"model\":\"test\"}",
        "{\"type\":\"end\",\"finish_reason\":\"stop\"}",
    };

    var total_frame_time: u64 = 0;
    var total_raw_bytes: usize = 0;
    var total_framed_bytes: usize = 0;

    var buffer = std.ArrayListUnmanaged(u8).empty;
    defer buffer.deinit(allocator);

    for (0..iterations) |_| {
        buffer.clearRetainingCapacity();

        const timer = abi.services.shared.time.Timer.start() catch continue;

        for (0..message_count) |i| {
            const message = sample_messages[i % sample_messages.len];
            total_raw_bytes += message.len;

            // Simulate WebSocket framing (simplified binary frame)
            // Byte 0: 0x81 (text frame, FIN=1)
            // Byte 1: length (assuming < 126 bytes)
            // Payload follows
            try buffer.append(allocator, 0x81);
            try buffer.append(allocator, @intCast(message.len));
            try buffer.appendSlice(allocator, message);
        }

        var t = timer;
        total_frame_time += t.read();
        total_framed_bytes += buffer.items.len;
    }

    const messages_total = message_count * iterations;
    const mean_frame_time = @as(f64, @floatFromInt(total_frame_time)) /
        @as(f64, @floatFromInt(messages_total));

    return WsFramingResult{
        .messages_framed = messages_total,
        .frame_time_mean_ns = mean_frame_time,
        .bytes_with_framing = total_framed_bytes,
        .overhead_ratio = @as(f64, @floatFromInt(total_framed_bytes)) /
            @as(f64, @floatFromInt(total_raw_bytes)),
    };
}

// Statistics helpers

const StatResult = struct {
    min: u64 = 0,
    max: u64 = 0,
    mean: f64 = 0,
    p50: u64 = 0,
    p90: u64 = 0,
    p99: u64 = 0,
};

fn calculateStats(allocator: std.mem.Allocator, samples: []const u64) !StatResult {
    if (samples.len == 0) return StatResult{};

    const sorted = try allocator.alloc(u64, samples.len);
    defer allocator.free(sorted);
    @memcpy(sorted, samples);
    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));

    var sum: u128 = 0;
    for (sorted) |s| sum += s;

    return StatResult{
        .min = sorted[0],
        .max = sorted[sorted.len - 1],
        .mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(sorted.len)),
        .p50 = sorted[@min(sorted.len * 50 / 100, sorted.len - 1)],
        .p90 = sorted[@min(sorted.len * 90 / 100, sorted.len - 1)],
        .p99 = sorted[@min(sorted.len * 99 / 100, sorted.len - 1)],
    };
}

const FloatStatResult = struct {
    mean: f64 = 0,
    p50: f64 = 0,
    p99: f64 = 0,
};

fn calculateFloatStats(allocator: std.mem.Allocator, samples: []const f64) !FloatStatResult {
    if (samples.len == 0) return FloatStatResult{};

    var sum: f64 = 0;
    for (samples) |s| sum += s;
    const mean = sum / @as(f64, @floatFromInt(samples.len));

    // Sort for proper percentile calculation
    var sorted = try allocator.alloc(f64, samples.len);
    defer allocator.free(sorted);
    @memcpy(sorted, samples);

    // Simple insertion sort for small sample sizes (typically < 1000)
    for (1..sorted.len) |i| {
        const key = sorted[i];
        var j: usize = i;
        while (j > 0 and sorted[j - 1] > key) : (j -= 1) {
            sorted[j] = sorted[j - 1];
        }
        sorted[j] = key;
    }

    return FloatStatResult{
        .mean = mean,
        .p50 = sorted[@min(sorted.len * 50 / 100, sorted.len - 1)],
        .p99 = sorted[@min(sorted.len * 99 / 100, sorted.len - 1)],
    };
}

// Output formatting

fn printTokenGenerationResult(result: StreamingBenchResult) void {
    const ttft_ms = @as(f64, @floatFromInt(result.ttft_p50_ns)) / 1_000_000.0;
    const itl_ms = @as(f64, @floatFromInt(result.itl_p50_ns)) / 1_000_000.0;

    std.debug.print("  tokens={d:>4}: TTFT={d:>6.1}ms  ITL_p50={d:>5.1}ms  ITL_p99={d:>6.1}ms  throughput={d:>6.1} tok/s\n", .{
        result.tokens_generated,
        ttft_ms,
        itl_ms,
        @as(f64, @floatFromInt(result.itl_p99_ns)) / 1_000_000.0,
        result.throughput_mean,
    });
}

fn printSseOverheadResult(result: SseOverheadResult) void {
    std.debug.print("  tokens={d:>6}: encode={d:>6.0}ns/tok  overhead={d:.2}x  ({d} bytes)\n", .{
        result.tokens_encoded,
        result.encode_time_mean_ns,
        result.overhead_ratio,
        result.bytes_written,
    });
}

fn printWsFramingResult(result: WsFramingResult) void {
    std.debug.print("  msgs={d:>6}: frame={d:>6.0}ns/msg  overhead={d:.2}x  ({d} bytes)\n", .{
        result.messages_framed,
        result.frame_time_mean_ns,
        result.overhead_ratio,
        result.bytes_with_framing,
    });
}

// Tests

test "mock token generator constant rate" {
    const allocator = std.testing.allocator;

    var gen = MockTokenGenerator.init(
        allocator,
        .constant_rate,
        10,
        42,
        100_000, // 0.1ms
        100_000,
    );

    var count: usize = 0;
    while (gen.next()) |_| {
        count += 1;
    }

    try std.testing.expectEqual(@as(usize, 10), count);
}

test "mock token generator variable rate" {
    const allocator = std.testing.allocator;

    var gen = MockTokenGenerator.init(
        allocator,
        .variable_rate,
        5,
        123,
        10_000, // 0.01ms
        50_000, // 0.05ms
    );

    var count: usize = 0;
    while (gen.next()) |_| {
        count += 1;
    }

    try std.testing.expectEqual(@as(usize, 5), count);
}

test "sse encoding overhead" {
    const allocator = std.testing.allocator;

    const result = try benchmarkSseEncoding(allocator, 10, 1);

    try std.testing.expect(result.tokens_encoded == 10);
    try std.testing.expect(result.overhead_ratio > 1.0);
    try std.testing.expect(result.bytes_written > 0);
}

test "ws framing overhead" {
    const allocator = std.testing.allocator;

    const result = try benchmarkWsFraming(allocator, 10, 1);

    try std.testing.expect(result.messages_framed == 10);
    try std.testing.expect(result.overhead_ratio > 1.0);
    try std.testing.expect(result.bytes_with_framing > 0);
}

test "calculate stats" {
    const allocator = std.testing.allocator;
    const samples = [_]u64{ 100, 200, 150, 300, 250 };

    const stats = try calculateStats(allocator, &samples);

    try std.testing.expectEqual(@as(u64, 100), stats.min);
    try std.testing.expectEqual(@as(u64, 300), stats.max);
    try std.testing.expect(stats.mean > 0);
}

test "streaming bench config presets" {
    try std.testing.expect(StreamingBenchConfig.quick.iterations < StreamingBenchConfig.standard.iterations);
    try std.testing.expect(StreamingBenchConfig.standard.iterations < StreamingBenchConfig.comprehensive.iterations);
}
