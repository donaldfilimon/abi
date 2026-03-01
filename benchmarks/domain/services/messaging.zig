//! Messaging Benchmarks
//!
//! Performance measurement for the topic-based pub/sub messaging module:
//! - Publish throughput (single topic, N messages)
//! - Fanout throughput (1/10/100 subscribers on same topic)
//! - Topic matching (MQTT-style wildcard patterns: `*`, `#`)
//! - Subscribe/unsubscribe churn (rapid lifecycle cycles)

const std = @import("std");
const abi = @import("abi");
const framework = @import("../../system/framework.zig");

pub const MessagingBenchConfig = struct {
    message_counts: []const usize = &.{ 100, 1000, 10_000 },
    fanout_sizes: []const usize = &.{ 1, 10, 100 },
    churn_cycles: []const usize = &.{ 100, 1000, 10_000 },
};

// ── Helpers ──────────────────────────────────────────────────────────

const messaging_config = abi.features.messaging.MessagingConfig{
    .max_channels = 1000,
    .buffer_size = 200,
};

fn noopCallback(
    _: abi.features.messaging.Message,
    _: ?*anyopaque,
) abi.features.messaging.DeliveryResult {
    return .ok;
}

fn generateTopic(buf: *[64]u8, i: usize) []const u8 {
    const len = std.fmt.bufPrint(buf, "bench.topic.{d:0>6}", .{i}) catch
        return "bench.topic.000000";
    return len;
}

fn generatePayload(buf: *[128]u8, i: usize) []const u8 {
    const len = std.fmt.bufPrint(buf, "payload-{d:0>8}-data", .{i}) catch
        return "payload-00000000-data";
    return len;
}

// ── Publish Throughput ───────────────────────────────────────────────

fn benchPublishThroughput(
    allocator: std.mem.Allocator,
    count: usize,
) !void {
    try abi.features.messaging.init(allocator, messaging_config);
    defer abi.features.messaging.deinit();

    // Subscribe one listener so messages are actually delivered
    _ = try abi.features.messaging.subscribe(
        allocator,
        "bench.publish",
        noopCallback,
        null,
    );

    var payload_buf: [128]u8 = undefined;
    for (0..count) |i| {
        const payload = generatePayload(&payload_buf, i);
        try abi.features.messaging.publish(allocator, "bench.publish", payload);
    }
}

// ── Fanout Throughput ────────────────────────────────────────────────

fn benchFanoutThroughput(
    allocator: std.mem.Allocator,
    subscriber_count: usize,
    message_count: usize,
) !void {
    try abi.features.messaging.init(allocator, messaging_config);
    defer abi.features.messaging.deinit();

    // Register N subscribers on the same topic
    for (0..subscriber_count) |_| {
        _ = try abi.features.messaging.subscribe(
            allocator,
            "bench.fanout",
            noopCallback,
            null,
        );
    }

    var payload_buf: [128]u8 = undefined;
    for (0..message_count) |i| {
        const payload = generatePayload(&payload_buf, i);
        try abi.features.messaging.publish(allocator, "bench.fanout", payload);
    }
}

// ── Topic Matching (wildcard patterns) ───────────────────────────────

fn benchTopicMatching(
    allocator: std.mem.Allocator,
    count: usize,
) !void {
    try abi.features.messaging.init(allocator, messaging_config);
    defer abi.features.messaging.deinit();

    // Subscribe with various wildcard patterns (MQTT-style, `.` separator)
    // `*` = single-level wildcard, `#` = multi-level wildcard
    _ = try abi.features.messaging.subscribe(
        allocator,
        "sensor.*.temp",
        noopCallback,
        null,
    );
    _ = try abi.features.messaging.subscribe(
        allocator,
        "sensor.#",
        noopCallback,
        null,
    );
    _ = try abi.features.messaging.subscribe(
        allocator,
        "logs.#",
        noopCallback,
        null,
    );
    _ = try abi.features.messaging.subscribe(
        allocator,
        "events.*",
        noopCallback,
        null,
    );
    _ = try abi.features.messaging.subscribe(
        allocator,
        "*.alert",
        noopCallback,
        null,
    );

    // Publish messages that match various patterns
    const topics = [_][]const u8{
        "sensor.room1.temp",
        "sensor.room2.temp",
        "sensor.room1.humidity",
        "logs.app.error",
        "logs.system.warn.critical",
        "events.user",
        "events.order",
        "other.alert",
        "unmatched.topic",
    };

    var payload_buf: [128]u8 = undefined;
    for (0..count) |i| {
        const topic = topics[i % topics.len];
        const payload = generatePayload(&payload_buf, i);
        try abi.features.messaging.publish(allocator, topic, payload);
    }
}

// ── Subscribe/Unsubscribe Churn ──────────────────────────────────────

fn benchSubscribeChurn(
    allocator: std.mem.Allocator,
    cycles: usize,
) !void {
    try abi.features.messaging.init(allocator, messaging_config);
    defer abi.features.messaging.deinit();

    var topic_buf: [64]u8 = undefined;
    for (0..cycles) |i| {
        const topic = generateTopic(&topic_buf, i % 100);
        const sub_id = try abi.features.messaging.subscribe(
            allocator,
            topic,
            noopCallback,
            null,
        );
        const removed = try abi.features.messaging.unsubscribe(sub_id);
        std.mem.doNotOptimizeAway(&removed);
    }
}

// ── Runner ───────────────────────────────────────────────────────────

pub fn runMessagingBenchmarks(
    allocator: std.mem.Allocator,
    config: MessagingBenchConfig,
) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print(
        "                       MESSAGING BENCHMARKS\n",
        .{},
    );
    std.debug.print("=" ** 80 ++ "\n\n", .{});

    // Publish throughput
    std.debug.print("[Publish Throughput]\n", .{});
    for (config.message_counts) |count| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "publish_{d}",
            .{count},
        ) catch "publish";

        const result = try runner.run(
            .{
                .name = name,
                .category = "messaging/publish",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchPublishThroughput(a, c);
                }
            }.bench,
            .{ allocator, count },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name, result.stats.opsPerSecond(),
        });
    }

    // Fanout throughput
    std.debug.print("\n[Fanout Throughput]\n", .{});
    for (config.fanout_sizes) |fan| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "fanout_{d}sub",
            .{fan},
        ) catch "fanout";

        const result = try runner.run(
            .{
                .name = name,
                .category = "messaging/fanout",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, f: usize) !void {
                    try benchFanoutThroughput(a, f, 1000);
                }
            }.bench,
            .{ allocator, fan },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name, result.stats.opsPerSecond(),
        });
    }

    // Topic matching (wildcard patterns)
    std.debug.print("\n[Topic Matching (wildcard patterns)]\n", .{});
    for (config.message_counts) |count| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "wildcard_{d}",
            .{count},
        ) catch "wildcard";

        const result = try runner.run(
            .{
                .name = name,
                .category = "messaging/matching",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchTopicMatching(a, c);
                }
            }.bench,
            .{ allocator, count },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name, result.stats.opsPerSecond(),
        });
    }

    // Subscribe/unsubscribe churn
    std.debug.print("\n[Subscribe/Unsubscribe Churn]\n", .{});
    for (config.churn_cycles) |cycles| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "churn_{d}",
            .{cycles},
        ) catch "churn";

        const result = try runner.run(
            .{
                .name = name,
                .category = "messaging/churn",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchSubscribeChurn(a, c);
                }
            }.bench,
            .{ allocator, cycles },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name, result.stats.opsPerSecond(),
        });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

pub fn run(allocator: std.mem.Allocator) !void {
    try runMessagingBenchmarks(allocator, .{});
}

test "messaging benchmarks compile" {
    const allocator = std.testing.allocator;
    try benchPublishThroughput(allocator, 10);
    try benchFanoutThroughput(allocator, 3, 10);
    try benchTopicMatching(allocator, 20);
    try benchSubscribeChurn(allocator, 10);
}
