const std = @import("std");
const streaming = @import("../streaming.zig");
const main_generator = @import("../generator.zig");

test "streaming generator init" {
    const allocator = std.testing.allocator;

    var gen = streaming.StreamingGenerator.init(allocator, .{});
    defer gen.deinit();

    try std.testing.expectEqual(streaming.StreamingState.idle, gen.getState());
}

test "streaming stats calculation" {
    const stats = streaming.StreamingStats{
        .tokens_generated = 100,
        .prefill_time_ns = 50_000_000, // 50ms
        .generation_time_ns = 1_000_000_000, // 1s
        .time_to_first_token_ns = 100_000_000, // 100ms
        .prompt_tokens = 50,
    };

    // 100 tokens / 1 second = 100 tok/s
    try std.testing.expectEqual(@as(f64, 100.0), stats.tokensPerSecond());
    // 100ms
    try std.testing.expectEqual(@as(f64, 100.0), stats.timeToFirstTokenMs());
}

test "sse formatter" {
    const allocator = std.testing.allocator;

    const event = streaming.TokenEvent{
        .token_id = 42,
        .text = "hello",
        .position = 10,
        .is_final = false,
        .timestamp_ns = 1000,
    };

    const sse = try streaming.SSEFormatter.formatTokenEvent(allocator, event);
    defer allocator.free(sse);

    try std.testing.expect(std.mem.indexOf(u8, sse, "data:") != null);
    try std.testing.expect(std.mem.indexOf(u8, sse, "\"token_id\":42") != null);
    try std.testing.expect(std.mem.indexOf(u8, sse, "\"text\":\"hello\"") != null);
}

test "cancellation" {
    const allocator = std.testing.allocator;

    var gen = streaming.StreamingGenerator.init(allocator, .{});
    defer gen.deinit();

    try std.testing.expect(!gen.isCancelled());
    gen.cancel();
    try std.testing.expect(gen.isCancelled());
}

test "streaming config defaults" {
    const config = streaming.StreamingConfig{};

    try std.testing.expectEqual(@as(u32, 256), config.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.7), config.temperature);
    try std.testing.expectEqual(@as(u32, 40), config.top_k);
    try std.testing.expectEqual(@as(f32, 0.9), config.top_p);
    try std.testing.expectEqual(@as(u32, 256), config.initial_buffer_capacity);
    try std.testing.expect(config.decode_tokens);
}

test "streaming config from generator config" {
    const gen_config = main_generator.GeneratorConfig{
        .max_tokens = 128,
        .temperature = 0.7,
    };

    const stream_config = streaming.StreamingConfig.fromGeneratorConfig(gen_config);
    try std.testing.expectEqual(@as(u32, 128), stream_config.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.7), stream_config.temperature);
}

test "streaming config to generator config" {
    const stream_config = streaming.StreamingConfig{
        .max_tokens = 512,
        .temperature = 0.8,
    };

    const gen_config = stream_config.toGeneratorConfig();
    try std.testing.expectEqual(@as(u32, 512), gen_config.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.8), gen_config.temperature);
}

test "token event creation" {
    const event = streaming.TokenEvent{
        .token_id = 123,
        .text = "world",
        .position = 5,
        .is_final = false,
        .timestamp_ns = 0,
    };
    try std.testing.expectEqual(@as(u32, 123), event.token_id);
    try std.testing.expectEqualStrings("world", event.text.?);
    try std.testing.expectEqual(@as(u32, 5), event.position);
    try std.testing.expect(!event.is_final);
}

test "streaming state transitions" {
    const allocator = std.testing.allocator;
    var gen = streaming.StreamingGenerator.init(allocator, .{});
    defer gen.deinit();

    try std.testing.expectEqual(streaming.StreamingState.idle, gen.getState());

    // Manual transition for test
    gen.state = .generating;
    try std.testing.expectEqual(streaming.StreamingState.generating, gen.getState());
}

test "sse formatter with special characters" {
    const allocator = std.testing.allocator;

    const event = streaming.TokenEvent{
        .token_id = 1,
        .text = "line1\nline2\"quote\"",
        .position = 0,
        .is_final = false,
        .timestamp_ns = 0,
    };

    const sse = try streaming.SSEFormatter.formatTokenEvent(allocator, event);
    defer allocator.free(sse);

    try std.testing.expect(std.mem.indexOf(u8, sse, "line1\\nline2") != null);
    try std.testing.expect(std.mem.indexOf(u8, sse, "\\\"quote\\\"") != null);
}

test "sse formatter completion event" {
    const allocator = std.testing.allocator;

    const stats = streaming.StreamingStats{
        .tokens_generated = 10,
        .prefill_time_ns = 100,
        .generation_time_ns = 1000,
        .time_to_first_token_ns = 50,
        .prompt_tokens = 5,
    };

    const sse = try streaming.SSEFormatter.formatCompletion(allocator, stats);
    defer allocator.free(sse);

    try std.testing.expect(std.mem.indexOf(u8, sse, "\"event\":\"complete\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, sse, "\"tokens_generated\":10") != null);
}

test "streaming callbacks struct" {
    const Callbacks = struct {
        var call_count: usize = 0;
        fn onToken(_: streaming.TokenEvent) void {
            call_count += 1;
        }
    };

    const cb = streaming.StreamingCallbacks{
        .on_token = Callbacks.onToken,
    };

    const event = streaming.TokenEvent{
        .token_id = 1,
        .text = "test",
        .position = 0,
        .is_final = false,
        .timestamp_ns = 0,
    };
    cb.on_token.?(event);

    try std.testing.expectEqual(@as(usize, 1), Callbacks.call_count);
}
