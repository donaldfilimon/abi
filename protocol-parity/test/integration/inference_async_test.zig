//! Deeper public async integration coverage for inference.

const std = @import("std");
const abi = @import("abi");

const Engine = abi.inference.Engine;

const ShutdownCallbackHelper = struct {
    var done: std.atomic.Value(bool) = .{ .raw = false };
    var saw_text: std.atomic.Value(bool) = .{ .raw = false };

    fn cb(result: abi.inference.EngineResult) void {
        saw_text.store(result.text.len > 0, .release);
        done.store(true, .release);
    }
};

test "inference async: waitTimeout returns owned results the caller must clean up" {
    var engine = try Engine.init(std.testing.allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });
    defer engine.deinit();

    const ar = try engine.generateAsyncWithTimeout(.{
        .id = 501,
        .prompt = "owned result contract",
        .max_tokens = 16,
    });
    defer ar.destroy();

    const maybe_result = ar.waitTimeout(5000);
    try std.testing.expect(maybe_result != null);

    var result = maybe_result.?;
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u64, 501), result.id);
    try std.testing.expect(result.text_owned);
    try std.testing.expect(result.tokens_owned);
    try std.testing.expect(result.text.len > 0);
}

test "inference async: timeout then abandon leaves cleanup to the background thread" {
    var engine = try Engine.init(std.testing.allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });

    const ar = try engine.generateAsyncWithTimeout(.{
        .id = 502,
        .prompt = "timeout abandon cleanup",
        .max_tokens = 50_000,
    });

    try std.testing.expect(ar.waitTimeout(0) == null);
    ar.deinit();
    engine.deinit();
}

test "inference async: deinit waits for in-flight async callbacks" {
    ShutdownCallbackHelper.done.store(false, .release);
    ShutdownCallbackHelper.saw_text.store(false, .release);

    var engine = try Engine.init(std.testing.allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 4,
        .vocab_size = 256,
    });

    try engine.generateAsync(.{
        .id = 503,
        .prompt = "shutdown waits for async jobs",
        .max_tokens = 20_000,
    }, ShutdownCallbackHelper.cb);

    engine.deinit();

    try std.testing.expect(ShutdownCallbackHelper.done.load(.acquire));
    try std.testing.expect(ShutdownCallbackHelper.saw_text.load(.acquire));
}

test {
    std.testing.refAllDecls(@This());
}
