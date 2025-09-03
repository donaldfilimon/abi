const std = @import("std");
const testing = std.testing;
const plugin = @import("discord_plugin");

// Note: This test only verifies compilation and basic plugin interface
// It doesn't test actual Discord API functionality since that requires tokens

test "discord plugin compilation" {
    // Test that the plugin compiles without errors

    // Test that we can access the plugin interface
    const interface = plugin.abi_plugin_create();
    try testing.expect(interface != null);

    const info = interface.?.get_info();
    try testing.expect(info.name.len > 0);
    try testing.expectEqualStrings("discord_integration", info.name);
}

test "websocket transport initialization" {
    const allocator = testing.allocator;
    var transport = plugin.WebSocketTransport.init(allocator);
    defer transport.deinit();

    // Test basic fields
    try testing.expect(!transport.connected);
    try testing.expect(transport.heartbeat_interval_ms > 0);
    try testing.expect(transport.last_sequence == null);
    try testing.expect(transport.max_reconnect_attempts == 5);
}

test "exponential backoff calculation" {
    const allocator = testing.allocator;
    var transport = plugin.WebSocketTransport.init(allocator);
    defer transport.deinit();

    // Test backoff delays increase exponentially
    const delay1 = transport.calculateBackoffDelay(0);
    const delay2 = transport.calculateBackoffDelay(1);
    const delay3 = transport.calculateBackoffDelay(2);

    // Test that delays are reasonable and increase
    try testing.expect(delay1 >= 1000 and delay1 <= 1500); // ~1s base + jitter
    try testing.expect(delay2 >= 1500 and delay2 <= 3000); // ~2s base + jitter
    try testing.expect(delay3 >= 2000 and delay3 <= 5000); // ~4s base + jitter

    // Test maximum delay cap
    const delay_high = transport.calculateBackoffDelay(10);
    try testing.expect(delay_high <= 30000); // Should not exceed 30s

    // Test that delays generally increase with attempt number
    try testing.expect(delay2 >= delay1);
    try testing.expect(delay3 >= delay2);
}

test "command spec structure" {
    const cmd = plugin.CommandSpec{
        .name = "test",
        .description = "Test command",
    };

    try testing.expectEqualStrings("test", cmd.name);
    try testing.expectEqualStrings("Test command", cmd.description);
}

test "rest client error types" {
    // Test that error types are defined
    const RestError = plugin.DiscordRestClient.RestError;

    // Should be able to use error types
    const conflict_err: RestError = RestError.HttpConflict;
    const rate_limited_err: RestError = RestError.RateLimited;

    try testing.expect(conflict_err == RestError.HttpConflict);
    try testing.expect(rate_limited_err == RestError.RateLimited);
}

test "log functions" {
    // Test that log functions are accessible (they just print to debug)
    plugin.Log.info("Test info log", .{});
    plugin.Log.warn("Test warn log", .{});
    plugin.Log.err("Test error log", .{});
}

test "json content body builder" {
    var allocator = testing.allocator;
    const json = try plugin.buildJsonContentBody(allocator, "Hello World");
    defer allocator.free(json);

    try testing.expectEqualStrings("{\"content\":\"Hello World\"}", json);
}
