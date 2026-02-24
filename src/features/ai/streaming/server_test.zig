const std = @import("std");
const server = @import("server.zig");
const ServerConfig = server.ServerConfig;
const StreamingServerError = server.StreamingServerError;
const request_types = @import("request_types.zig");
const parseAbiStreamRequest = request_types.parseAbiStreamRequest;
const extractJsonString = request_types.extractJsonString;
const extractJsonInt = request_types.extractJsonInt;
const splitTarget = server.splitTarget;
const backends = @import("backends/mod.zig");
const websocket = @import("websocket.zig");

test "streaming server config defaults" {
    const config = ServerConfig{};
    try std.testing.expectEqualStrings("127.0.0.1:8080", config.address);
    try std.testing.expect(config.auth_token == null);
    try std.testing.expect(config.enable_openai_compat);
    try std.testing.expect(config.enable_websocket);
}

test "heartbeat configuration" {
    // Default heartbeat interval is 15 seconds
    const default_config = ServerConfig{};
    try std.testing.expectEqual(@as(u64, 15000), default_config.heartbeat_interval_ms);

    // Custom heartbeat interval
    const custom_config = ServerConfig{ .heartbeat_interval_ms = 5000 };
    try std.testing.expectEqual(@as(u64, 5000), custom_config.heartbeat_interval_ms);

    // Heartbeat can be disabled by setting to 0
    const disabled_config = ServerConfig{ .heartbeat_interval_ms = 0 };
    try std.testing.expectEqual(@as(u64, 0), disabled_config.heartbeat_interval_ms);
}

test "heartbeat interval conversion to nanoseconds" {
    // Verify the nanosecond conversion used in streaming loops
    const interval_ms: u64 = 15000;
    const interval_ns: u64 = interval_ms * 1_000_000;
    try std.testing.expectEqual(@as(u64, 15_000_000_000), interval_ns);
}

test "split target" {
    const parts = splitTarget("/api/stream?model=gpt");
    try std.testing.expectEqualStrings("/api/stream", parts.path);
    try std.testing.expectEqualStrings("model=gpt", parts.query);
}

test "extract json string" {
    const json = "{\"prompt\":\"hello world\",\"model\":\"gpt-4\"}";
    const prompt = extractJsonString(json, "prompt");
    try std.testing.expect(prompt != null);
    try std.testing.expectEqualStrings("hello world", prompt.?);
}

test "extract json int" {
    const json = "{\"max_tokens\":1024,\"other\":\"value\"}";
    const max_tokens = extractJsonInt(json, "max_tokens");
    try std.testing.expect(max_tokens != null);
    try std.testing.expectEqual(@as(i64, 1024), max_tokens.?);
}

test "websocket message parsing - cancel message" {
    // Test that cancel messages are correctly identified
    const cancel_json = "{\"type\":\"cancel\"}";
    const msg_type = extractJsonString(cancel_json, "type");
    try std.testing.expect(msg_type != null);
    try std.testing.expectEqualStrings("cancel", msg_type.?);
}

test "websocket message parsing - stream request" {
    const allocator = std.testing.allocator;

    const request_json = "{\"prompt\":\"Hello world\",\"backend\":\"local\",\"max_tokens\":100}";
    const request = try parseAbiStreamRequest(allocator, request_json);
    defer request.deinit(allocator);

    try std.testing.expectEqualStrings("Hello world", request.prompt);
    try std.testing.expectEqual(backends.BackendType.local, request.backend.?);
    try std.testing.expectEqual(@as(u32, 100), request.config.max_tokens);
}

test "websocket handler initialization" {
    const allocator = std.testing.allocator;

    var handler = try websocket.WebSocketHandler.init(allocator, .{});
    defer handler.deinit();

    try std.testing.expectEqual(websocket.ConnectionState.connecting, handler.state);
}

test "websocket frame encoding for streaming" {
    const allocator = std.testing.allocator;

    var handler = try websocket.WebSocketHandler.init(allocator, .{});
    defer handler.deinit();

    // Encode a token message
    const msg = try websocket.createStreamingMessage(allocator, "token", "hello");
    defer allocator.free(msg);

    const frame = try handler.sendText(msg);
    defer allocator.free(frame);

    // Verify frame structure: FIN + text opcode = 0x81
    try std.testing.expectEqual(@as(u8, 0x81), frame[0]);
    // Payload length should be > 0
    try std.testing.expect(frame[1] > 0);
}

test "admin reload request parsing" {
    // Test JSON parsing for reload endpoint
    const json = "{\"model_path\":\"/path/to/model.gguf\"}";
    const model_path = extractJsonString(json, "model_path");
    try std.testing.expect(model_path != null);
    try std.testing.expectEqualStrings("/path/to/model.gguf", model_path.?);
}

test "admin reload missing model_path" {
    // Test that missing model_path is detected
    const json = "{\"other_field\":\"value\"}";
    const model_path = extractJsonString(json, "model_path");
    try std.testing.expect(model_path == null);
}

test "admin reload error types" {
    // Test that new error types are available
    const err1: StreamingServerError = StreamingServerError.ModelReloadFailed;
    const err2: StreamingServerError = StreamingServerError.ModelReloadTimeout;
    try std.testing.expect(err1 != err2);
}

test {
    std.testing.refAllDecls(@This());
}
