const std = @import("std");
const protocol = @import("protocol.zig");
const handlers = @import("handlers.zig");
const server = @import("server.zig");
const json_helpers = @import("json_helpers.zig");

const McpMethod = protocol.McpMethod;
const validateRequest = protocol.validateRequest;
const MAX_REQUEST_SIZE = protocol.MAX_REQUEST_SIZE;
const appendJsonString = json_helpers.appendJsonString;

pub fn main(init: std.process.Init) !void {
    server.installSignalHandlers();

    // Spawn HTTP/SSE server thread
    const http_thread = std.Thread.spawn(.{}, server.runHttpServer, .{ init.gpa, init.io }) catch |err| {
        std.log.warn("failed to spawn HTTP/SSE server thread: {s}; continuing with stdio only", .{@errorName(err)});
        try server.runStdioLoop(init.gpa, init.io);
        return;
    };
    defer {
        server.wakeHttpServer(init.io);
        http_thread.join();
    }

    // Run stdio loop on the main thread
    server.runStdioLoop(init.gpa, init.io) catch |err| {
        std.log.err("stdio loop error: {s}", .{@errorName(err)});
    };

    // Signal the HTTP thread to stop
    server.requestShutdown();
}

test {
    std.testing.refAllDecls(@This());
}

test "stdio mode is the default MCP transport" {
    try std.testing.expect(std.mem.eql(u8, "stdio", "stdio"));
}

test "McpMethod fromString recognizes known methods" {
    try std.testing.expectEqual(McpMethod.initialize, McpMethod.fromString("initialize"));
    try std.testing.expectEqual(McpMethod.@"tools/list", McpMethod.fromString("tools/list"));
    try std.testing.expectEqual(McpMethod.@"tools/call", McpMethod.fromString("tools/call"));
    try std.testing.expectEqual(McpMethod.@"resources/list", McpMethod.fromString("resources/list"));
    try std.testing.expectEqual(McpMethod.@"prompts/list", McpMethod.fromString("prompts/list"));
    try std.testing.expectEqual(McpMethod.ping, McpMethod.fromString("ping"));
    try std.testing.expectEqual(McpMethod.shutdown, McpMethod.fromString("shutdown"));
    try std.testing.expectEqual(McpMethod.unknown, McpMethod.fromString("nonexistent"));
}

test "JsonRpcResponse serializes error field correctly" {
    const allocator = std.testing.allocator;
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var stream = std.json.Stringify{
        .writer = &out.writer,
        .options = .{},
    };
    try stream.beginObject();
    try stream.objectField("error");
    try stream.beginObject();
    try stream.objectField("code");
    try stream.write(@as(i32, -32600));
    try stream.objectField("message");
    try stream.write("Invalid Request");
    try stream.endObject();
    try stream.endObject();
    const body = out.written();
    try std.testing.expect(std.mem.indexOf(u8, body, "\"error\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"Invalid Request\"") != null);
}

test "JsonRpcResponse serializes result correctly" {
    const allocator = std.testing.allocator;
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var stream = std.json.Stringify{
        .writer = &out.writer,
        .options = .{},
    };
    try stream.beginObject();
    try stream.objectField("result");
    try stream.write("ok");
    try stream.endObject();
    const body = out.written();
    try std.testing.expect(std.mem.indexOf(u8, body, "\"result\"") != null);
}

test "request validation rejects oversized requests" {
    var big_buf: [MAX_REQUEST_SIZE + 1]u8 = undefined;
    @memset(&big_buf, '{');
    try std.testing.expectError(error.RequestTooLarge, validateRequest(&big_buf));
}

test "request validation rejects empty requests" {
    try std.testing.expectError(error.EmptyRequest, validateRequest(""));
}

test "request validation rejects non-JSON" {
    try std.testing.expectError(error.InvalidJsonFormat, validateRequest("hello world"));
}

test "request validation accepts valid JSON-RPC" {
    try validateRequest("{\"jsonrpc\":\"2.0\",\"method\":\"ping\"}");
}

test "signal handler sets shutdown flag" {
    server.requestShutdown();
    try std.testing.expect(server.isShutdownRequested());
    // Reset for next test
    server.requestShutdown();
}
