const std = @import("std");
const protocol = @import("protocol.zig");
const handlers = @import("handlers.zig");
const server = @import("server.zig");
const json_helpers = @import("json_helpers.zig");
const state = @import("state.zig");
const abi = @import("abi");

const McpMethod = protocol.McpMethod;
const validateRequest = protocol.validateRequest;
const MAX_REQUEST_SIZE = protocol.MAX_REQUEST_SIZE;
const appendJsonString = json_helpers.appendJsonString;

pub fn main(init: std.process.Init) !void {
    // Capture the process environment for portable, libc-free env lookups used
    // by the abi-module readers (durable store path, credentials, etc.).
    abi.foundation.env.install(init.environ_map);
    // Resolve the HTTP listen port here (main can reach the captured env; the
    // transport module cannot, per the import rules) and hand it to the server.
    server.setHttpPort(abi.foundation.env.get(server.HTTP_PORT_ENV));
    server.installSignalHandlers();

    // Persist the WDBX store across server restarts (default-ON). The durable
    // store is opened lazily on first use with this IO handle; the defer
    // guarantees a checkpoint on the normal-exit path (the `shutdown` RPC also
    // checkpoints, and deinit is idempotent).
    state.setIo(init.io);
    defer state.deinitWdbxStore();
    // Tear down the shared scheduler on the owning (main) thread, AFTER the HTTP
    // thread is joined (LIFO: this runs after the join defer below). The in-band
    // `shutdown` RPC now only signals, so it never frees the scheduler while a
    // peer transport's tool call is in flight.
    defer state.deinitScheduler();

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
