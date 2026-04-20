//! MCP Server — JSON-RPC 2.0 over stdio.
//!
//! Reads newline-delimited JSON-RPC messages from stdin, dispatches to
//! registered tool handlers, and writes responses to stdout.
//!
//! Designed for use with Claude Desktop, Cursor, and other MCP-compatible clients.
//!
//! Implementation is decomposed into submodules under `server/`:
//!   - `registration.zig` — handler and registered-type definitions
//!   - `lifecycle.zig` — Server struct, init/deinit, run, message processing
//!   - `dispatch.zig` — JSON-RPC method routing
//!   - `io_loop.zig` — stdin/stdout I/O loop
//!   - `tools.zig` — tools/list and tools/call handlers
//!   - `resources.zig` — resources/list and resources/read handlers
//!   - `json_write.zig` — JSON string escaping utilities

const std = @import("std");

// ─── Submodule imports ───────────────────────────────────────────────
const registration = @import("server/registration.zig");
const lifecycle = @import("server/lifecycle.zig");

// Re-exported for external consumers (real.zig, zls_bridge.zig, tests)
pub const json_write = @import("server/json_write.zig");

// ─── Public type re-exports (preserve exact public surface) ──────────
pub const ToolHandler = registration.ToolHandler;
pub const RegisteredTool = registration.RegisteredTool;
pub const ResourceHandler = registration.ResourceHandler;
pub const RegisteredResource = registration.RegisteredResource;
pub const MAX_MESSAGE_SIZE = lifecycle.MAX_MESSAGE_SIZE;
pub const Server = lifecycle.Server;

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

test "Server init and deinit" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test-server", "0.1.0");
    defer server.deinit();
    try std.testing.expectEqualStrings("test-server", server.server_name);
}

test "Server tool registration" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addTool(.{
        .def = .{
            .name = "test_tool",
            .description = "A test tool",
            .input_schema = "{}",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: ?std.json.ObjectMap, out: *std.ArrayListUnmanaged(u8)) !void {
                try out.appendSlice(std.testing.allocator, "hello");
            }
        }.handle,
    });

    try std.testing.expectEqual(@as(usize, 1), server.tools.items.len);
}

test "appendJsonEscaped via json_write" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    try json_write.appendJsonEscaped(allocator, &buf, "hello \"world\"");
    try std.testing.expectEqualStrings("hello \\\"world\\\"", buf.items);
}

test "handleMessage initialize" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test-server", "0.1.0");
    defer server.deinit();

    var out: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    const msg =
        \\{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}
    ;
    try server.processMessage(msg, &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"protocolVersion\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "\"id\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "test-server") != null);
}

test "handleMessage ping" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"ping","id":42}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expectEqualStrings(
        "{\"jsonrpc\":\"2.0\",\"id\":42,\"result\":{}}\n",
        written,
    );
}

test "handleMessage tools/list" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addTool(.{
        .def = .{ .name = "echo", .description = "Echo tool", .input_schema = "{}" },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: ?std.json.ObjectMap, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "ok");
            }
        }.handle,
    });

    var out: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"tools/list","id":2}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"echo\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "\"tools\":[") != null);
}

test "handleMessage tools/call" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addTool(.{
        .def = .{ .name = "greet", .description = "Greeting", .input_schema = "{}" },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: ?std.json.ObjectMap, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "Hello, MCP!");
            }
        }.handle,
    });

    var out: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"tools/call","id":3,"params":{"name":"greet","arguments":{}}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Hello, MCP!") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "\"content\"") != null);
}

test "handleMessage unknown method" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"nonexistent/method","id":5}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "-32601") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "Method not found") != null);
}

test "handleMessage invalid JSON" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage("not json at all", &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "-32700") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "Parse error") != null);
}

test "handleMessage notifications/initialized" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try std.testing.expect(!server.initialized);

    var out: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"notifications/initialized"}
    , &writer);

    // Notification — no response written
    try std.testing.expectEqual(@as(usize, 0), writer.end);
    try std.testing.expect(server.initialized);
}

test "handleMessage unknown tool" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"tools/call","id":4,"params":{"name":"nonexistent"}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Unknown tool") != null);
}

test "handleMessage rejects invalid jsonrpc version" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"1.0","method":"ping","id":1}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Invalid JSON-RPC version") != null);
}

test "handleMessage array instead of object" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage("[1,2,3]", &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Expected JSON object") != null);
}

test "handleMessage missing tool name in params" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"tools/call","id":1,"params":{}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Missing tool name") != null);
}

test "handleMessage non-string method" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":42,"id":1}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Method must be string") != null);
}

test "handleMessage tools/call with no params" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"tools/call","id":10}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Missing params") != null);
}

test "handleMessage with string request ID" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"ping","id":"abc-123"}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"abc-123\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "\"result\":{}") != null);
}

test "handleMessage resources/list empty" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"resources/list","id":5}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"resources\":[]") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "\"id\":5") != null);
}

test "handleMessage resources/list with registered resources" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addResource(.{
        .def = .{
            .uri = "abi://status",
            .name = "Server Status",
            .description = "Current server status",
            .mime_type = "application/json",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "{\"status\":\"ok\"}");
            }
        }.handle,
    });

    var out: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"resources/list","id":6}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"resources\":[") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "abi://status") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "Server Status") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "application/json") != null);
}

test "handleMessage resources/read" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addResource(.{
        .def = .{
            .uri = "abi://version",
            .name = "Version",
            .description = "Server version",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "1.0.0");
            }
        }.handle,
    });

    var out: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"resources/read","id":7,"params":{"uri":"abi://version"}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"contents\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "abi://version") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "1.0.0") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "text/plain") != null);
}

test "handleMessage resources/read unknown resource" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"resources/read","id":8,"params":{"uri":"abi://nonexistent"}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Resource not found") != null);
}

test "handleMessage resources/read missing params" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"resources/read","id":9}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Missing params") != null);
}

test "handleMessage resources/read missing URI" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"resources/read","id":10,"params":{}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Missing resource URI") != null);
}

test "handleMessage resources/read error returns error in content" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addResource(.{
        .def = .{
            .uri = "abi://broken",
            .name = "Broken",
            .description = "Always fails",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, _: *std.ArrayListUnmanaged(u8)) !void {
                return error.ResourceUnavailable;
            }
        }.handle,
    });

    var out: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"resources/read","id":11,"params":{"uri":"abi://broken"}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"contents\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "Error:") != null);
}

test "handleMessage initialize advertises resources when registered" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addResource(.{
        .def = .{
            .uri = "abi://info",
            .name = "Info",
            .description = "Server info",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "info");
            }
        }.handle,
    });

    var out: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"resources\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "\"tools\"") != null);
}

test "handleMessage tool error returns isError" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addTool(.{
        .def = .{ .name = "fail", .description = "Always fails", .input_schema = "{}" },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: ?std.json.ObjectMap, _: *std.ArrayListUnmanaged(u8)) !void {
                return error.SomethingBad;
            }
        }.handle,
    });

    var out: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"tools/call","id":7,"params":{"name":"fail"}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"isError\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "Error:") != null);
}

test "processMessage rejects oversized message" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    // Create a message larger than MAX_MESSAGE_SIZE
    const oversized = try allocator.alloc(u8, MAX_MESSAGE_SIZE + 1);
    defer allocator.free(oversized);
    @memset(oversized, 'x');

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(oversized, &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "-32600") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "message too large") != null);
}

test "processMessage rejects missing jsonrpc field" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"method":"ping","id":1}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "-32600") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "Missing required jsonrpc field") != null);
}

test "processMessage handler error produces JSON-RPC error not crash" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addTool(.{
        .def = .{ .name = "crashy", .description = "Fails with error", .input_schema = "{}" },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: ?std.json.ObjectMap, _: *std.ArrayListUnmanaged(u8)) !void {
                return error.OutOfMemory;
            }
        }.handle,
    });

    var out: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.processMessage(
        \\{"jsonrpc":"2.0","method":"tools/call","id":99,"params":{"name":"crashy"}}
    , &writer);

    const written = out[0..writer.end];
    // Should get a proper response (not a crash), with isError flag
    try std.testing.expect(std.mem.indexOf(u8, written, "\"isError\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "\"id\":99") != null);
}

test "processMessage normal request works after error recovery" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addTool(.{
        .def = .{ .name = "bad", .description = "Fails", .input_schema = "{}" },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: ?std.json.ObjectMap, _: *std.ArrayListUnmanaged(u8)) !void {
                return error.Broken;
            }
        }.handle,
    });

    // First: send a request that triggers an error
    {
        var out: [1024]u8 = undefined;
        var writer = std.Io.Writer.fixed(&out);
        try server.processMessage(
            \\{"jsonrpc":"2.0","method":"tools/call","id":1,"params":{"name":"bad"}}
        , &writer);
        const written = out[0..writer.end];
        try std.testing.expect(std.mem.indexOf(u8, written, "\"isError\":true") != null);
    }

    // Second: send a normal ping — server should still work
    {
        var out: [256]u8 = undefined;
        var writer = std.Io.Writer.fixed(&out);
        try server.processMessage(
            \\{"jsonrpc":"2.0","method":"ping","id":2}
        , &writer);
        const written = out[0..writer.end];
        try std.testing.expectEqualStrings(
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{}}\n",
            written,
        );
    }
}

test "processMessage validates size then content" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    // A message within size limit but with invalid JSON still gets parse error
    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);
    try server.processMessage("not valid json", &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "-32700") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "Parse error") != null);
}

test {
    std.testing.refAllDecls(@This());
}
