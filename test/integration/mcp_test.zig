//! Integration Tests: MCP (Model Context Protocol)
//!
//! Verifies MCP server types, tool registration, JSON-RPC types,
//! and server initialization without starting the stdio loop.

const std = @import("std");
const abi = @import("abi");

const mcp = abi.mcp;

// ============================================================================
// Module Exports
// ============================================================================

test "mcp: module exports core types" {
    _ = mcp.types;
    _ = mcp.Server;
    _ = mcp.RegisteredTool;
    _ = mcp.RegisteredResource;
    _ = mcp.ResourceHandler;
    _ = mcp.zls_bridge;
}

test "mcp: isEnabled reflects build option" {
    const enabled = mcp.isEnabled();
    _ = enabled;
}

test "mcp: Context type exists" {
    const ctx = mcp.Context;
    const enabled = ctx.isEnabled();
    _ = enabled;
}

// ============================================================================
// JSON-RPC Types
// ============================================================================

test "mcp: RequestId integer variant" {
    const id = mcp.types.RequestId{ .integer = 42 };
    switch (id) {
        .integer => |v| try std.testing.expectEqual(@as(i64, 42), v),
        .string => unreachable,
    }
}

test "mcp: RequestId string variant" {
    const id = mcp.types.RequestId{ .string = "req-abc" };
    switch (id) {
        .string => |v| try std.testing.expectEqualStrings("req-abc", v),
        .integer => unreachable,
    }
}

test "mcp: protocol version is defined" {
    try std.testing.expect(mcp.types.PROTOCOL_VERSION.len > 0);
}

// ============================================================================
// Tool Definition Types
// ============================================================================

test "mcp: ToolDef struct layout" {
    const tool_def = mcp.types.ToolDef{
        .name = "test_tool",
        .description = "A test tool for integration testing",
        .input_schema =
        \\{"type":"object","properties":{},"required":[]}
        ,
    };

    try std.testing.expectEqualStrings("test_tool", tool_def.name);
    try std.testing.expectEqualStrings("A test tool for integration testing", tool_def.description);
    try std.testing.expect(tool_def.input_schema.len > 0);
}

test "mcp: ToolResult struct" {
    const content_blocks = [_]mcp.types.ContentBlock{
        .{ .text = "Hello from tool" },
    };
    const result = mcp.types.ToolResult{
        .content = &content_blocks,
        .is_error = false,
    };

    try std.testing.expect(!result.is_error);
    try std.testing.expectEqual(@as(usize, 1), result.content.len);
    try std.testing.expectEqualStrings("Hello from tool", result.content[0].text);
}

test "mcp: ContentBlock defaults to text type" {
    const block = mcp.types.ContentBlock{ .text = "test content" };
    try std.testing.expectEqualStrings("text", block.type);
    try std.testing.expectEqualStrings("test content", block.text);
}

// ============================================================================
// Server Capabilities
// ============================================================================

test "mcp: ServerCapabilities defaults" {
    const caps = mcp.types.ServerCapabilities{};
    try std.testing.expect(caps.tools == null);
    try std.testing.expect(caps.resources == null);
}

test "mcp: ServerCapabilities with tools" {
    const caps = mcp.types.ServerCapabilities{
        .tools = .{ .listChanged = true },
    };
    try std.testing.expect(caps.tools != null);
    try std.testing.expect(caps.tools.?.listChanged);
}

test "mcp: ServerInfo struct" {
    const info = mcp.types.ServerInfo{
        .name = "abi-test",
        .version = "1.0.0",
    };
    try std.testing.expectEqualStrings("abi-test", info.name);
    try std.testing.expectEqualStrings("1.0.0", info.version);
}

// ============================================================================
// Server Init / Deinit
// ============================================================================

test "mcp: server init and deinit" {
    var server = mcp.Server.init(std.testing.allocator, "test-server", "0.1.0");
    defer server.deinit();

    try std.testing.expectEqualStrings("test-server", server.server_name);
    try std.testing.expectEqualStrings("0.1.0", server.server_version);
    try std.testing.expect(!server.initialized);
}

test "mcp: server starts with no tools" {
    var server = mcp.Server.init(std.testing.allocator, "empty-server", "0.0.1");
    defer server.deinit();

    try std.testing.expectEqual(@as(usize, 0), server.tools.len);
    try std.testing.expectEqual(@as(usize, 0), server.resources.len);
}

// ============================================================================
// Tool Registration
// ============================================================================

fn dummyHandler(_: std.mem.Allocator, _: ?std.json.ObjectMap, out: *std.ArrayListUnmanaged(u8)) anyerror!void {
    try out.appendSlice(std.testing.allocator, "ok");
}

test "mcp: register tool with server" {
    var server = mcp.Server.init(std.testing.allocator, "tool-test", "0.1.0");
    defer server.deinit();

    try server.addTool(.{
        .def = .{
            .name = "my_tool",
            .description = "A test tool",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = dummyHandler,
    });

    try std.testing.expectEqual(@as(usize, 1), server.tools.len);
    try std.testing.expectEqualStrings("my_tool", server.tools.items[0].def.name);
}

test "mcp: register multiple tools" {
    var server = mcp.Server.init(std.testing.allocator, "multi-tool", "0.1.0");
    defer server.deinit();

    try server.addTool(.{
        .def = .{ .name = "tool_a", .description = "Tool A", .input_schema = "{}" },
        .handler = dummyHandler,
    });
    try server.addTool(.{
        .def = .{ .name = "tool_b", .description = "Tool B", .input_schema = "{}" },
        .handler = dummyHandler,
    });

    try std.testing.expectEqual(@as(usize, 2), server.tools.len);
    try std.testing.expectEqualStrings("tool_a", server.tools.items[0].def.name);
    try std.testing.expectEqualStrings("tool_b", server.tools.items[1].def.name);
}

// ============================================================================
// Factory Functions
// ============================================================================

test "mcp: createStatusServer" {
    var server = try mcp.createStatusServer(std.testing.allocator, "0.1.0-test");
    defer server.deinit();

    try std.testing.expectEqualStrings("abi-status", server.server_name);
    try std.testing.expect(server.tools.len > 0);
}

test {
    std.testing.refAllDecls(@This());
}
