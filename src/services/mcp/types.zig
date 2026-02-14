//! MCP (Model Context Protocol) type definitions.
//!
//! JSON-RPC 2.0 message types for the MCP protocol used by AI clients
//! (Claude Desktop, Cursor, etc.) to interact with tool servers.

const std = @import("std");

/// JSON-RPC 2.0 request ID — can be string or integer
pub const RequestId = union(enum) {
    integer: i64,
    string: []const u8,

    pub fn format(self: RequestId, writer: anytype) !void {
        switch (self) {
            .integer => |v| try writer.print("{d}", .{v}),
            .string => |v| {
                try writeJsonString(writer, v);
            },
        }
    }

    pub fn fromJson(value: std.json.Value) ?RequestId {
        return switch (value) {
            .integer => |v| .{ .integer = v },
            .string => |v| .{ .string = v },
            .number_string => |v| .{ .string = v },
            else => null,
        };
    }
};

/// MCP protocol version
pub const PROTOCOL_VERSION = "2024-11-05";

/// Server capabilities advertised during initialization
pub const ServerCapabilities = struct {
    tools: ?ToolsCapability = null,
    resources: ?ResourcesCapability = null,

    pub const ToolsCapability = struct {
        listChanged: bool = false,
    };

    pub const ResourcesCapability = struct {
        subscribe: bool = false,
        listChanged: bool = false,
    };
};

/// Server info returned during initialization
pub const ServerInfo = struct {
    name: []const u8,
    version: []const u8,
};

/// Tool definition exposed to MCP clients
pub const ToolDef = struct {
    name: []const u8,
    description: []const u8,
    /// JSON Schema for the tool's input parameters (as pre-built JSON string)
    input_schema: []const u8,
};

/// Result of a tool call
pub const ToolResult = struct {
    content: []const ContentBlock,
    is_error: bool = false,
};

/// Content block in a tool result
pub const ContentBlock = struct {
    type: []const u8 = "text",
    text: []const u8,
};

/// Write a JSON-RPC 2.0 response to a writer
pub fn writeResponse(
    writer: anytype,
    id: RequestId,
    result_json: []const u8,
) !void {
    try writer.writeAll("{\"jsonrpc\":\"2.0\",\"id\":");
    try id.format(writer);
    try writer.writeAll(",\"result\":");
    try writer.writeAll(result_json);
    try writer.writeAll("}\n");
}

/// Write a JSON-RPC 2.0 error response
pub fn writeError(
    writer: anytype,
    id: ?RequestId,
    code: i32,
    message: []const u8,
) !void {
    try writer.writeAll("{\"jsonrpc\":\"2.0\",\"id\":");
    if (id) |rid| {
        try rid.format(writer);
    } else {
        try writer.writeAll("null");
    }
    try writer.print(",\"error\":{{\"code\":{d},\"message\":", .{code});
    try writeJsonString(writer, message);
    try writer.writeAll("}}\n");
}

/// Write a JSON-escaped string
pub fn writeJsonString(writer: anytype, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (c < 0x20) {
                    try writer.print("\\u{x:0>4}", .{c});
                } else {
                    try writer.writeByte(c);
                }
            },
        }
    }
    try writer.writeByte('"');
}

/// Standard JSON-RPC error codes
pub const ErrorCode = struct {
    pub const parse_error: i32 = -32700;
    pub const invalid_request: i32 = -32600;
    pub const method_not_found: i32 = -32601;
    pub const invalid_params: i32 = -32602;
    pub const internal_error: i32 = -32603;
};

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

test "RequestId format integer" {
    var buf: [64]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    const id = RequestId{ .integer = 42 };
    try id.format(&writer);
    try std.testing.expectEqualStrings("42", buf[0..writer.end]);
}

test "RequestId format string" {
    var buf: [64]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    const id = RequestId{ .string = "abc" };
    try id.format(&writer);
    try std.testing.expectEqualStrings("\"abc\"", buf[0..writer.end]);
}

test "RequestId format string escapes special chars" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    const id = RequestId{ .string = "ab\"cd\\ef\ngh" };
    try id.format(&writer);
    try std.testing.expectEqualStrings("\"ab\\\"cd\\\\ef\\ngh\"", buf[0..writer.end]);
}

test "writeJsonString escapes" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeJsonString(&writer, "hello \"world\"\nnew");
    try std.testing.expectEqualStrings("\"hello \\\"world\\\"\\nnew\"", buf[0..writer.end]);
}

test "writeError format" {
    var buf: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try writeError(&writer, .{ .integer = 1 }, -32601, "Method not found");
    const expected = "{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-32601,\"message\":\"Method not found\"}}\n";
    try std.testing.expectEqualStrings(expected, buf[0..writer.end]);
}
