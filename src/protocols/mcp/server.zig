//! MCP Server — JSON-RPC 2.0 over stdio.
//!
//! Reads newline-delimited JSON-RPC messages from stdin, dispatches to
//! registered tool handlers, and writes responses to stdout.
//!
//! Designed for use with Claude Desktop, Cursor, and other MCP-compatible clients.

const std = @import("std");
const types = @import("types.zig");

/// Tool handler function signature
pub const ToolHandler = *const fn (
    allocator: std.mem.Allocator,
    params_json: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) anyerror!void;

/// Registered tool with metadata and handler
pub const RegisteredTool = struct {
    def: types.ToolDef,
    handler: ToolHandler,
};

/// Resource handler function signature
pub const ResourceHandler = *const fn (
    allocator: std.mem.Allocator,
    uri: []const u8,
    out: *std.ArrayListUnmanaged(u8),
) anyerror!void;

/// Registered resource with metadata and handler
pub const RegisteredResource = struct {
    def: types.ResourceDef,
    handler: ResourceHandler,
};

/// Maximum message size accepted by the server (4 MB).
/// Messages exceeding this limit are rejected with a JSON-RPC Invalid Request error
/// to prevent denial-of-service via oversized stdin payloads.
pub const MAX_MESSAGE_SIZE: usize = 4 * 1024 * 1024;

/// MCP Server state
pub const Server = struct {
    allocator: std.mem.Allocator,
    tools: std.ArrayListUnmanaged(RegisteredTool),
    resources: std.ArrayListUnmanaged(RegisteredResource),
    subscriptions: std.StringHashMapUnmanaged(bool),
    server_name: []const u8,
    server_version: []const u8,
    initialized: bool,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        version: []const u8,
    ) Self {
        return .{
            .allocator = allocator,
            .tools = .empty,
            .resources = .empty,
            .subscriptions = .empty,
            .server_name = name,
            .server_version = version,
            .initialized = false,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free duped subscription keys
        var it = self.subscriptions.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.subscriptions.deinit(self.allocator);
        self.tools.deinit(self.allocator);
        self.resources.deinit(self.allocator);
    }

    /// Register a tool with the server
    pub fn addTool(self: *Self, tool: RegisteredTool) !void {
        try self.tools.append(self.allocator, tool);
    }

    /// Register a resource with the server
    pub fn addResource(self: *Self, resource: RegisteredResource) !void {
        try self.resources.append(self.allocator, resource);
    }

    /// Subscribe to change notifications for a resource URI.
    /// The URI must correspond to a registered resource.
    /// Returns true if the subscription was newly created, false if already subscribed.
    pub fn subscribeResource(self: *Self, uri: []const u8) !bool {
        // Verify the resource exists
        var found = false;
        for (self.resources.items) |resource| {
            if (std.mem.eql(u8, resource.def.uri, uri)) {
                found = true;
                break;
            }
        }
        if (!found) return error.ResourceNotFound;

        const result = try self.subscriptions.getOrPut(self.allocator, uri);
        if (result.found_existing) {
            // Already subscribed
            return false;
        }
        // Dupe the key so it is owned by the subscriptions map
        result.key_ptr.* = try self.allocator.dupe(u8, uri);
        result.value_ptr.* = true;
        return true;
    }

    /// Unsubscribe from change notifications for a resource URI.
    /// Returns true if the subscription was removed, false if not subscribed.
    pub fn unsubscribeResource(self: *Self, uri: []const u8) bool {
        const entry = self.subscriptions.fetchRemove(uri);
        if (entry) |kv| {
            self.allocator.free(kv.key);
            return true;
        }
        return false;
    }

    /// Check whether a resource URI is currently subscribed.
    pub fn isSubscribed(self: *Self, uri: []const u8) bool {
        return self.subscriptions.get(uri) orelse false;
    }

    /// Send a resource-change notification to the client.
    /// Only sends if the URI is currently subscribed.
    /// The caller provides the writer (stdout in production).
    pub fn notifyResourceChanged(self: *Self, uri: []const u8, writer: anytype) !void {
        if (!self.isSubscribed(uri)) return;

        // Build params: {"uri":"<escaped-uri>"}
        var buf = std.ArrayListUnmanaged(u8).empty;
        defer buf.deinit(self.allocator);

        try buf.appendSlice(self.allocator, "{\"uri\":\"");
        try appendJsonEscaped(self.allocator, &buf, uri);
        try buf.appendSlice(self.allocator, "\"}");

        try types.writeNotification(writer, "notifications/resources/updated", buf.items);
    }

    /// Run the server loop — reads from stdin, writes to stdout.
    /// The caller must provide a Zig 0.16 I/O handle (from `std.Io.Threaded`).
    pub fn run(self: *Self, io: std.Io) !void {
        std.log.info("MCP server ready ({d} tools registered)", .{self.tools.items.len});

        // Set up stdin reader
        var stdin_file = std.Io.File.stdin();
        var read_buf: [65536]u8 = undefined;
        var reader = stdin_file.reader(io, &read_buf);

        // Set up stdout writer
        var stdout_file = std.Io.File.stdout();
        var write_buf: [65536]u8 = undefined;
        var writer = stdout_file.writer(io, &write_buf);

        // Read newline-delimited JSON-RPC messages
        while (true) {
            const line_opt = reader.interface.takeDelimiter('\n') catch |err| switch (err) {
                error.StreamTooLong => {
                    // Message exceeded buffer — send parse error and continue
                    try types.writeError(
                        &writer.interface,
                        null,
                        types.ErrorCode.parse_error,
                        "Message too large",
                    );
                    try writer.flush();
                    continue;
                },
                else => break, // Read failure — exit loop
            };

            const line = line_opt orelse break; // EOF — client disconnected
            const trimmed = std.mem.trim(u8, line, " \t\r\n");
            if (trimmed.len == 0) continue; // Skip empty lines

            // Dispatch message and write response
            self.processMessage(trimmed, &writer.interface) catch |err| {
                std.log.err("Error handling message: {t}", .{err});
                types.writeError(
                    &writer.interface,
                    null,
                    types.ErrorCode.internal_error,
                    "Internal error",
                ) catch |write_err| {
                    std.log.err("MCP: failed to write error response: {t}", .{write_err});
                    break;
                };
            };

            // Flush after each message to ensure client receives response
            writer.flush() catch |flush_err| {
                std.log.err("MCP: flush error, closing connection: {t}", .{flush_err});
                break;
            };
        }

        // Final flush before exit
        writer.flush() catch |err| {
            std.log.warn("MCP: final flush failed: {t}", .{err});
        };
    }

    /// Run without I/O — logs readiness (for environments without I/O backend).
    pub fn runInfo(self: *Self) void {
        std.log.info("MCP server ready ({d} tools registered). Use run(io) with I/O backend.", .{self.tools.items.len});
    }

    /// Process a single JSON-RPC message with size validation.
    /// This is the public entry point for message handling — it enforces
    /// MAX_MESSAGE_SIZE and delegates to the internal dispatch logic.
    /// Returns without error even when the message is invalid; error
    /// responses are written to `writer` per JSON-RPC 2.0 spec.
    pub fn processMessage(self: *Self, line: []const u8, writer: anytype) !void {
        // Enforce message size limit to prevent DoS
        if (line.len > MAX_MESSAGE_SIZE) {
            std.log.warn("MCP: rejecting oversized message ({d} bytes, limit {d})", .{ line.len, MAX_MESSAGE_SIZE });
            try types.writeError(
                writer,
                null,
                types.ErrorCode.invalid_request,
                "Invalid Request - message too large",
            );
            return;
        }
        return self.handleMessage(line, writer);
    }

    fn handleMessage(self: *Self, line: []const u8, writer: anytype) !void {
        // Parse JSON
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            line,
            .{},
        ) catch {
            try types.writeError(writer, null, types.ErrorCode.parse_error, "Parse error");
            return;
        };
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) {
            try types.writeError(writer, null, types.ErrorCode.invalid_request, "Expected JSON object");
            return;
        }

        const obj = root.object;

        // Extract request ID (may be absent for notifications)
        const id: ?types.RequestId = if (obj.get("id")) |id_val|
            types.RequestId.fromJson(id_val)
        else
            null;

        // Validate JSON-RPC version (spec requires "2.0" field to be present)
        const ver = obj.get("jsonrpc") orelse {
            try types.writeError(writer, id, types.ErrorCode.invalid_request, "Missing required jsonrpc field");
            return;
        };
        if (ver != .string or !std.mem.eql(u8, ver.string, "2.0")) {
            try types.writeError(writer, id, types.ErrorCode.invalid_request, "Invalid JSON-RPC version");
            return;
        }

        // Extract method
        const method_val = obj.get("method") orelse {
            try types.writeError(writer, id, types.ErrorCode.invalid_request, "Missing method");
            return;
        };
        if (method_val != .string) {
            try types.writeError(writer, id, types.ErrorCode.invalid_request, "Method must be string");
            return;
        }
        const method = method_val.string;

        // Extract params (optional)
        const params: ?std.json.ObjectMap = if (obj.get("params")) |p|
            (if (p == .object) p.object else null)
        else
            null;

        // Dispatch
        if (std.mem.eql(u8, method, "initialize")) {
            try self.handleInitialize(writer, id);
        } else if (std.mem.eql(u8, method, "notifications/initialized")) {
            // Notification — no response needed
            self.initialized = true;
        } else if (std.mem.eql(u8, method, "tools/list")) {
            try self.handleToolsList(writer, id);
        } else if (std.mem.eql(u8, method, "tools/call")) {
            try self.handleToolsCall(writer, id, params);
        } else if (std.mem.eql(u8, method, "resources/list")) {
            try self.handleResourcesList(writer, id);
        } else if (std.mem.eql(u8, method, "resources/read")) {
            try self.handleResourcesRead(writer, id, params);
        } else if (std.mem.eql(u8, method, "resources/subscribe")) {
            try self.handleResourcesSubscribe(writer, id, params);
        } else if (std.mem.eql(u8, method, "resources/unsubscribe")) {
            try self.handleResourcesUnsubscribe(writer, id, params);
        } else if (std.mem.eql(u8, method, "ping")) {
            try self.handlePing(writer, id);
        } else {
            try types.writeError(writer, id, types.ErrorCode.method_not_found, "Method not found");
        }
    }

    fn handleInitialize(self: *Self, writer: anytype, id: ?types.RequestId) !void {
        const rid = id orelse return;
        var buf = std.ArrayListUnmanaged(u8).empty;
        defer buf.deinit(self.allocator);

        try buf.appendSlice(self.allocator, "{\"protocolVersion\":\"");
        try buf.appendSlice(self.allocator, types.PROTOCOL_VERSION);
        try buf.appendSlice(self.allocator, "\",\"capabilities\":{\"tools\":{\"listChanged\":false}");
        if (self.resources.items.len > 0) {
            try buf.appendSlice(self.allocator, ",\"resources\":{\"subscribe\":true,\"listChanged\":false}");
        }
        try buf.appendSlice(self.allocator, "}");
        try buf.appendSlice(self.allocator, ",\"serverInfo\":{\"name\":\"");
        try appendJsonEscaped(self.allocator, &buf, self.server_name);
        try buf.appendSlice(self.allocator, "\",\"version\":\"");
        try appendJsonEscaped(self.allocator, &buf, self.server_version);
        try buf.appendSlice(self.allocator, "\"}}");

        try types.writeResponse(writer, rid, buf.items);
    }

    fn handlePing(_: *Self, writer: anytype, id: ?types.RequestId) !void {
        const rid = id orelse return;
        try types.writeResponse(writer, rid, "{}");
    }

    fn handleToolsList(self: *Self, writer: anytype, id: ?types.RequestId) !void {
        const rid = id orelse return;
        var buf = std.ArrayListUnmanaged(u8).empty;
        defer buf.deinit(self.allocator);

        try buf.appendSlice(self.allocator, "{\"tools\":[");

        for (self.tools.items, 0..) |tool, i| {
            if (i > 0) try buf.append(self.allocator, ',');
            try buf.appendSlice(self.allocator, "{\"name\":\"");
            try buf.appendSlice(self.allocator, tool.def.name);
            try buf.appendSlice(self.allocator, "\",\"description\":\"");
            try appendJsonEscaped(self.allocator, &buf, tool.def.description);
            try buf.appendSlice(self.allocator, "\",\"inputSchema\":");
            try buf.appendSlice(self.allocator, tool.def.input_schema);
            try buf.append(self.allocator, '}');
        }

        try buf.appendSlice(self.allocator, "]}");
        try types.writeResponse(writer, rid, buf.items);
    }

    fn handleToolsCall(self: *Self, writer: anytype, id: ?types.RequestId, params: ?std.json.ObjectMap) !void {
        const rid = id orelse return;

        // Get tool name from params
        const p = params orelse {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing params");
            return;
        };

        const name_val = p.get("name") orelse {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing tool name");
            return;
        };
        if (name_val != .string) {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Tool name must be string");
            return;
        }
        const tool_name = name_val.string;

        // Get arguments
        const args: ?std.json.ObjectMap = if (p.get("arguments")) |a|
            (if (a == .object) a.object else null)
        else
            null;

        // Find and execute tool
        for (self.tools.items) |tool| {
            if (std.mem.eql(u8, tool.def.name, tool_name)) {
                var result_buf = std.ArrayListUnmanaged(u8).empty;
                defer result_buf.deinit(self.allocator);

                // Call tool handler
                tool.handler(self.allocator, args, &result_buf) catch |err| {
                    // Tool error — return as MCP tool error content
                    // Use individual catch blocks to avoid cascading OOM disconnecting the client
                    var err_buf = std.ArrayListUnmanaged(u8).empty;
                    defer err_buf.deinit(self.allocator);

                    err_buf.appendSlice(self.allocator, "{\"content\":[{\"type\":\"text\",\"text\":\"Error: ") catch |alloc_err| {
                        std.log.err("MCP: OOM formatting tool error: {t}", .{alloc_err});
                        return;
                    };
                    var err_msg_buf: [128]u8 = undefined;
                    const err_msg = std.fmt.bufPrint(&err_msg_buf, "{t}", .{err}) catch "unknown error";
                    appendJsonEscaped(self.allocator, &err_buf, err_msg) catch |alloc_err| {
                        std.log.err("MCP: OOM escaping tool error message: {t}", .{alloc_err});
                        return;
                    };
                    err_buf.appendSlice(self.allocator, "\"}],\"isError\":true}") catch |alloc_err| {
                        std.log.err("MCP: OOM formatting tool error suffix: {t}", .{alloc_err});
                        return;
                    };
                    types.writeResponse(writer, rid, err_buf.items) catch |write_err| {
                        std.log.err("MCP: failed to write tool error response: {t}", .{write_err});
                        return;
                    };
                    return;
                };

                // Wrap result in MCP content format
                var out = std.ArrayListUnmanaged(u8).empty;
                defer out.deinit(self.allocator);

                try out.appendSlice(self.allocator, "{\"content\":[{\"type\":\"text\",\"text\":\"");
                try appendJsonEscaped(self.allocator, &out, result_buf.items);
                try out.appendSlice(self.allocator, "\"}]}");
                try types.writeResponse(writer, rid, out.items);
                return;
            }
        }

        try types.writeError(writer, rid, types.ErrorCode.method_not_found, "Unknown tool");
    }

    fn handleResourcesList(self: *Self, writer: anytype, id: ?types.RequestId) !void {
        const rid = id orelse return;
        var buf = std.ArrayListUnmanaged(u8).empty;
        defer buf.deinit(self.allocator);

        try buf.appendSlice(self.allocator, "{\"resources\":[");

        for (self.resources.items, 0..) |resource, i| {
            if (i > 0) try buf.append(self.allocator, ',');
            try buf.appendSlice(self.allocator, "{\"uri\":\"");
            try appendJsonEscaped(self.allocator, &buf, resource.def.uri);
            try buf.appendSlice(self.allocator, "\",\"name\":\"");
            try appendJsonEscaped(self.allocator, &buf, resource.def.name);
            try buf.appendSlice(self.allocator, "\",\"description\":\"");
            try appendJsonEscaped(self.allocator, &buf, resource.def.description);
            try buf.appendSlice(self.allocator, "\",\"mimeType\":\"");
            try appendJsonEscaped(self.allocator, &buf, resource.def.mime_type);
            try buf.appendSlice(self.allocator, "\"}");
        }

        try buf.appendSlice(self.allocator, "]}");
        try types.writeResponse(writer, rid, buf.items);
    }

    fn handleResourcesRead(self: *Self, writer: anytype, id: ?types.RequestId, params: ?std.json.ObjectMap) !void {
        const rid = id orelse return;

        const p = params orelse {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing params");
            return;
        };

        const uri_val = p.get("uri") orelse {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing resource URI");
            return;
        };
        if (uri_val != .string) {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Resource URI must be string");
            return;
        }
        const uri = uri_val.string;

        // Find and read resource
        for (self.resources.items) |resource| {
            if (std.mem.eql(u8, resource.def.uri, uri)) {
                var result_buf = std.ArrayListUnmanaged(u8).empty;
                defer result_buf.deinit(self.allocator);

                // Call resource handler
                resource.handler(self.allocator, uri, &result_buf) catch |err| {
                    var err_buf = std.ArrayListUnmanaged(u8).empty;
                    defer err_buf.deinit(self.allocator);

                    err_buf.appendSlice(self.allocator, "{\"contents\":[{\"uri\":\"") catch |alloc_err| {
                        std.log.err("MCP: OOM formatting resource error: {t}", .{alloc_err});
                        return;
                    };
                    appendJsonEscaped(self.allocator, &err_buf, uri) catch |alloc_err| {
                        std.log.err("MCP: OOM escaping resource URI: {t}", .{alloc_err});
                        return;
                    };
                    err_buf.appendSlice(self.allocator, "\",\"mimeType\":\"text/plain\",\"text\":\"Error: ") catch |alloc_err| {
                        std.log.err("MCP: OOM formatting resource error mid: {t}", .{alloc_err});
                        return;
                    };
                    var err_msg_buf: [128]u8 = undefined;
                    const err_msg = std.fmt.bufPrint(&err_msg_buf, "{t}", .{err}) catch "unknown error";
                    appendJsonEscaped(self.allocator, &err_buf, err_msg) catch |alloc_err| {
                        std.log.err("MCP: OOM escaping resource error message: {t}", .{alloc_err});
                        return;
                    };
                    err_buf.appendSlice(self.allocator, "\"}]}") catch |alloc_err| {
                        std.log.err("MCP: OOM formatting resource error suffix: {t}", .{alloc_err});
                        return;
                    };
                    types.writeResponse(writer, rid, err_buf.items) catch |write_err| {
                        std.log.err("MCP: failed to write resource error response: {t}", .{write_err});
                        return;
                    };
                    return;
                };

                // Wrap result in MCP resource content format
                var out = std.ArrayListUnmanaged(u8).empty;
                defer out.deinit(self.allocator);

                try out.appendSlice(self.allocator, "{\"contents\":[{\"uri\":\"");
                try appendJsonEscaped(self.allocator, &out, uri);
                try out.appendSlice(self.allocator, "\",\"mimeType\":\"");
                try appendJsonEscaped(self.allocator, &out, resource.def.mime_type);
                try out.appendSlice(self.allocator, "\",\"text\":\"");
                try appendJsonEscaped(self.allocator, &out, result_buf.items);
                try out.appendSlice(self.allocator, "\"}]}");
                try types.writeResponse(writer, rid, out.items);
                return;
            }
        }

        try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Resource not found");
    }

    fn handleResourcesSubscribe(self: *Self, writer: anytype, id: ?types.RequestId, params: ?std.json.ObjectMap) !void {
        const rid = id orelse return;

        const p = params orelse {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing params");
            return;
        };

        const uri_val = p.get("uri") orelse {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing resource URI");
            return;
        };
        if (uri_val != .string) {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Resource URI must be string");
            return;
        }
        const uri = uri_val.string;

        _ = self.subscribeResource(uri) catch {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Resource not found");
            return;
        };

        try types.writeResponse(writer, rid, "{}");
    }

    fn handleResourcesUnsubscribe(self: *Self, writer: anytype, id: ?types.RequestId, params: ?std.json.ObjectMap) !void {
        const rid = id orelse return;

        const p = params orelse {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing params");
            return;
        };

        const uri_val = p.get("uri") orelse {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Missing resource URI");
            return;
        };
        if (uri_val != .string) {
            try types.writeError(writer, rid, types.ErrorCode.invalid_params, "Resource URI must be string");
            return;
        }
        const uri = uri_val.string;

        _ = self.unsubscribeResource(uri);

        try types.writeResponse(writer, rid, "{}");
    }
};

/// Append a JSON-escaped string to a buffer
fn appendJsonEscaped(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => {
                if (c < 0x20) {
                    var hex_buf: [6]u8 = undefined;
                    const hex = std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{c}) catch continue;
                    try buf.appendSlice(allocator, hex);
                } else {
                    try buf.append(allocator, c);
                }
            },
        }
    }
}

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

test "appendJsonEscaped" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonEscaped(allocator, &buf, "hello \"world\"");
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
    try server.handleMessage(msg, &writer);

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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage("not json at all", &writer);

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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage("[1,2,3]", &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Expected JSON object") != null);
}

test "handleMessage missing tool name in params" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

    try server.handleMessage(
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

// ═══════════════════════════════════════════════════════════════
// Resource Subscription Tests
// ═══════════════════════════════════════════════════════════════

test "subscribeResource tracks subscription" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addResource(.{
        .def = .{
            .uri = "abi://status",
            .name = "Status",
            .description = "Server status",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "ok");
            }
        }.handle,
    });

    try std.testing.expect(!server.isSubscribed("abi://status"));

    const was_new = try server.subscribeResource("abi://status");
    try std.testing.expect(was_new);
    try std.testing.expect(server.isSubscribed("abi://status"));

    // Subscribe again — should return false (already subscribed)
    const was_new2 = try server.subscribeResource("abi://status");
    try std.testing.expect(!was_new2);
}

test "subscribeResource rejects unknown URI" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    const result = server.subscribeResource("abi://nonexistent");
    try std.testing.expectError(error.ResourceNotFound, result);
}

test "unsubscribeResource removes subscription" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addResource(.{
        .def = .{
            .uri = "abi://data",
            .name = "Data",
            .description = "Some data",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "data");
            }
        }.handle,
    });

    _ = try server.subscribeResource("abi://data");
    try std.testing.expect(server.isSubscribed("abi://data"));

    const removed = server.unsubscribeResource("abi://data");
    try std.testing.expect(removed);
    try std.testing.expect(!server.isSubscribed("abi://data"));

    // Unsubscribe again — should return false
    const removed2 = server.unsubscribeResource("abi://data");
    try std.testing.expect(!removed2);
}

test "notifyResourceChanged sends notification for subscribed URI" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addResource(.{
        .def = .{
            .uri = "abi://metrics",
            .name = "Metrics",
            .description = "Server metrics",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "{}");
            }
        }.handle,
    });

    _ = try server.subscribeResource("abi://metrics");

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.notifyResourceChanged("abi://metrics", &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"notifications/resources/updated\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "abi://metrics") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "\"jsonrpc\":\"2.0\"") != null);
    // Notification must NOT have an "id" field
    try std.testing.expect(std.mem.indexOf(u8, written, "\"id\"") == null);
}

test "notifyResourceChanged is silent for unsubscribed URI" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.notifyResourceChanged("abi://unknown", &writer);

    // Nothing should be written
    try std.testing.expectEqual(@as(usize, 0), writer.end);
}

test "handleMessage resources/subscribe via JSON-RPC" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addResource(.{
        .def = .{
            .uri = "abi://live",
            .name = "Live",
            .description = "Live data",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "live");
            }
        }.handle,
    });

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.handleMessage(
        \\{"jsonrpc":"2.0","method":"resources/subscribe","id":20,"params":{"uri":"abi://live"}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"id\":20") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "\"result\":{}") != null);

    // Verify subscription is active
    try std.testing.expect(server.isSubscribed("abi://live"));
}

test "handleMessage resources/subscribe rejects unknown resource" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.handleMessage(
        \\{"jsonrpc":"2.0","method":"resources/subscribe","id":21,"params":{"uri":"abi://missing"}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Resource not found") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "-32602") != null);
}

test "handleMessage resources/unsubscribe via JSON-RPC" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addResource(.{
        .def = .{
            .uri = "abi://feed",
            .name = "Feed",
            .description = "Data feed",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "feed");
            }
        }.handle,
    });

    // Subscribe first
    _ = try server.subscribeResource("abi://feed");
    try std.testing.expect(server.isSubscribed("abi://feed"));

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.handleMessage(
        \\{"jsonrpc":"2.0","method":"resources/unsubscribe","id":22,"params":{"uri":"abi://feed"}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"id\":22") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "\"result\":{}") != null);

    // Verify unsubscribed
    try std.testing.expect(!server.isSubscribed("abi://feed"));
}

test "handleMessage resources/subscribe missing params" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.handleMessage(
        \\{"jsonrpc":"2.0","method":"resources/subscribe","id":23}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Missing params") != null);
}

test "handleMessage resources/subscribe missing URI" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    var out: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.handleMessage(
        \\{"jsonrpc":"2.0","method":"resources/subscribe","id":24,"params":{}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "Missing resource URI") != null);
}

test "handleMessage initialize advertises subscribe capability" {
    const allocator = std.testing.allocator;
    var server = Server.init(allocator, "test", "0.1.0");
    defer server.deinit();

    try server.addResource(.{
        .def = .{
            .uri = "abi://test",
            .name = "Test",
            .description = "Test resource",
        },
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, o: *std.ArrayListUnmanaged(u8)) !void {
                try o.appendSlice(std.testing.allocator, "test");
            }
        }.handle,
    });

    var out: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&out);

    try server.handleMessage(
        \\{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}
    , &writer);

    const written = out[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "\"subscribe\":true") != null);
}

test {
    std.testing.refAllDecls(@This());
}
