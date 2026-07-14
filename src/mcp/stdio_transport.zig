const std = @import("std");
const protocol = @import("protocol.zig");
const handlers = @import("handlers.zig");
const json_helpers = @import("json_helpers.zig");
const shutdown = @import("shutdown.zig");

const JsonRpcRequest = protocol.JsonRpcRequest;
const McpMethod = protocol.McpMethod;
const validateRequest = protocol.validateRequest;
const MAX_REQUEST_SIZE = protocol.MAX_REQUEST_SIZE;
const valueToJson = json_helpers.valueToJson;
const appendJsonString = json_helpers.appendJsonString;
const requestShutdown = shutdown.request;
const isShutdownRequested = shutdown.isRequested;

// --- Stdio Transport ---

pub fn runStdioLoop(allocator: std.mem.Allocator, io: std.Io) !void {
    var read_buf: [4096]u8 = undefined;
    var line_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer line_buf.deinit(allocator);

    // Portable stdin read via the std `Io` File API (works on POSIX + Windows);
    // replaces the POSIX-only `std.posix.read(STDIN_FILENO, ...)`.
    const stdin = std.Io.File.stdin();
    while (!isShutdownRequested()) {
        var bufs = [_][]u8{&read_buf};
        const n = stdin.readStreaming(io, &bufs) catch break;
        if (n == 0) break;

        for (read_buf[0..n]) |byte| {
            if (byte == '\n') {
                const line = std.mem.trimEnd(u8, line_buf.items, "\r");
                var arena = std.heap.ArenaAllocator.init(allocator);
                defer arena.deinit();
                processRequest(arena.allocator(), io, line) catch |err| {
                    std.log.warn("failed to process MCP request: {s}", .{@errorName(err)});
                };
                line_buf.clearRetainingCapacity();
                continue;
            }

            if (line_buf.items.len >= MAX_REQUEST_SIZE) {
                std.log.warn("dropping overlong MCP request line", .{});
                writeError(io, null, -32700, "Parse error");
                line_buf.clearRetainingCapacity();
                continue;
            }

            try line_buf.append(allocator, byte);
        }
    }

    if (line_buf.items.len > 0) {
        const line = std.mem.trimEnd(u8, line_buf.items, "\r");
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        processRequest(arena.allocator(), io, line) catch |err| {
            std.log.warn("failed to process MCP request: {s}", .{@errorName(err)});
        };
    }
}

fn processRequest(allocator: std.mem.Allocator, io: std.Io, line: []const u8) !void {
    if (line.len == 0) return;

    validateRequest(line) catch |err| {
        const message: []const u8 = switch (err) {
            error.JsonTooDeep => "Parse error: JSON nesting too deep",
            error.RequestTooLarge, error.InvalidJsonFormat, error.EmptyRequest => "Parse error",
        };
        writeError(io, null, -32700, message);
        return err;
    };

    const request = std.json.parseFromSlice(JsonRpcRequest, allocator, line, .{
        .ignore_unknown_fields = true,
    }) catch |err| {
        writeError(io, null, -32700, "Parse error");
        return err;
    };
    defer request.deinit();

    if (!std.mem.eql(u8, request.value.jsonrpc, "2.0")) {
        writeError(io, request.value.id, -32600, "Invalid Request");
        return;
    }

    const method = McpMethod.fromString(request.value.method);
    const result_json = switch (method) {
        .initialize => handlers.handleInitializeJson(allocator, request.value.params) catch |err| {
            writeError(io, request.value.id, -32603, "Internal error");
            return err;
        },
        .@"tools/list" => handlers.handleToolsListJson(allocator) catch |err| {
            writeError(io, request.value.id, -32603, "Internal error");
            return err;
        },
        .@"tools/call" => handlers.handleToolsCallJson(allocator, request.value.params) catch |err| {
            writeError(io, request.value.id, -32603, handlers.errorMessage(err));
            return err;
        },
        .ping => try allocator.dupe(u8, "{}"),
        .shutdown => blk: {
            // Only signal shutdown; `main` tears down the shared scheduler/store
            // after it joins the HTTP thread, so teardown never races a peer
            // transport's in-flight tool call (see src/mcp/main.zig).
            requestShutdown();
            break :blk try allocator.dupe(u8, "null");
        },
        .@"resources/list" => try allocator.dupe(u8, "{\"resources\":[]}"),
        .@"prompts/list" => try allocator.dupe(u8, "{\"prompts\":[]}"),
        .unknown => {
            writeError(io, request.value.id, -32601, "Method not found");
            return;
        },
    };
    defer allocator.free(result_json);

    writeResult(allocator, io, request.value.id, result_json);
}

fn buildError(gpa: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), id: ?std.json.Value, code: i32, message: []const u8) !void {
    try buf.appendSlice(gpa, "{\"jsonrpc\":\"2.0\"");
    try appendId(gpa, buf, id);
    try buf.appendSlice(gpa, ",\"error\":{\"code\":");
    try buf.print(gpa, "{d}", .{code});
    try buf.appendSlice(gpa, ",\"message\":");
    try appendJsonString(buf, gpa, message);
    try buf.appendSlice(gpa, "}}\n");
}

fn writeError(io: std.Io, id: ?std.json.Value, code: i32, message: []const u8) void {
    const gpa = std.heap.page_allocator;
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(gpa);

    // Don't drop the error frame silently: log like the write path below so an
    // OOM while serializing leaves a trace instead of a vanished response.
    buildError(gpa, &buf, id, code, message) catch |err| {
        std.log.warn("failed to build MCP error response: {s}", .{@errorName(err)});
        return;
    };

    writeStdoutAll(io, buf.items) catch |err| {
        std.log.warn("failed to write MCP error response: {s}", .{@errorName(err)});
    };
}

fn buildResult(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), id: ?std.json.Value, result_json: []const u8) !void {
    try buf.appendSlice(allocator, "{\"jsonrpc\":\"2.0\"");
    try appendId(allocator, buf, id);
    try buf.appendSlice(allocator, ",\"result\":");
    try buf.appendSlice(allocator, result_json);
    try buf.appendSlice(allocator, "}\n");
}

fn writeResult(allocator: std.mem.Allocator, io: std.Io, id: ?std.json.Value, result_json: []const u8) void {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(allocator);

    buildResult(allocator, &buf, id, result_json) catch |err| {
        std.log.warn("failed to build MCP response: {s}", .{@errorName(err)});
        return;
    };

    writeStdoutAll(io, buf.items) catch |err| {
        std.log.warn("failed to write MCP response: {s}", .{@errorName(err)});
    };
}

fn appendId(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), id: ?std.json.Value) !void {
    if (id) |value| {
        const id_json = try valueToJson(value, allocator);
        defer allocator.free(id_json);
        try out.appendSlice(allocator, ",\"id\":");
        try out.appendSlice(allocator, id_json);
    } else {
        try out.appendSlice(allocator, ",\"id\":null");
    }
}

fn writeStdoutAll(io: std.Io, bytes: []const u8) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &buffer);
    const stdout = &stdout_writer.interface;
    try stdout.writeAll(bytes);
    try stdout.flush();
}

test {
    std.testing.refAllDecls(@This());
}
