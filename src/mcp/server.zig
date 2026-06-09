const std = @import("std");
const protocol = @import("protocol.zig");
const handlers = @import("handlers.zig");
const json_helpers = @import("json_helpers.zig");
const state = @import("state.zig");

const DEFAULT_HTTP_PORT: u16 = 8080;
const HTTP_PORT_ENV = "ABI_MCP_HTTP_PORT";

const JsonRpcRequest = protocol.JsonRpcRequest;
const McpMethod = protocol.McpMethod;
const validateRequest = protocol.validateRequest;
const MAX_REQUEST_SIZE = protocol.MAX_REQUEST_SIZE;
const valueToJson = json_helpers.valueToJson;
const appendJsonString = json_helpers.appendJsonString;

// --- Shutdown Coordination ---

var g_shutdown_requested = std.atomic.Value(bool).init(false);

pub fn requestShutdown() void {
    g_shutdown_requested.store(true, .release);
}

pub fn isShutdownRequested() bool {
    return g_shutdown_requested.load(.acquire);
}

pub fn installSignalHandlers() void {
    const posix = std.posix;
    const handler = posix.Sigaction{
        .handler = .{ .handler = signalHandler },
        .mask = posix.sigemptyset(),
        .flags = 0,
    };
    posix.sigaction(posix.SIG.INT, &handler, null);
    posix.sigaction(posix.SIG.TERM, &handler, null);
}

fn signalHandler(sig: @TypeOf(std.posix.SIG.INT)) callconv(.c) void {
    _ = sig;
    g_shutdown_requested.store(true, .release);
}

// --- Stdio Transport ---

pub fn runStdioLoop(allocator: std.mem.Allocator, io: std.Io) !void {
    var read_buf: [4096]u8 = undefined;
    var line_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer line_buf.deinit(allocator);

    while (!isShutdownRequested()) {
        const n = std.posix.read(std.posix.STDIN_FILENO, &read_buf) catch break;
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
        const code: i32 = switch (err) {
            error.RequestTooLarge => -32700,
            error.InvalidJsonFormat => -32700,
            else => -32700,
        };
        writeError(io, null, code, "Parse error");
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
            const msg = switch (err) {
                error.MissingParams => "Missing params",
                error.MissingToolName => "Missing tool name",
                error.MissingArguments => "Missing arguments",
                error.MissingInput => "Missing input",
                error.MissingProfile => "Missing profile",
                error.MissingDataset => "Missing dataset",
                error.MissingQuery => "Missing query",
                error.UnknownTool => "Method not found",
                else => "Internal error",
            };
            writeError(io, request.value.id, -32603, msg);
            return err;
        },
        .ping => try allocator.dupe(u8, "{}"),
        .shutdown => blk: {
            requestShutdown();
            state.deinitWdbxStore();
            state.deinitScheduler();
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

fn writeError(io: std.Io, id: ?std.json.Value, code: i32, message: []const u8) void {
    const gpa = std.heap.page_allocator;
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(gpa);

    buf.appendSlice(gpa, "{\"jsonrpc\":\"2.0\"") catch return;
    appendId(gpa, &buf, id) catch return;
    buf.appendSlice(gpa, ",\"error\":{\"code\":") catch return;
    buf.print(gpa, "{d}", .{code}) catch return;
    buf.appendSlice(gpa, ",\"message\":") catch return;
    appendJsonString(&buf, gpa, message) catch return;
    buf.appendSlice(gpa, "}}\n") catch return;

    writeStdoutAll(io, buf.items) catch |err| {
        std.log.warn("failed to write MCP error response: {s}", .{@errorName(err)});
    };
}

fn writeResult(allocator: std.mem.Allocator, io: std.Io, id: ?std.json.Value, result_json: []const u8) void {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(allocator);

    buf.appendSlice(allocator, "{\"jsonrpc\":\"2.0\"") catch return;
    appendId(allocator, &buf, id) catch return;
    buf.appendSlice(allocator, ",\"result\":") catch return;
    buf.appendSlice(allocator, result_json) catch return;
    buf.appendSlice(allocator, "}\n") catch return;

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

// --- HTTP/SSE Transport ---

pub fn runHttpServer(allocator: std.mem.Allocator, io: std.Io) void {
    const net = std.Io.net;
    const port = configuredHttpPort();
    const address = net.IpAddress.parseIp4("127.0.0.1", port) catch {
        std.log.warn("failed to parse HTTP listen address", .{});
        return;
    };
    var tcp_server = address.listen(io, .{
        .reuse_address = true,
    }) catch |err| {
        std.log.warn("failed to bind HTTP server on port {d}: {s}; set {s} to choose another loopback port", .{ port, @errorName(err), HTTP_PORT_ENV });
        return;
    };
    defer tcp_server.deinit(io);

    std.log.info("MCP HTTP/SSE server listening on http://127.0.0.1:{d}", .{port});

    while (!isShutdownRequested()) {
        const conn = tcp_server.accept(io) catch |err| {
            if (isShutdownRequested()) break;
            std.log.warn("HTTP accept error: {s}", .{@errorName(err)});
            continue;
        };
        handleHttpConnection(allocator, io, conn) catch |err| {
            std.log.warn("HTTP connection error: {s}", .{@errorName(err)});
        };
    }
}

pub fn wakeHttpServer(io: std.Io) void {
    const net = std.Io.net;
    const address = net.IpAddress.parseIp4("127.0.0.1", configuredHttpPort()) catch return;
    const stream = address.connect(io, .{ .mode = .stream }) catch return;
    defer stream.close(io);
}

pub fn configuredHttpPort() u16 {
    const raw = std.c.getenv(HTTP_PORT_ENV) orelse return DEFAULT_HTTP_PORT;
    return parseHttpPort(std.mem.span(raw)) orelse DEFAULT_HTTP_PORT;
}

fn parseHttpPort(raw: []const u8) ?u16 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    const port = std.fmt.parseInt(u16, trimmed, 10) catch return null;
    if (port == 0) return null;
    return port;
}

fn handleHttpConnection(allocator: std.mem.Allocator, io: std.Io, conn: std.Io.net.Stream) !void {
    defer conn.close(io);

    var read_buf: [MAX_REQUEST_SIZE]u8 = undefined;
    var read_vec: [1][]u8 = .{&read_buf};
    const n = conn.read(io, &read_vec) catch return;
    if (n == 0) return;
    const raw_request = read_buf[0..n];

    var line_end: usize = 0;
    while (line_end < raw_request.len and raw_request[line_end] != '\n') : (line_end += 1) {}
    const request_line = std.mem.trimEnd(u8, raw_request[0..line_end], "\r");

    var req_it = std.mem.splitScalar(u8, request_line, ' ');
    const http_method = req_it.next() orelse return;
    const path = req_it.next() orelse return;

    if (std.mem.eql(u8, http_method, "GET") and std.mem.eql(u8, path, "/sse")) {
        const sse_response =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n" ++
            "event: endpoint\r\ndata: /message\r\n\r\n";
        try writeHttpAll(io, conn, sse_response);
        return;
    }

    if (std.mem.eql(u8, http_method, "POST") and std.mem.eql(u8, path, "/message")) {
        const body = findHttpBody(raw_request) orelse {
            const err_resp = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"no body\"}";
            try writeHttpAll(io, conn, err_resp);
            return;
        };

        if (body.len > MAX_REQUEST_SIZE) {
            const err_resp = "HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"request too large\"}";
            try writeHttpAll(io, conn, err_resp);
            return;
        }

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        const result_json = processJsonRpc(arena.allocator(), body) catch |err| {
            const err_body = try std.fmt.allocPrint(arena.allocator(), "{{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{{\"code\":-32603,\"message\":\"{s}\"}}}}", .{@errorName(err)});
            const header = try std.fmt.allocPrint(arena.allocator(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{err_body.len});
            try writeHttpAll(io, conn, header);
            try writeHttpAll(io, conn, err_body);
            return;
        };

        const header = try std.fmt.allocPrint(arena.allocator(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{result_json.len});
        try writeHttpAll(io, conn, header);
        try writeHttpAll(io, conn, result_json);
        return;
    }

    const resp = "HTTP/1.1 405 Method Not Allowed\r\nConnection: close\r\n\r\n";
    try writeHttpAll(io, conn, resp);
}

fn writeHttpAll(io: std.Io, conn: std.Io.net.Stream, bytes: []const u8) !void {
    var buffer: [4096]u8 = undefined;
    var stream_writer = conn.writer(io, &buffer);
    const writer = &stream_writer.interface;
    try writer.writeAll(bytes);
    try writer.flush();
}

fn findHttpBody(raw: []const u8) ?[]const u8 {
    const needle = "\r\n\r\n";
    const idx = std.mem.indexOf(u8, raw, needle) orelse return null;
    const body_start = idx + needle.len;
    if (body_start >= raw.len) return null;
    return raw[body_start..];
}

test "MCP HTTP port parser accepts valid user ports" {
    try std.testing.expectEqual(@as(?u16, 18080), parseHttpPort("18080"));
    try std.testing.expectEqual(@as(?u16, 18080), parseHttpPort(" 18080\n"));
}

// Keep invalid overrides non-fatal so stdio mode still works when users typo the environment.
test "MCP HTTP port parser rejects invalid overrides" {
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort(""));
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort("0"));
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort("65536"));
    try std.testing.expectEqual(@as(?u16, null), parseHttpPort("not-a-port"));
}

pub fn processJsonRpc(allocator: std.mem.Allocator, body: []const u8) ![]u8 {
    const request = std.json.parseFromSlice(JsonRpcRequest, allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch return error.ParseError;
    defer request.deinit();

    if (!std.mem.eql(u8, request.value.jsonrpc, "2.0")) return error.InvalidRequest;

    const method = McpMethod.fromString(request.value.method);
    const result_json = switch (method) {
        .initialize => try handlers.handleInitializeJson(allocator, request.value.params),
        .@"tools/list" => try handlers.handleToolsListJson(allocator),
        .@"tools/call" => try handlers.handleToolsCallJson(allocator, request.value.params),
        .ping => try allocator.dupe(u8, "{}"),
        .shutdown => blk: {
            requestShutdown();
            state.deinitWdbxStore();
            state.deinitScheduler();
            break :blk try allocator.dupe(u8, "null");
        },
        .@"resources/list" => try allocator.dupe(u8, "{\"resources\":[]}"),
        .@"prompts/list" => try allocator.dupe(u8, "{\"prompts\":[]}"),
        .unknown => return error.MethodNotFound,
    };
    defer allocator.free(result_json);

    const id_json = if (request.value.id) |id_val| try valueToJson(id_val, allocator) else try allocator.dupe(u8, "null");
    defer allocator.free(id_json);

    return try std.fmt.allocPrint(allocator, "{{\"jsonrpc\":\"2.0\",\"id\":{s},\"result\":{s}}}", .{ id_json, result_json });
}

test "processJsonRpc handles tools list and preserves ids" {
    const allocator = std.testing.allocator;

    const numeric = try processJsonRpc(allocator, "{\"jsonrpc\":\"2.0\",\"id\":42,\"method\":\"tools/list\"}");
    defer allocator.free(numeric);
    try std.testing.expect(std.mem.indexOf(u8, numeric, "\"id\":42") != null);
    try std.testing.expect(std.mem.indexOf(u8, numeric, "\"tools\"") != null);

    const string_id = try processJsonRpc(allocator, "{\"jsonrpc\":\"2.0\",\"id\":\"abc\",\"method\":\"ping\"}");
    defer allocator.free(string_id);
    try std.testing.expect(std.mem.indexOf(u8, string_id, "\"id\":\"abc\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, string_id, "\"result\":{}") != null);
}

test "processJsonRpc rejects invalid requests" {
    try std.testing.expectError(error.ParseError, processJsonRpc(std.testing.allocator, "not json"));
    try std.testing.expectError(error.InvalidRequest, processJsonRpc(std.testing.allocator, "{\"jsonrpc\":\"1.0\",\"method\":\"ping\"}"));
    try std.testing.expectError(error.MethodNotFound, processJsonRpc(std.testing.allocator, "{\"jsonrpc\":\"2.0\",\"method\":\"unknown\"}"));
}
