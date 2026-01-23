const std = @import("std");

const wdbx = @import("wdbx.zig");
const db_helpers = @import("db_helpers.zig");
const json_utils = @import("../shared/utils_combined.zig").json;
const net_utils = @import("../shared/utils_combined.zig").net;

pub const HttpError = std.mem.Allocator.Error || error{
    InvalidAddress,
    InvalidRequest,
    InvalidVector,
    ReadFailed,
    RequestTooLarge,
    Unauthorized,
};

const max_body_bytes = 1024 * 1024;

/// Server configuration including authentication settings
pub const ServerConfig = struct {
    /// Bearer token for authentication (null = no auth required)
    auth_token: ?[]const u8 = null,
    /// Allow unauthenticated access to health endpoint
    allow_health_without_auth: bool = true,
    /// Allow unauthenticated access to stats endpoint
    allow_stats_without_auth: bool = false,
};

pub fn serve(allocator: std.mem.Allocator, address: []const u8) !void {
    var handle = try wdbx.createDatabase(allocator, "http");
    defer wdbx.closeDatabase(&handle);

    try serveDatabaseWithConfig(allocator, &handle, address, .{});
}

/// Serve with authentication enabled
pub fn serveWithAuth(allocator: std.mem.Allocator, address: []const u8, auth_token: []const u8) !void {
    var handle = try wdbx.createDatabase(allocator, "http");
    defer wdbx.closeDatabase(&handle);

    try serveDatabaseWithConfig(allocator, &handle, address, .{ .auth_token = auth_token });
}

pub fn serveDatabase(
    allocator: std.mem.Allocator,
    handle: *wdbx.DatabaseHandle,
    address: []const u8,
) !void {
    try serveDatabaseWithConfig(allocator, handle, address, .{});
}

pub fn serveDatabaseWithConfig(
    allocator: std.mem.Allocator,
    handle: *wdbx.DatabaseHandle,
    address: []const u8,
    config: ServerConfig,
) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    std.Io.Dir.cwd().createDir(io, "backups", .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) {
            std.log.err("Failed to create backups directory: {t}", .{err});
            return err;
        }
    };

    const listen_addr = try resolveAddress(io, allocator, address);
    var server = try listen_addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    std.debug.print("Database HTTP server listening on {s}\n", .{address});

    while (true) {
        var stream = server.accept(io) catch |err| {
            std.debug.print("Database HTTP accept error: {t}\n", .{err});
            continue;
        };
        defer stream.close(io);
        handleConnectionWithConfig(allocator, io, handle, stream, config) catch |err| {
            std.debug.print("Database HTTP connection error: {t}\n", .{err});
        };
    }
}

fn resolveAddress(
    io: std.Io,
    allocator: std.mem.Allocator,
    address: []const u8,
) !std.Io.net.IpAddress {
    var host_port = net_utils.parseHostPort(allocator, address) catch
        return HttpError.InvalidAddress;
    defer host_port.deinit(allocator);
    return std.Io.net.IpAddress.resolve(io, host_port.host, host_port.port) catch
        return HttpError.InvalidAddress;
}

fn handleConnection(
    allocator: std.mem.Allocator,
    io: std.Io,
    handle: *wdbx.DatabaseHandle,
    stream: std.Io.net.Stream,
) !void {
    return handleConnectionWithConfig(allocator, io, handle, stream, .{});
}

fn handleConnectionWithConfig(
    allocator: std.mem.Allocator,
    io: std.Io,
    handle: *wdbx.DatabaseHandle,
    stream: std.Io.net.Stream,
    config: ServerConfig,
) !void {
    var send_buffer: [4096]u8 = undefined;
    var recv_buffer: [4096]u8 = undefined;
    var connection_reader = stream.reader(io, &recv_buffer);
    var connection_writer = stream.writer(io, &send_buffer);
    var server: std.http.Server = .init(
        &connection_reader.interface,
        &connection_writer.interface,
    );

    while (true) {
        var request = server.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return err,
        };
        dispatchRequestWithConfig(allocator, handle, &request, config) catch |err| {
            std.debug.print("Database HTTP request error: {t}\n", .{err});
            const error_body = if (err == HttpError.Unauthorized)
                "{\"error\":\"unauthorized\"}"
            else
                "{\"error\":\"internal server error\"}";
            const status: std.http.Status = if (err == HttpError.Unauthorized)
                .unauthorized
            else
                .internal_server_error;
            respondJson(
                &request,
                error_body,
                status,
            ) catch |respond_err| {
                std.log.err("Failed to send error response: {t} (original error: {t})", .{
                    respond_err,
                    err,
                });
            };
            return;
        };
    }
}

fn dispatchRequest(
    allocator: std.mem.Allocator,
    handle: *wdbx.DatabaseHandle,
    request: *std.http.Server.Request,
) !void {
    return dispatchRequestWithConfig(allocator, handle, request, .{});
}

fn dispatchRequestWithConfig(
    allocator: std.mem.Allocator,
    handle: *wdbx.DatabaseHandle,
    request: *std.http.Server.Request,
    config: ServerConfig,
) !void {
    const target = request.head.target;
    const parts = splitTarget(target);

    // Health endpoint - optionally allow without auth
    if (std.mem.eql(u8, parts.path, "/health")) {
        if (!config.allow_health_without_auth) {
            try validateAuth(request, config);
        }
        return respondText(request, "ok\n", .ok);
    }

    // Stats endpoint - optionally allow without auth
    if (std.mem.eql(u8, parts.path, "/stats")) {
        if (!config.allow_stats_without_auth) {
            try validateAuth(request, config);
        }
        const stats = wdbx.getStats(handle);
        const body = try buildStatsJson(allocator, stats);
        defer allocator.free(body);
        return respondJson(request, body, .ok);
    }

    // All other endpoints require authentication if configured
    try validateAuth(request, config);

    if (std.mem.eql(u8, parts.path, "/backup")) {
        return handleBackup(handle, request, parts.query);
    }

    if (std.mem.eql(u8, parts.path, "/restore")) {
        return handleRestore(handle, request, parts.query);
    }

    if (std.mem.eql(u8, parts.path, "/optimize")) {
        return handleOptimize(handle, request);
    }

    if (std.mem.eql(u8, parts.path, "/vectors")) {
        return handleVectors(allocator, handle, request, parts.query);
    }

    if (std.mem.eql(u8, parts.path, "/search")) {
        return handleSearch(allocator, handle, request, parts.query);
    }

    return respondJson(request, "{\"error\":\"not found\"}", .not_found);
}

fn handleBackup(
    handle: *wdbx.DatabaseHandle,
    request: *std.http.Server.Request,
    query: []const u8,
) !void {
    if (request.head.method != .POST) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
    }
    const path = getQueryParam(query, "path") orelse
        return respondJson(request, "{\"error\":\"missing path\"}", .bad_request);
    if (path.len == 0) {
        return respondJson(request, "{\"error\":\"missing path\"}", .bad_request);
    }
    try wdbx.backup(handle, path);
    return respondJson(request, "{\"status\":\"backed up\"}", .ok);
}

fn handleRestore(
    handle: *wdbx.DatabaseHandle,
    request: *std.http.Server.Request,
    query: []const u8,
) !void {
    if (request.head.method != .POST) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
    }
    const path = getQueryParam(query, "path") orelse
        return respondJson(request, "{\"error\":\"missing path\"}", .bad_request);
    if (path.len == 0) {
        return respondJson(request, "{\"error\":\"missing path\"}", .bad_request);
    }
    try wdbx.restore(handle, path);
    return respondJson(request, "{\"status\":\"restored\"}", .ok);
}

fn handleOptimize(handle: *wdbx.DatabaseHandle, request: *std.http.Server.Request) !void {
    if (request.head.method != .POST) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
    }
    try wdbx.optimize(handle);
    return respondJson(request, "{\"status\":\"optimized\"}", .ok);
}

fn handleVectors(
    allocator: std.mem.Allocator,
    handle: *wdbx.DatabaseHandle,
    request: *std.http.Server.Request,
    query: []const u8,
) !void {
    const method = request.head.method;

    if (method == .GET) {
        if (getQueryParam(query, "id")) |id_text| {
            const id = std.fmt.parseInt(u64, id_text, 10) catch {
                return respondJson(request, "{\"error\":\"invalid id\"}", .bad_request);
            };
            const view = wdbx.getVector(handle, id) orelse
                return respondJson(request, "{\"error\":\"not found\"}", .not_found);
            const body = try buildVectorJson(allocator, view);
            defer allocator.free(body);
            return respondJson(request, body, .ok);
        }

        const limit = parseQueryInt(query, "limit", usize, 25);
        const views = try wdbx.listVectors(handle, allocator, limit);
        defer allocator.free(views);
        const body = try buildVectorListJson(allocator, views);
        defer allocator.free(body);
        return respondJson(request, body, .ok);
    }

    if (method == .POST or method == .PUT) {
        const id_text = getQueryParam(query, "id") orelse
            return respondJson(request, "{\"error\":\"missing id\"}", .bad_request);
        const id = std.fmt.parseInt(u64, id_text, 10) catch {
            return respondJson(request, "{\"error\":\"invalid id\"}", .bad_request);
        };
        const body = readRequestBody(allocator, request) catch |err| switch (err) {
            HttpError.RequestTooLarge => {
                return respondJson(
                    request,
                    "{\"error\":\"payload too large\"}",
                    .payload_too_large,
                );
            },
            HttpError.ReadFailed => {
                return respondJson(request, "{\"error\":\"invalid request body\"}", .bad_request);
            },
            else => return err,
        };
        defer allocator.free(body);
        const vector = db_helpers.parseVector(allocator, body) catch {
            return respondJson(request, "{\"error\":\"invalid vector\"}", .bad_request);
        };
        defer allocator.free(vector);

        if (method == .POST) {
            const meta = getQueryParam(query, "meta");
            wdbx.insertVector(handle, id, vector, meta) catch |err| switch (err) {
                error.DuplicateId => {
                    return respondJson(request, "{\"error\":\"duplicate id\"}", .conflict);
                },
                else => return err,
            };
            return respondJson(request, "{\"status\":\"inserted\"}", .created);
        }

        const updated = try wdbx.updateVector(handle, id, vector);
        if (!updated) {
            return respondJson(request, "{\"error\":\"not found\"}", .not_found);
        }
        return respondJson(request, "{\"status\":\"updated\"}", .ok);
    }

    if (method == .DELETE) {
        const id_text = getQueryParam(query, "id") orelse
            return respondJson(request, "{\"error\":\"missing id\"}", .bad_request);
        const id = std.fmt.parseInt(u64, id_text, 10) catch {
            return respondJson(request, "{\"error\":\"invalid id\"}", .bad_request);
        };
        if (!wdbx.deleteVector(handle, id)) {
            return respondJson(request, "{\"error\":\"not found\"}", .not_found);
        }
        return respondJson(request, "{\"status\":\"deleted\"}", .ok);
    }

    return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
}

fn handleSearch(
    allocator: std.mem.Allocator,
    handle: *wdbx.DatabaseHandle,
    request: *std.http.Server.Request,
    query: []const u8,
) !void {
    if (request.head.method != .GET) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed);
    }

    const vector_text = getQueryParam(query, "vector") orelse
        return respondJson(request, "{\"error\":\"missing vector\"}", .bad_request);

    const vector = db_helpers.parseVector(allocator, vector_text) catch {
        return respondJson(request, "{\"error\":\"invalid vector\"}", .bad_request);
    };
    defer allocator.free(vector);

    const top_k = parseQueryInt(query, "top_k", usize, 3);
    const results = try wdbx.searchVectors(handle, allocator, vector, top_k);
    defer allocator.free(results);

    const body = try buildSearchResultsJson(allocator, results);
    defer allocator.free(body);
    return respondJson(request, body, .ok);
}

fn readRequestBody(
    allocator: std.mem.Allocator,
    request: *std.http.Server.Request,
) HttpError![]u8 {
    var buffer: [4096]u8 = undefined;
    const reader = request.readerExpectContinue(&buffer) catch
        return HttpError.ReadFailed;
    return readAll(reader, allocator, max_body_bytes);
}

fn readAll(
    reader: *std.Io.Reader,
    allocator: std.mem.Allocator,
    limit: usize,
) HttpError![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);

    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = reader.readSliceShort(chunk[0..]) catch
            return HttpError.ReadFailed;
        if (n == 0) break;
        if (list.items.len + n > limit) return HttpError.RequestTooLarge;
        try list.appendSlice(allocator, chunk[0..n]);
        if (n < chunk.len) break;
    }
    return list.toOwnedSlice(allocator);
}

/// Validate authentication for a request.
/// Returns success if no auth is configured or if valid token is provided.
fn validateAuth(request: *std.http.Server.Request, config: ServerConfig) HttpError!void {
    const expected_token = config.auth_token orelse return; // No auth configured

    // Look for Authorization header in the request
    const auth_header = getAuthorizationHeader(request) orelse {
        return HttpError.Unauthorized;
    };

    // Expect "Bearer <token>" format
    const bearer_prefix = "Bearer ";
    if (!std.mem.startsWith(u8, auth_header, bearer_prefix)) {
        return HttpError.Unauthorized;
    }

    const provided_token = auth_header[bearer_prefix.len..];

    // Timing-safe comparison to prevent timing attacks
    if (provided_token.len != expected_token.len or
        !timingSafeEqual(provided_token, expected_token))
    {
        return HttpError.Unauthorized;
    }
}

/// Timing-safe byte comparison to prevent timing attacks
fn timingSafeEqual(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, b) |x, y| {
        diff |= x ^ y;
    }
    return diff == 0;
}

/// Get the Authorization header value from a request by parsing the raw header buffer
fn getAuthorizationHeader(request: *std.http.Server.Request) ?[]const u8 {
    // In Zig 0.16, we need to parse the raw header buffer to find custom headers
    // The head_buffer contains the raw HTTP headers
    return findHeaderInBuffer(request.head_buffer, "authorization");
}

/// Find a header value in the raw HTTP header buffer (case-insensitive)
fn findHeaderInBuffer(buffer: []const u8, header_name: []const u8) ?[]const u8 {
    var it = std.mem.splitSequence(u8, buffer, "\r\n");
    // Skip the request line
    _ = it.next();

    while (it.next()) |line| {
        if (line.len == 0) break; // End of headers

        const colon_idx = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = line[0..colon_idx];

        if (std.ascii.eqlIgnoreCase(name, header_name)) {
            const value_start = colon_idx + 1;
            if (value_start >= line.len) return "";
            // Trim leading whitespace from value
            const value = std.mem.trim(u8, line[value_start..], " \t");
            return value;
        }
    }
    return null;
}

const TargetParts = struct {
    path: []const u8,
    query: []const u8,
};

fn splitTarget(target: []const u8) TargetParts {
    if (std.mem.indexOfScalar(u8, target, '?')) |idx| {
        return .{
            .path = target[0..idx],
            .query = target[idx + 1 ..],
        };
    }
    return .{ .path = target, .query = "" };
}

fn getQueryParam(query: []const u8, key: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, query, '&');
    while (it.next()) |pair| {
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
        const name = pair[0..eq];
        if (std.mem.eql(u8, name, key)) {
            return pair[eq + 1 ..];
        }
    }
    return null;
}

/// URL-decode a percent-encoded string.
/// Caller owns the returned memory.
fn urlDecode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var output = std.ArrayListUnmanaged(u8){};
    errdefer output.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) {
        if (input[i] == '%' and i + 2 < input.len) {
            const hex = input[i + 1 .. i + 3];
            const byte = std.fmt.parseInt(u8, hex, 16) catch {
                // Invalid hex, keep literal
                try output.append(allocator, input[i]);
                i += 1;
                continue;
            };
            try output.append(allocator, byte);
            i += 3;
        } else if (input[i] == '+') {
            // '+' is commonly used for spaces in query strings
            try output.append(allocator, ' ');
            i += 1;
        } else {
            try output.append(allocator, input[i]);
            i += 1;
        }
    }

    return output.toOwnedSlice(allocator);
}

/// Get and URL-decode a query parameter.
/// Caller owns the returned memory if not null.
fn getQueryParamDecoded(allocator: std.mem.Allocator, query: []const u8, key: []const u8) !?[]u8 {
    const raw = getQueryParam(query, key) orelse return null;
    return try urlDecode(allocator, raw);
}

fn parseQueryInt(
    query: []const u8,
    key: []const u8,
    comptime T: type,
    default_value: T,
) T {
    const text = getQueryParam(query, key) orelse return default_value;
    return std.fmt.parseInt(T, text, 10) catch default_value;
}

fn respondJson(
    request: *std.http.Server.Request,
    body: []const u8,
    status: std.http.Status,
) !void {
    const headers = [_]std.http.Header{
        .{ .name = "content-type", .value = "application/json" },
    };
    try request.respond(body, .{
        .status = status,
        .extra_headers = &headers,
    });
}

fn respondText(
    request: *std.http.Server.Request,
    body: []const u8,
    status: std.http.Status,
) !void {
    const headers = [_]std.http.Header{
        .{ .name = "content-type", .value = "text/plain; charset=utf-8" },
    };
    try request.respond(body, .{
        .status = status,
        .extra_headers = &headers,
    });
}

fn buildStatsJson(allocator: std.mem.Allocator, stats: wdbx.Stats) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{{\"count\":{d},\"dimension\":{d}}}",
        .{ stats.count, stats.dimension },
    );
}

fn buildVectorJson(allocator: std.mem.Allocator, view: wdbx.VectorView) ![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);

    try list.appendSlice(allocator, "{\"id\":");
    try list.print(allocator, "{d}", .{view.id});
    try list.appendSlice(allocator, ",\"vector\":");
    try appendVectorJson(&list, allocator, view.vector);
    try list.appendSlice(allocator, ",\"metadata\":");
    if (view.metadata) |meta| {
        const escaped = try json_utils.escapeString(allocator, meta);
        defer allocator.free(escaped);
        try list.appendSlice(allocator, escaped);
    } else {
        try list.appendSlice(allocator, "null");
    }
    try list.appendSlice(allocator, "}");

    return list.toOwnedSlice(allocator);
}

fn buildVectorListJson(allocator: std.mem.Allocator, views: []const wdbx.VectorView) ![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);
    try list.appendSlice(allocator, "{\"vectors\":[");
    for (views, 0..) |view, i| {
        if (i > 0) try list.appendSlice(allocator, ",");
        const entry = try buildVectorJson(allocator, view);
        defer allocator.free(entry);
        try list.appendSlice(allocator, entry);
    }
    try list.appendSlice(allocator, "]}");
    return list.toOwnedSlice(allocator);
}

fn buildSearchResultsJson(
    allocator: std.mem.Allocator,
    results: []const wdbx.SearchResult,
) ![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);
    try list.appendSlice(allocator, "{\"results\":[");
    for (results, 0..) |result, i| {
        if (i > 0) try list.appendSlice(allocator, ",");
        try list.appendSlice(allocator, "{\"id\":");
        try list.print(allocator, "{d}", .{result.id});
        try list.appendSlice(allocator, ",\"score\":");
        try list.print(allocator, "{d:.6}", .{result.score});
        try list.appendSlice(allocator, "}");
    }
    try list.appendSlice(allocator, "]}");
    return list.toOwnedSlice(allocator);
}

fn appendVectorJson(
    list: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    vector: []const f32,
) !void {
    try list.append(allocator, '[');
    for (vector, 0..) |value, i| {
        if (i > 0) try list.append(allocator, ',');
        try list.print(allocator, "{d:.6}", .{value});
    }
    try list.append(allocator, ']');
}
