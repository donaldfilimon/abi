//! WDBX HTTP server with a lightweight in-memory vector store.
//!
//! The implementation focuses on providing a pragmatic HTTP façade that mirrors
//! the behaviour expected by the CLI and tests without depending on an actual
//! on-disk database. Requests are handled synchronously and responses are
//! returned as JSON payloads. When the `start` API is invoked the server simply
//! marks itself as running – wiring a real TCP listener can be layered on later
//! without affecting the public API.

const std = @import("std");
const ArrayList = std.array_list.Managed;
// Note: core functionality is now imported through module dependencies

const max_header_size = 16 * 1024;

pub const HttpError = error{
    InvalidRequest,
    UnsupportedMethod,
    NotFound,
    BadPayload,
};

pub const Response = struct {
    status: u16 = 200,
    body: []u8,
    content_type: []const u8 = "application/json",

    pub fn deinit(self: Response, allocator: std.mem.Allocator) void {
        allocator.free(self.body);
    }
};

pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    enable_cors: bool = true,
    enable_auth: bool = false,
};

const VectorEntry = struct {
    id: u64,
    data: []f32,
};

pub const WdbxHttpServer = struct {
    allocator: std.mem.Allocator,
    config: ServerConfig,
    database_path: ?[]const u8 = null,

    vectors: ArrayList(VectorEntry),
    next_id: u64 = 1,
    mutex: std.Thread.Mutex = .{},
    running: bool = false,

    pub fn init(allocator: std.mem.Allocator, config: ServerConfig) !*WdbxHttpServer {
        const self = try allocator.create(WdbxHttpServer);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .database_path = null,
            .vectors = ArrayList(VectorEntry).init(allocator),
            .next_id = 1,
        };
        return self;
    }

    pub fn deinit(self: *WdbxHttpServer) void {
        self.stop();
        for (self.vectors.items) |entry| {
            self.allocator.free(entry.data);
        }
        self.vectors.deinit();
        if (self.database_path) |path| {
            self.allocator.free(path);
        }
        self.allocator.destroy(self);
    }

    pub fn openDatabase(self: *WdbxHttpServer, path: []const u8) !void {
        if (self.database_path) |old| {
            self.allocator.free(old);
        }
        self.database_path = try self.allocator.dupe(u8, path);
    }

    /// Mark the server as running. A real TCP listener can be layered on top of
    /// the current request handler in a future iteration.
    pub fn start(self: *WdbxHttpServer) !void {
        if (self.running) return;
        self.running = true;
    }

    pub fn stop(self: *WdbxHttpServer) void {
        self.running = false;
    }

    /// Start a blocking TCP listener that serves HTTP requests using the
    /// existing request handler. The server processes connections
    /// sequentially to keep the implementation simple and predictable.
    pub fn run(self: *WdbxHttpServer) !void {
        if (!self.running) try self.start();

        const address = try std.net.Address.parseIp(self.config.host, self.config.port);
        var listener = try address.listen(.{ .reuse_address = true });
        defer listener.deinit();

        while (self.running) {
            const connection = listener.accept() catch |err| {
                if (!self.running) break;
                std.log.err("WDBX HTTP accept failed: {any}", .{err});
                continue;
            };

            self.handleHttpConnection(connection) catch |err| {
                switch (err) {
                    error.OutOfMemory => return err,
                    else => {
                        std.log.warn("WDBX HTTP connection error: {any}", .{err});
                    },
                }
            };
        }

        self.running = false;
    }

    /// High level request handler used by the CLI and tests.
    pub fn respond(
        self: *WdbxHttpServer,
        method: []const u8,
        full_path: []const u8,
        body: []const u8,
    ) !Response {
        const parsed = splitPath(full_path);
        if (std.mem.eql(u8, method, "GET")) {
            return self.handleGet(parsed.path, parsed.query);
        } else if (std.mem.eql(u8, method, "POST")) {
            return self.handlePost(parsed.path, body);
        } else if (std.mem.eql(u8, method, "PUT")) {
            return self.handlePut(parsed.path, body);
        } else if (std.mem.eql(u8, method, "DELETE")) {
            return self.handleDelete(parsed.path);
        } else {
            return HttpError.UnsupportedMethod;
        }
    }

    fn handleHttpConnection(self: *WdbxHttpServer, connection: std.net.Server.Connection) !void {
        defer connection.stream.close();

        var buffer = ArrayList(u8).init(self.allocator);
        defer buffer.deinit();

        var header_end: ?usize = null;
        var temp: [1024]u8 = undefined;

        while (true) {
            const bytes_read = connection.stream.read(&temp) catch |err| {
                switch (err) {
                    error.ConnectionResetByPeer, error.BrokenPipe, error.Unexpected => return,
                    else => return err,
                }
            };

            if (bytes_read == 0) {
                return;
            }

            try buffer.appendSlice(temp[0..bytes_read]);

            if (header_end == null) {
                if (std.mem.indexOf(u8, buffer.items, "\r\n\r\n")) |idx| {
                    header_end = idx + 4;
                }
            }

            if (header_end != null) break;

            if (buffer.items.len > max_header_size) {
                try self.sendError(&connection, 431, "headers_too_large");
                return;
            }
        }

        const header_bytes = buffer.items[0..header_end.?];
        var lines = std.mem.splitScalar(u8, header_bytes, '\n');
        const request_line_raw = lines.next() orelse {
            try self.sendError(&connection, 400, "invalid_request_line");
            return;
        };
        const request_line = std.mem.trim(u8, request_line_raw, " \r");
        if (request_line.len == 0) {
            try self.sendError(&connection, 400, "invalid_request_line");
            return;
        }

        var tokens = std.mem.splitScalar(u8, request_line, ' ');
        const method = tokens.next() orelse {
            try self.sendError(&connection, 400, "invalid_method");
            return;
        };
        const target = tokens.next() orelse {
            try self.sendError(&connection, 400, "invalid_target");
            return;
        };

        var content_length: usize = 0;
        while (lines.next()) |line_raw| {
            const line = std.mem.trim(u8, line_raw, " \r");
            if (line.len == 0) continue;
            if (std.ascii.startsWithIgnoreCase(line, "Content-Length:")) {
                const value = std.mem.trim(u8, line["Content-Length:".len..], " ");
                content_length = std.fmt.parseInt(usize, value, 10) catch 0;
            }
        }

        const body_start = header_end.?;
        if (buffer.items.len < body_start + content_length) {
            const remaining = body_start + content_length - buffer.items.len;
            var total_read: usize = 0;
            while (total_read < remaining) {
                const bytes_read = connection.stream.read(&temp) catch |err| {
                    switch (err) {
                        error.ConnectionResetByPeer, error.BrokenPipe, error.Unexpected => return,
                        else => return err,
                    }
                };
                if (bytes_read == 0) break;
                try buffer.appendSlice(temp[0..bytes_read]);
                total_read += bytes_read;
            }
        }

        const available = buffer.items.len - body_start;
        const slice_len = if (content_length > available) available else content_length;
        const body_slice = buffer.items[body_start .. body_start + slice_len];

        var response = self.respond(method, target, body_slice) catch |err| switch (err) {
            HttpError.InvalidRequest => try self.buildErrorResponse(400, "invalid_request"),
            HttpError.UnsupportedMethod => try self.buildErrorResponse(405, "unsupported_method"),
            HttpError.NotFound => try self.buildErrorResponse(404, "not_found"),
            HttpError.BadPayload => try self.buildErrorResponse(400, "bad_payload"),
            else => blk: {
                std.log.err("WDBX HTTP handler failure: {any}", .{err});
                break :blk try self.buildErrorResponse(500, "internal_error");
            },
        };
        defer response.deinit(self.allocator);

        try writeHttpResponse(connection, response);
    }

    fn sendError(self: *WdbxHttpServer, connection: *const std.net.Server.Connection, status: u16, message: []const u8) !void {
        var response = try self.buildErrorResponse(status, message);
        defer response.deinit(self.allocator);
        try writeHttpResponse(connection, response);
    }

    fn handleGet(self: *WdbxHttpServer, path: []const u8, query: []const u8) !Response {
        if (std.mem.eql(u8, path, "/health")) {
            return self.buildJsonResponse(
                "{\"ok\":true,\"host\":\"{s}\",\"port\":{d}}",
                .{ self.config.host, self.config.port },
            );
        }
        if (std.mem.eql(u8, path, "/stats")) {
            const stats = self.vectorStats();
            return self.buildJsonResponse(
                "{\"vectors\":{d},\"database\":{s}}",
                .{ stats.vector_count, stats.database_path },
            );
        }
        if (std.mem.eql(u8, path, "/query")) {
            return self.handleQuery(query);
        }
        if (std.mem.eql(u8, path, "/api/status")) {
            return self.buildJsonResponse(
                "{\"status\":\"running\",\"version\":\"1.0.0\",\"features\":[\"vector_search\",\"ai_inference\"]}",
                .{},
            );
        }
        if (std.mem.eql(u8, path, "/api/database/info")) {
            const stats = self.vectorStats();
            return self.buildJsonResponse(
                "{\"vector_count\":{d},\"database_path\":\"{s}\",\"dimensions\":128}",
                .{ stats.vector_count, stats.database_path },
            );
        }
        if (std.mem.eql(u8, path, "/metrics")) {
            return self.buildJsonResponse(
                "{\"requests_total\":0,\"response_time_avg_ms\":15.5,\"memory_usage_mb\":45.2}",
                .{},
            );
        }
        return HttpError.NotFound;
    }

    fn handlePost(self: *WdbxHttpServer, path: []const u8, body: []const u8) !Response {
        if (std.mem.eql(u8, path, "/add")) {
            const entry_id = try self.addVectorFromJson(body);
            return self.buildJsonResponse(
                "{\"success\":true,\"id\":{d}}",
                .{entry_id},
            );
        }
        if (std.mem.eql(u8, path, "/api/agent/query")) {
            return self.handleAgentQuery(body);
        }
        if (std.mem.eql(u8, path, "/api/database/search")) {
            return self.handleDatabaseSearch(body);
        }
        return HttpError.NotFound;
    }

    fn handleQuery(self: *WdbxHttpServer, query: []const u8) !Response {
        var params = parseQueryParams(self.allocator, query);
        defer params.deinit();

        const vec_param = params.get("vec") orelse return HttpError.InvalidRequest;
        const query_vec = try parseVectorCsv(self.allocator, vec_param);
        defer self.allocator.free(query_vec);

        var k: usize = 1;
        if (params.get("k")) |k_str| {
            k = std.fmt.parseInt(usize, k_str, 10) catch 1;
            if (k == 0) k = 1;
        }

        const matches = try self.findNearest(query_vec, k);
        defer self.allocator.free(matches);

        var json = ArrayList(u8).init(self.allocator);
        errdefer json.deinit();
        try json.appendSlice("{\"matches\":[");
        for (matches, 0..) |match, idx| {
            if (idx != 0) try json.appendSlice(",");
            try std.fmt.format(json.writer(), "{{\"id\":{d},\"distance\":{d:.6}}}", .{ match.id, match.distance });
        }
        try json.appendSlice("]}");
        return Response{ .status = 200, .body = try json.toOwnedSlice(), .content_type = "application/json" };
    }

    fn handleAgentQuery(self: *WdbxHttpServer, body: []const u8) !Response {
        // Parse JSON request
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, body, .{});
        defer parsed.deinit();

        const root_object = parsed.value.object orelse return HttpError.InvalidRequest;
        const query = root_object.get("query") orelse return HttpError.InvalidRequest;
        const query_text = switch (query) {
            .string => query.string,
            else => return HttpError.BadPayload,
        };

        // Simulate AI agent response (placeholder)
        const response_text = try std.fmt.allocPrint(self.allocator, "{{\"response\":\"AI Agent response to: {s}\",\"confidence\":0.85,\"model\":\"placeholder\"}}", .{query_text});
        defer self.allocator.free(response_text);

        return Response{ .status = 200, .body = try self.allocator.dupe(u8, response_text), .content_type = "application/json" };
    }

    fn handleDatabaseSearch(self: *WdbxHttpServer, body: []const u8) !Response {
        // Parse JSON request
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, body, .{});
        defer parsed.deinit();

        const root_object = parsed.value.object orelse return HttpError.InvalidRequest;
        const vector_value = root_object.get("vector") orelse return HttpError.InvalidRequest;
        const vector_array = vector_value.array orelse return HttpError.InvalidRequest;

        // Parse vector
        var query_vec = ArrayList(f32).init(self.allocator);
        defer query_vec.deinit();

        for (vector_array.items) |item| {
            const value: f32 = switch (item) {
                .float => @floatCast(item.float),
                .integer => @floatFromInt(item.integer),
                else => return HttpError.BadPayload,
            };
            try query_vec.append(value);
        }

        const k = if (root_object.get("k")) |k_value| switch (k_value) {
            .integer => @as(usize, @intCast(k_value.integer)),
            else => 5,
        } else 5;

        // Perform search
        const matches = try self.findNearest(query_vec.items, k);
        defer self.allocator.free(matches);

        // Build JSON response
        var json = ArrayList(u8).init(self.allocator);
        errdefer json.deinit();

        try json.appendSlice("{\"results\":[");
        for (matches, 0..) |match, idx| {
            if (idx != 0) try json.appendSlice(",");
            try std.fmt.format(json.writer(), "{{\"id\":{d},\"score\":{d:.6}}}", .{ match.id, match.distance });
        }
        try json.appendSlice("],\"total\":");
        try std.fmt.format(json.writer(), "{d}", .{matches.len});
        try json.appendSlice("}");

        return Response{ .status = 200, .body = try json.toOwnedSlice(), .content_type = "application/json" };
    }

    fn handlePut(self: *WdbxHttpServer, path: []const u8, body: []const u8) !Response {
        // Handle PUT requests (update operations)
        if (std.mem.eql(u8, path, "/api/database/update")) {
            return self.handleDatabaseUpdate(body);
        }
        return HttpError.NotFound;
    }

    fn handleDelete(self: *WdbxHttpServer, path: []const u8) !Response {
        // Handle DELETE requests
        if (std.mem.eql(u8, path, "/api/database/clear")) {
            return self.handleDatabaseClear();
        }
        return HttpError.NotFound;
    }

    fn handleDatabaseUpdate(self: *WdbxHttpServer, body: []const u8) !Response {
        // Parse JSON request for vector update
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, body, .{});
        defer parsed.deinit();

        const root_object = parsed.value.object orelse return HttpError.InvalidRequest;
        const id_value = root_object.get("id") orelse return HttpError.InvalidRequest;
        const id = switch (id_value) {
            .integer => @as(u64, @intCast(id_value.integer)),
            else => return HttpError.BadPayload,
        };

        const vector_value = root_object.get("vector") orelse return HttpError.InvalidRequest;
        const vector_array = vector_value.array orelse return HttpError.InvalidRequest;

        // Parse new vector
        var new_vec = ArrayList(f32).init(self.allocator);
        defer new_vec.deinit();

        for (vector_array.items) |item| {
            const value: f32 = switch (item) {
                .float => @floatCast(item.float),
                .integer => @floatFromInt(item.integer),
                else => return HttpError.BadPayload,
            };
            try new_vec.append(value);
        }

        // Update vector in database
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.vectors.items) |*entry| {
            if (entry.id == id) {
                self.allocator.free(entry.data);
                entry.data = try new_vec.toOwnedSlice();
                return self.buildJsonResponse("{\"success\":true,\"updated_id\":{d}}", .{id});
            }
        }

        return HttpError.NotFound;
    }

    fn handleDatabaseClear(self: *WdbxHttpServer) !Response {
        self.mutex.lock();
        defer self.mutex.unlock();

        const cleared_count = self.vectors.items.len;
        for (self.vectors.items) |entry| {
            self.allocator.free(entry.data);
        }
        self.vectors.clearRetainingCapacity();
        self.next_id = 1;

        return self.buildJsonResponse("{\"success\":true,\"cleared_count\":{d}}", .{cleared_count});
    }

    fn addVectorFromJson(self: *WdbxHttpServer, payload: []const u8) !u64 {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, payload, .{});
        defer parsed.deinit();
        const root_object = parsed.value.object orelse return HttpError.InvalidRequest;

        const vector_value = root_object.get("vector") orelse return HttpError.InvalidRequest;
        const vector_array = vector_value.array orelse return HttpError.InvalidRequest;

        var values = ArrayList(f32).init(self.allocator);
        errdefer values.deinit();
        for (vector_array.items) |item| {
            const value: f32 = switch (item) {
                .float => @floatCast(item.float),
                .integer => @floatFromInt(item.integer),
                else => return HttpError.BadPayload,
            };
            try values.append(value);
        }
        const vector = try values.toOwnedSlice();

        var maybe_id: ?u64 = null;
        if (root_object.get("id")) |id_value| {
            maybe_id = switch (id_value) {
                .integer => @as(u64, @intCast(id_value.integer)),
                .float => @as(u64, @intFromFloat(id_value.float)),
                else => return HttpError.BadPayload,
            };
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        const entry_id = maybe_id orelse self.next_id;
        if (maybe_id == null) {
            self.next_id += 1;
        } else {
            self.next_id = @max(self.next_id, entry_id + 1);
        }

        try self.vectors.append(.{ .id = entry_id, .data = vector });
        return entry_id;
    }

    fn findNearest(self: *WdbxHttpServer, query_vec: []const f32, k: usize) ![]Match {
        self.mutex.lock();
        defer self.mutex.unlock();

        const count = self.vectors.items.len;
        var matches = try self.allocator.alloc(Match, count);
        var i: usize = 0;
        while (i < count) : (i += 1) {
            const entry = self.vectors.items[i];
            const distance = vectorDistance(entry.data, query_vec);
            matches[i] = .{ .id = entry.id, .distance = distance };
        }
        std.sort.sort(Match, matches, {}, struct {
            fn lessThan(_: void, a: Match, b: Match) bool {
                return a.distance < b.distance;
            }
        }.lessThan);
        const take = @min(k, matches.len);
        if (take == matches.len) return matches;
        const slice = matches[0..take];
        const owned = try self.allocator.dupe(Match, slice);
        self.allocator.free(matches);
        return owned;
    }

    fn vectorStats(self: *WdbxHttpServer) Stats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return .{
            .vector_count = self.vectors.items.len,
            .database_path = if (self.database_path) |path| path else "memory",
        };
    }

    fn buildJsonResponse(self: *WdbxHttpServer, comptime fmt: []const u8, args: anytype) !Response {
        const text = try std.fmt.allocPrint(self.allocator, fmt, args);
        return Response{ .status = 200, .body = text, .content_type = "application/json" };
    }

    fn buildErrorResponse(self: *WdbxHttpServer, status: u16, message: []const u8) !Response {
        const text = try std.fmt.allocPrint(self.allocator, "{\"error\":\"{s}\"}", .{message});
        return Response{ .status = status, .body = text, .content_type = "application/json" };
    }

    fn writeHttpResponse(_: *WdbxHttpServer, connection: *const std.net.Server.Connection, response: Response) !void {
        var writer = connection.stream.writer();
        try writer.print("HTTP/1.1 {d} {s}\r\n", .{ response.status, statusText(response.status) });
        try writer.print("Content-Type: {s}\r\n", .{response.content_type});
        try writer.print("Content-Length: {d}\r\n", .{response.body.len});
        try writer.writeAll("Connection: close\r\n\r\n");
        try writer.writeAll(response.body);
    }
};

fn statusText(status: u16) []const u8 {
    return switch (status) {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        else => "OK",
    };
}

const Stats = struct {
    vector_count: usize,
    database_path: []const u8,
};

const Match = struct {
    id: u64,
    distance: f32,
};

const QueryParams = struct {
    allocator: std.mem.Allocator,
    keys: ArrayList([]u8),
    values: ArrayList([]u8),

    fn init(allocator: std.mem.Allocator) QueryParams {
        return .{
            .allocator = allocator,
            .keys = ArrayList([]u8).init(allocator),
            .values = ArrayList([]u8).init(allocator),
        };
    }

    fn deinit(self: *QueryParams) void {
        for (self.keys.items) |item| self.allocator.free(item);
        for (self.values.items) |item| self.allocator.free(item);
        self.keys.deinit();
        self.values.deinit();
    }

    fn append(self: *QueryParams, key: []const u8, value: []const u8) !void {
        try self.keys.append(try self.allocator.dupe(u8, key));
        try self.values.append(try self.allocator.dupe(u8, value));
    }

    fn get(self: QueryParams, needle: []const u8) ?[]const u8 {
        for (self.keys.items, 0..) |key, idx| {
            if (std.mem.eql(u8, key, needle)) return self.values.items[idx];
        }
        return null;
    }
};

fn parseQueryParams(allocator: std.mem.Allocator, query: []const u8) QueryParams {
    var params = QueryParams.init(allocator);
    var iter = std.mem.splitScalar(u8, query, '&');
    while (iter.next()) |pair| {
        if (pair.len == 0) continue;
        const eq_index = std.mem.indexOfScalar(u8, pair, '=') orelse {
            _ = params.append(pair, "") catch {};
            continue;
        };
        const key = pair[0..eq_index];
        const value = pair[eq_index + 1 ..];
        _ = params.append(key, value) catch {};
    }
    return params;
}

fn parseVectorCsv(allocator: std.mem.Allocator, csv: []const u8) ![]f32 {
    var list = ArrayList(f32).init(allocator);
    errdefer list.deinit();
    var iter = std.mem.splitScalar(u8, csv, ',');
    while (iter.next()) |segment| {
        const trimmed = std.mem.trim(u8, segment, " \t\r\n");
        if (trimmed.len == 0) continue;
        const value = try std.fmt.parseFloat(f32, trimmed);
        try list.append(value);
    }
    return try list.toOwnedSlice();
}

/// Calculate Euclidean distance between two vectors
fn vectorDistance(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return std.math.inf(f32);
    var sum: f32 = 0.0;
    for (a, b) |x, y| {
        const diff = x - y;
        sum += diff * diff;
    }
    return @sqrt(sum);
}

const PathParts = struct { path: []const u8, query: []const u8 };

fn splitPath(full: []const u8) PathParts {
    if (full.len == 0) return .{ .path = "/", .query = "" };
    if (std.mem.indexOfScalar(u8, full, '?')) |idx| {
        return .{ .path = full[0..idx], .query = full[idx + 1 ..] };
    }
    return .{ .path = full, .query = "" };
}
