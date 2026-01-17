//! Abbey HTTP API Server
//!
//! REST API for Abbey AI interactions:
//! - POST /chat - Send a message and get a response
//! - POST /session/start - Start a new conversation
//! - POST /session/end - End current conversation
//! - POST /feedback - Provide feedback on last response
//! - GET /health - Health check
//! - GET /stats - Get engine statistics
//! - GET /emotional-state - Get current emotional state

const std = @import("std");
const engine = @import("engine.zig");
const core_types = @import("core/types.zig");
const core_config = @import("core/config.zig");
const net_utils = @import("../../../shared/utils/net/mod.zig");

// ============================================================================
// Server Types
// ============================================================================

pub const ServerError = std.mem.Allocator.Error || error{
    InvalidAddress,
    InvalidRequest,
    ReadFailed,
    RequestTooLarge,
    Unauthorized,
    EngineNotInitialized,
    SessionNotActive,
};

const max_body_bytes = 1024 * 1024; // 1MB max request body

/// Server configuration
pub const ServerConfig = struct {
    /// Bearer token for authentication (null = no auth required)
    auth_token: ?[]const u8 = null,
    /// Allow unauthenticated access to health endpoint
    allow_health_without_auth: bool = true,
    /// Allow unauthenticated access to stats endpoint
    allow_stats_without_auth: bool = false,
    /// CORS origin (null = no CORS headers)
    cors_origin: ?[]const u8 = null,
};

/// Combined configuration for Abbey server
pub const AbbeyServerConfig = struct {
    /// Abbey engine configuration
    abbey: core_config.AbbeyConfig = .{},
    /// Server configuration
    server: ServerConfig = .{},
    /// Listen address (e.g., "127.0.0.1:8080")
    address: []const u8 = "127.0.0.1:8080",
};

// ============================================================================
// Server Implementation
// ============================================================================

/// Start Abbey HTTP server with default configuration
pub fn serve(allocator: std.mem.Allocator, address: []const u8) !void {
    try serveWithConfig(allocator, .{ .address = address });
}

/// Start Abbey HTTP server with custom configuration
pub fn serveWithConfig(allocator: std.mem.Allocator, config: AbbeyServerConfig) !void {
    // Initialize Abbey engine
    var abbey_engine = try engine.AbbeyEngine.init(allocator, config.abbey);
    defer abbey_engine.deinit();

    // Start server
    try runServer(allocator, &abbey_engine, config.address, config.server);
}

/// Run the HTTP server (internal)
fn runServer(
    allocator: std.mem.Allocator,
    abbey_engine: *engine.AbbeyEngine,
    address: []const u8,
    config: ServerConfig,
) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    const listen_addr = try resolveAddress(io, allocator, address);
    var server = try listen_addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    std.debug.print("Abbey HTTP server listening on {s}\n", .{address});

    while (true) {
        var stream = server.accept(io) catch |err| {
            std.debug.print("Abbey HTTP accept error: {t}\n", .{err});
            continue;
        };
        defer stream.close(io);
        handleConnection(allocator, io, abbey_engine, stream, config) catch |err| {
            std.debug.print("Abbey HTTP connection error: {t}\n", .{err});
        };
    }
}

fn resolveAddress(
    io: std.Io,
    allocator: std.mem.Allocator,
    address: []const u8,
) !std.Io.net.IpAddress {
    var host_port = net_utils.parseHostPort(allocator, address) catch
        return ServerError.InvalidAddress;
    defer host_port.deinit(allocator);
    return std.Io.net.IpAddress.resolve(io, host_port.host, host_port.port) catch
        return ServerError.InvalidAddress;
}

fn handleConnection(
    allocator: std.mem.Allocator,
    io: std.Io,
    abbey_engine: *engine.AbbeyEngine,
    stream: std.Io.net.Stream,
    config: ServerConfig,
) !void {
    var send_buffer: [4096]u8 = undefined;
    var recv_buffer: [4096]u8 = undefined;
    var connection_reader = stream.reader(io, &recv_buffer);
    var connection_writer = stream.writer(io, &send_buffer);
    var server: std.http.Server = .init(
        &connection_reader,
        &connection_writer,
    );

    while (true) {
        var request = server.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return err,
        };
        dispatchRequest(allocator, abbey_engine, &request, config) catch |err| {
            std.debug.print("Abbey HTTP request error: {t}\n", .{err});
            const error_body = switch (err) {
                ServerError.Unauthorized => "{\"error\":\"unauthorized\"}",
                ServerError.SessionNotActive => "{\"error\":\"no active session\"}",
                ServerError.InvalidRequest => "{\"error\":\"invalid request\"}",
                else => "{\"error\":\"internal server error\"}",
            };
            const status: std.http.Status = switch (err) {
                ServerError.Unauthorized => .unauthorized,
                ServerError.SessionNotActive => .bad_request,
                ServerError.InvalidRequest => .bad_request,
                else => .internal_server_error,
            };
            respondJson(&request, error_body, status, config) catch |respond_err| {
                std.log.err("Failed to send error response: {t} (original: {t})", .{
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
    abbey_engine: *engine.AbbeyEngine,
    request: *std.http.Server.Request,
    config: ServerConfig,
) !void {
    const target = request.head.target;
    const parts = splitTarget(target);

    // Handle CORS preflight
    if (request.head.method == .OPTIONS) {
        return respondCorsPrelight(request, config);
    }

    // Health endpoint
    if (std.mem.eql(u8, parts.path, "/health")) {
        if (!config.allow_health_without_auth) {
            try validateAuth(request, config);
        }
        return respondJson(request, "{\"status\":\"ok\",\"service\":\"abbey\"}", .ok, config);
    }

    // Stats endpoint
    if (std.mem.eql(u8, parts.path, "/stats")) {
        if (!config.allow_stats_without_auth) {
            try validateAuth(request, config);
        }
        return handleStats(allocator, abbey_engine, request, config);
    }

    // All other endpoints require authentication if configured
    try validateAuth(request, config);

    // Chat endpoint
    if (std.mem.eql(u8, parts.path, "/chat")) {
        return handleChat(allocator, abbey_engine, request, config);
    }

    // Session management
    if (std.mem.eql(u8, parts.path, "/session/start")) {
        return handleSessionStart(allocator, abbey_engine, request, parts.query, config);
    }

    if (std.mem.eql(u8, parts.path, "/session/end")) {
        return handleSessionEnd(abbey_engine, request, config);
    }

    // Feedback
    if (std.mem.eql(u8, parts.path, "/feedback")) {
        return handleFeedback(allocator, abbey_engine, request, config);
    }

    // Emotional state
    if (std.mem.eql(u8, parts.path, "/emotional-state")) {
        return handleEmotionalState(allocator, abbey_engine, request, config);
    }

    // Knowledge learning
    if (std.mem.eql(u8, parts.path, "/learn")) {
        return handleLearn(allocator, abbey_engine, request, config);
    }

    return respondJson(request, "{\"error\":\"not found\"}", .not_found, config);
}

// ============================================================================
// Request Handlers
// ============================================================================

fn handleChat(
    allocator: std.mem.Allocator,
    abbey_engine: *engine.AbbeyEngine,
    request: *std.http.Server.Request,
    config: ServerConfig,
) !void {
    if (request.head.method != .POST) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed, config);
    }

    // Read request body
    const body = try readRequestBody(allocator, request);
    defer allocator.free(body);

    // Parse JSON to extract message
    const message = extractJsonField(body, "message") orelse {
        return respondJson(request, "{\"error\":\"missing message field\"}", .bad_request, config);
    };

    // Process with Abbey
    var response = try abbey_engine.process(message);
    defer response.deinit(allocator);

    // Build response JSON
    const response_json = try buildChatResponse(allocator, &response);
    defer allocator.free(response_json);

    return respondJson(request, response_json, .ok, config);
}

fn handleStats(
    allocator: std.mem.Allocator,
    abbey_engine: *engine.AbbeyEngine,
    request: *std.http.Server.Request,
    config: ServerConfig,
) !void {
    if (request.head.method != .GET) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed, config);
    }

    const stats = abbey_engine.getStats();
    const stats_json = try buildStatsJson(allocator, stats);
    defer allocator.free(stats_json);

    return respondJson(request, stats_json, .ok, config);
}

fn handleSessionStart(
    allocator: std.mem.Allocator,
    abbey_engine: *engine.AbbeyEngine,
    request: *std.http.Server.Request,
    query: []const u8,
    config: ServerConfig,
) !void {
    if (request.head.method != .POST) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed, config);
    }

    const user_id = getQueryParam(query, "user_id");
    try abbey_engine.startConversation(user_id);

    // Build session info response
    var response = std.ArrayListUnmanaged(u8){};
    defer response.deinit(allocator);

    try response.appendSlice(allocator, "{\"status\":\"session_started\"");
    if (abbey_engine.session_id) |sid| {
        try response.appendSlice(allocator, ",\"session_id\":\"");
        try response.print(allocator, "{d}", .{sid.id});
        try response.appendSlice(allocator, "\"");
    }
    try response.appendSlice(allocator, "}");

    return respondJson(request, response.items, .ok, config);
}

fn handleSessionEnd(
    abbey_engine: *engine.AbbeyEngine,
    request: *std.http.Server.Request,
    config: ServerConfig,
) !void {
    if (request.head.method != .POST) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed, config);
    }

    try abbey_engine.endConversation();
    return respondJson(request, "{\"status\":\"session_ended\"}", .ok, config);
}

fn handleFeedback(
    allocator: std.mem.Allocator,
    abbey_engine: *engine.AbbeyEngine,
    request: *std.http.Server.Request,
    config: ServerConfig,
) !void {
    if (request.head.method != .POST) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed, config);
    }

    const body = try readRequestBody(allocator, request);
    defer allocator.free(body);

    // Parse positive/negative feedback
    const positive = if (extractJsonField(body, "positive")) |val|
        std.mem.eql(u8, val, "true")
    else
        true; // Default to positive

    try abbey_engine.provideFeedback(positive);
    return respondJson(request, "{\"status\":\"feedback_recorded\"}", .ok, config);
}

fn handleEmotionalState(
    allocator: std.mem.Allocator,
    abbey_engine: *engine.AbbeyEngine,
    request: *std.http.Server.Request,
    config: ServerConfig,
) !void {
    if (request.head.method != .GET) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed, config);
    }

    const emotional = abbey_engine.getEmotionalState();
    const emotional_json = try buildEmotionalStateJson(allocator, emotional);
    defer allocator.free(emotional_json);

    return respondJson(request, emotional_json, .ok, config);
}

fn handleLearn(
    allocator: std.mem.Allocator,
    abbey_engine: *engine.AbbeyEngine,
    request: *std.http.Server.Request,
    config: ServerConfig,
) !void {
    if (request.head.method != .POST) {
        return respondJson(request, "{\"error\":\"method not allowed\"}", .method_not_allowed, config);
    }

    const body = try readRequestBody(allocator, request);
    defer allocator.free(body);

    const content = extractJsonField(body, "content") orelse {
        return respondJson(request, "{\"error\":\"missing content field\"}", .bad_request, config);
    };

    // Default to fact category
    const kid = try abbey_engine.learnFromUser(content, .fact);

    var response = std.ArrayListUnmanaged(u8){};
    defer response.deinit(allocator);
    try response.appendSlice(allocator, "{\"status\":\"learned\",\"knowledge_id\":");
    try response.print(allocator, "{d}", .{kid});
    try response.appendSlice(allocator, "}");

    return respondJson(request, response.items, .ok, config);
}

// ============================================================================
// Response Builders
// ============================================================================

fn buildChatResponse(allocator: std.mem.Allocator, response: *const engine.Response) ![]u8 {
    var json = std.ArrayListUnmanaged(u8){};
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\"content\":");
    try appendJsonString(&json, allocator, response.content);

    try json.appendSlice(allocator, ",\"confidence\":{\"level\":\"");
    try json.print(allocator, "{t}", .{response.confidence.level});
    try json.appendSlice(allocator, "\",\"score\":");
    try json.print(allocator, "{d:.4}", .{response.confidence.score});
    try json.appendSlice(allocator, "}");

    try json.appendSlice(allocator, ",\"emotion\":\"");
    try json.print(allocator, "{t}", .{response.emotional_context.detected});
    try json.appendSlice(allocator, "\"");

    try json.appendSlice(allocator, ",\"research_performed\":");
    try json.appendSlice(allocator, if (response.research_performed) "true" else "false");

    try json.appendSlice(allocator, ",\"generation_time_ms\":");
    try json.print(allocator, "{d}", .{response.generation_time_ms});

    if (response.reasoning_summary) |summary| {
        try json.appendSlice(allocator, ",\"reasoning\":");
        try appendJsonString(&json, allocator, summary);
    }

    try json.appendSlice(allocator, "}");

    return json.toOwnedSlice(allocator);
}

fn buildStatsJson(allocator: std.mem.Allocator, stats: engine.EngineStats) ![]u8 {
    var json = std.ArrayListUnmanaged(u8){};
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{");
    try json.appendSlice(allocator, "\"turn_count\":");
    try json.print(allocator, "{d}", .{stats.turn_count});
    try json.appendSlice(allocator, ",\"total_queries\":");
    try json.print(allocator, "{d}", .{stats.total_queries});
    try json.appendSlice(allocator, ",\"total_tokens_used\":");
    try json.print(allocator, "{d}", .{stats.total_tokens_used});
    try json.appendSlice(allocator, ",\"avg_response_time_ms\":");
    try json.print(allocator, "{d:.2}", .{stats.avg_response_time_ms});
    try json.appendSlice(allocator, ",\"relationship_score\":");
    try json.print(allocator, "{d:.4}", .{stats.relationship_score});
    try json.appendSlice(allocator, ",\"current_emotion\":\"");
    try json.print(allocator, "{t}", .{stats.current_emotion});
    try json.appendSlice(allocator, "\",\"topics_discussed\":");
    try json.print(allocator, "{d}", .{stats.topics_discussed});
    try json.appendSlice(allocator, ",\"conversation_active\":");
    try json.appendSlice(allocator, if (stats.conversation_active) "true" else "false");
    try json.appendSlice(allocator, ",\"llm_backend\":\"");
    try json.appendSlice(allocator, stats.llm_backend);
    try json.appendSlice(allocator, "\"}");

    return json.toOwnedSlice(allocator);
}

fn buildEmotionalStateJson(allocator: std.mem.Allocator, emotional: @import("emotions.zig").EmotionalState) ![]u8 {
    var json = std.ArrayListUnmanaged(u8){};
    errdefer json.deinit(allocator);

    try json.appendSlice(allocator, "{\"detected\":\"");
    try json.print(allocator, "{t}", .{emotional.detected});
    try json.appendSlice(allocator, "\",\"confidence\":");
    try json.print(allocator, "{d:.4}", .{emotional.confidence});
    try json.appendSlice(allocator, ",\"valence\":");
    try json.print(allocator, "{d:.4}", .{emotional.valence});
    try json.appendSlice(allocator, ",\"arousal\":");
    try json.print(allocator, "{d:.4}", .{emotional.arousal});
    try json.appendSlice(allocator, "}");

    return json.toOwnedSlice(allocator);
}

// ============================================================================
// Utilities
// ============================================================================

fn readRequestBody(
    allocator: std.mem.Allocator,
    request: *std.http.Server.Request,
) ServerError![]u8 {
    var buffer: [4096]u8 = undefined;
    const reader = request.readerExpectContinue(&buffer) catch
        return ServerError.ReadFailed;
    return readAll(reader, allocator, max_body_bytes);
}

fn readAll(
    reader: *std.Io.Reader,
    allocator: std.mem.Allocator,
    limit: usize,
) ServerError![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);

    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = reader.readSliceShort(chunk[0..]) catch
            return ServerError.ReadFailed;
        if (n == 0) break;
        if (list.items.len + n > limit) return ServerError.RequestTooLarge;
        try list.appendSlice(allocator, chunk[0..n]);
        if (n < chunk.len) break;
    }
    return list.toOwnedSlice(allocator);
}

fn validateAuth(request: *std.http.Server.Request, config: ServerConfig) ServerError!void {
    const expected_token = config.auth_token orelse return;

    const auth_header = getAuthorizationHeader(request) orelse {
        return ServerError.Unauthorized;
    };

    const bearer_prefix = "Bearer ";
    if (!std.mem.startsWith(u8, auth_header, bearer_prefix)) {
        return ServerError.Unauthorized;
    }

    const provided_token = auth_header[bearer_prefix.len..];

    if (provided_token.len != expected_token.len or
        !timingSafeEqual(provided_token, expected_token))
    {
        return ServerError.Unauthorized;
    }
}

fn timingSafeEqual(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, b) |x, y| {
        diff |= x ^ y;
    }
    return diff == 0;
}

fn getAuthorizationHeader(request: *std.http.Server.Request) ?[]const u8 {
    return findHeaderInBuffer(request.head_buffer, "authorization");
}

fn findHeaderInBuffer(buffer: []const u8, header_name: []const u8) ?[]const u8 {
    var it = std.mem.splitSequence(u8, buffer, "\r\n");
    _ = it.next(); // Skip request line

    while (it.next()) |line| {
        if (line.len == 0) break;

        const colon_idx = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = line[0..colon_idx];

        if (std.ascii.eqlIgnoreCase(name, header_name)) {
            const value_start = colon_idx + 1;
            if (value_start >= line.len) return "";
            return std.mem.trim(u8, line[value_start..], " \t");
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

/// Simple JSON field extraction (no full parser needed for simple cases)
fn extractJsonField(json: []const u8, field: []const u8) ?[]const u8 {
    // Look for "field":"value" or "field": "value"
    var search_buf: [256]u8 = undefined;
    const search = std.fmt.bufPrint(&search_buf, "\"{s}\"", .{field}) catch return null;

    const field_start = std.mem.indexOf(u8, json, search) orelse return null;
    const after_field = json[field_start + search.len ..];

    // Skip : and whitespace
    var i: usize = 0;
    while (i < after_field.len and (after_field[i] == ':' or after_field[i] == ' ')) : (i += 1) {}

    if (i >= after_field.len) return null;

    // Check if value is a string (starts with ")
    if (after_field[i] == '"') {
        const value_start = i + 1;
        var value_end = value_start;
        while (value_end < after_field.len) : (value_end += 1) {
            if (after_field[value_end] == '"' and (value_end == value_start or after_field[value_end - 1] != '\\')) {
                return after_field[value_start..value_end];
            }
        }
    }

    // Check for non-string values (true/false/numbers)
    const value_start = i;
    var value_end = i;
    while (value_end < after_field.len) : (value_end += 1) {
        const c = after_field[value_end];
        if (c == ',' or c == '}' or c == ']' or c == ' ' or c == '\n') {
            break;
        }
    }
    if (value_end > value_start) {
        return after_field[value_start..value_end];
    }

    return null;
}

fn appendJsonString(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try list.append(allocator, '"');
    for (s) |c| {
        switch (c) {
            '"' => try list.appendSlice(allocator, "\\\""),
            '\\' => try list.appendSlice(allocator, "\\\\"),
            '\n' => try list.appendSlice(allocator, "\\n"),
            '\r' => try list.appendSlice(allocator, "\\r"),
            '\t' => try list.appendSlice(allocator, "\\t"),
            else => {
                if (c < 0x20) {
                    try list.print(allocator, "\\u{x:0>4}", .{c});
                } else {
                    try list.append(allocator, c);
                }
            },
        }
    }
    try list.append(allocator, '"');
}

fn respondJson(
    request: *std.http.Server.Request,
    body: []const u8,
    status: std.http.Status,
    config: ServerConfig,
) !void {
    if (config.cors_origin) |origin| {
        const headers = [_]std.http.Header{
            .{ .name = "content-type", .value = "application/json" },
            .{ .name = "access-control-allow-origin", .value = origin },
        };
        try request.respond(body, .{
            .status = status,
            .extra_headers = &headers,
        });
    } else {
        const headers = [_]std.http.Header{
            .{ .name = "content-type", .value = "application/json" },
        };
        try request.respond(body, .{
            .status = status,
            .extra_headers = &headers,
        });
    }
}

fn respondCorsPrelight(request: *std.http.Server.Request, config: ServerConfig) !void {
    const origin = config.cors_origin orelse "*";
    const headers = [_]std.http.Header{
        .{ .name = "access-control-allow-origin", .value = origin },
        .{ .name = "access-control-allow-methods", .value = "GET, POST, OPTIONS" },
        .{ .name = "access-control-allow-headers", .value = "Content-Type, Authorization" },
        .{ .name = "access-control-max-age", .value = "86400" },
    };
    try request.respond("", .{
        .status = .no_content,
        .extra_headers = &headers,
    });
}

// ============================================================================
// Tests
// ============================================================================

test "extract json field" {
    const json = "{\"message\":\"hello world\",\"count\":42}";

    const message = extractJsonField(json, "message");
    try std.testing.expectEqualStrings("hello world", message.?);

    const count = extractJsonField(json, "count");
    try std.testing.expectEqualStrings("42", count.?);

    const missing = extractJsonField(json, "nonexistent");
    try std.testing.expect(missing == null);
}

test "split target" {
    const parts = splitTarget("/chat?user=test&mode=debug");
    try std.testing.expectEqualStrings("/chat", parts.path);
    try std.testing.expectEqualStrings("user=test&mode=debug", parts.query);
}

test "get query param" {
    const query = "user=test&mode=debug&count=5";

    const user = getQueryParam(query, "user");
    try std.testing.expectEqualStrings("test", user.?);

    const mode = getQueryParam(query, "mode");
    try std.testing.expectEqualStrings("debug", mode.?);

    const missing = getQueryParam(query, "nonexistent");
    try std.testing.expect(missing == null);
}
