//! Persona API Routes
//!
//! Defines HTTP routes for the Multi-Persona AI Assistant API.
//! Can be integrated with any HTTP server framework.
//!
//! Routes:
//! - POST /api/v1/chat              - Auto-routing chat
//! - POST /api/v1/chat/abbey        - Abbey-specific chat
//! - POST /api/v1/chat/aviva        - Aviva-specific chat
//! - GET  /api/v1/personas          - List personas
//! - GET  /api/v1/personas/metrics  - Get metrics
//! - GET  /api/v1/personas/health   - Health check

const std = @import("std");
const chat = @import("../handlers/chat.zig");
const types = @import("../../ai/personas/types.zig");
const health = @import("../../ai/personas/health.zig");
const time = @import("../../shared/time.zig");

/// HTTP method.
pub const Method = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    OPTIONS,
    HEAD,
};

/// Route definition.
pub const Route = struct {
    path: []const u8,
    method: Method,
    handler: RouteHandler,
    description: []const u8,
    requires_auth: bool = false,
};

pub const RouteError = error{
    InvalidRequest,
    NotFound,
    Unauthorized,
    InternalError,
    JsonParseError,
    DatabaseError,
    OutOfMemory,
    WriteFailed,
};

/// Route handler function type.
pub const RouteHandler = *const fn (*RouteContext) RouteError!void;

/// Context passed to route handlers.
pub const RouteContext = struct {
    allocator: std.mem.Allocator,
    /// Request body.
    body: []const u8,
    /// Path parameters (e.g., persona name).
    path_params: std.StringHashMap([]const u8),
    /// Query parameters.
    query_params: std.StringHashMap([]const u8),
    /// Request headers.
    headers: std.StringHashMap([]const u8),
    /// Response body buffer.
    response_body: std.ArrayListUnmanaged(u8),
    /// Response status.
    response_status: u16 = 200,
    /// Response content type.
    response_content_type: []const u8 = "application/json",
    /// Reference to chat handler.
    chat_handler: *chat.ChatHandler,
    /// Reference to health checker (optional).
    health_checker: ?*health.HealthChecker = null,

    pub fn init(allocator: std.mem.Allocator, handler: *chat.ChatHandler) RouteContext {
        return .{
            .allocator = allocator,
            .body = &.{},
            .path_params = std.StringHashMap([]const u8).init(allocator),
            .query_params = std.StringHashMap([]const u8).init(allocator),
            .headers = std.StringHashMap([]const u8).init(allocator),
            .response_body = std.ArrayListUnmanaged(u8).empty,
            .chat_handler = handler,
        };
    }

    pub fn deinit(self: *RouteContext) void {
        self.path_params.deinit();
        self.query_params.deinit();
        self.headers.deinit();
        self.response_body.deinit(self.allocator);
    }

    /// Write response body.
    pub fn write(self: *RouteContext, data: []const u8) !void {
        try self.response_body.appendSlice(self.allocator, data);
    }

    /// Set response status.
    pub fn setStatus(self: *RouteContext, status: u16) void {
        self.response_status = status;
    }

    /// Set content type.
    pub fn setContentType(self: *RouteContext, content_type: []const u8) void {
        self.response_content_type = content_type;
    }

    /// Write JSON response.
    pub fn writeJson(self: *RouteContext, json: []const u8) !void {
        try self.write(json);
        self.setContentType("application/json");
    }

    /// Write error response.
    pub fn writeError(self: *RouteContext, status: u16, code: []const u8, message: []const u8) !void {
        self.setStatus(status);
        const error_json = try self.chat_handler.formatError(code, message, null);
        defer self.allocator.free(error_json);
        try self.writeJson(error_json);
    }
};

/// Handle POST /api/v1/chat - Auto-routing chat.
fn handlePersonaChat(ctx: *RouteContext, persona: []const u8) RouteError!void {
    _ = ctx;
    _ = persona;
    return RouteError.NotFound;
}

fn handleChat(ctx: *RouteContext) RouteError!void {
    return handlePersonaChat(ctx, "chat");
}

fn handleAbbeyChat(ctx: *RouteContext) RouteError!void {
    return handlePersonaChat(ctx, "abbey");
}

fn handleAvivaChat(ctx: *RouteContext) RouteError!void {
    return handlePersonaChat(ctx, "aviva");
}

fn handleListPersonas(ctx: *RouteContext) RouteError!void {
    const response = ctx.chat_handler.listPersonas() catch |err| {
        const err_name = @errorName(err);
        try ctx.writeError(500, "INTERNAL_ERROR", err_name);
        return;
    };
    try ctx.writeJson(response);
}

/// Handle GET /api/v1/personas/metrics - Get metrics.
fn handleGetMetrics(ctx: *RouteContext) RouteError!void {
    const response = ctx.chat_handler.getMetrics() catch |err| {
        const err_name = @errorName(err);
        try ctx.writeError(500, "INTERNAL_ERROR", err_name);
        return RouteError.InternalError;
    };
    try ctx.writeJson(response);
}

/// Handle GET /api/v1/personas/health - Health check.
fn handleHealthCheck(ctx: *RouteContext) RouteError!void {
    var response_obj = std.json.ObjectMap.init(ctx.allocator);
    defer response_obj.deinit();

    try response_obj.put("status", std.json.Value{ .string = "ok" });
    try response_obj.put("timestamp", std.json.Value{ .integer = time.unixSeconds() });

    // Add health details if checker is available
    if (ctx.health_checker) |checker| {
        const aggregate = checker.getAggregateHealth();
        try response_obj.put("healthy_personas", std.json.Value{ .integer = @intCast(aggregate.healthy_count) });
        try response_obj.put("degraded_personas", std.json.Value{ .integer = @intCast(aggregate.degraded_count) });
        try response_obj.put("unhealthy_personas", std.json.Value{ .integer = @intCast(aggregate.unhealthy_count) });

        const status_str = switch (aggregate.getOverallStatus()) {
            .healthy => "healthy",
            .degraded => "degraded",
            .unhealthy => "unhealthy",
            .unknown => "unknown",
        };
        try response_obj.put("overall_status", std.json.Value{ .string = status_str });
    }

    var output: std.Io.Writer.Allocating = .init(ctx.allocator);
    defer output.deinit();

    try std.json.Stringify.value(std.json.Value{ .object = response_obj }, .{}, &output.writer);
    try ctx.writeJson(try output.toOwnedSlice());
}

/// All persona API routes.
pub const ROUTES = [_]Route{
    .{
        .path = "/api/v1/chat",
        .method = .POST,
        .handler = handleChat,
        .description = "Send a chat message with auto-routing to best persona",
    },
    .{
        .path = "/api/v1/chat/abbey",
        .method = .POST,
        .handler = handleAbbeyChat,
        .description = "Send a chat message to Abbey (empathetic polymath)",
    },
    .{
        .path = "/api/v1/chat/aviva",
        .method = .POST,
        .handler = handleAvivaChat,
        .description = "Send a chat message to Aviva (direct expert)",
    },
    .{
        .path = "/api/v1/personas",
        .method = .GET,
        .handler = handleListPersonas,
        .description = "List available personas",
    },
    .{
        .path = "/api/v1/personas/metrics",
        .method = .GET,
        .handler = handleGetMetrics,
        .description = "Get persona metrics and statistics",
    },
    .{
        .path = "/api/v1/personas/health",
        .method = .GET,
        .handler = handleHealthCheck,
        .description = "Health check endpoint",
    },
};

/// Router for persona API.
pub const Router = struct {
    allocator: std.mem.Allocator,
    routes: []const Route,
    chat_handler: *chat.ChatHandler,
    health_checker: ?*health.HealthChecker = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, handler: *chat.ChatHandler) Self {
        return .{
            .allocator = allocator,
            .routes = &ROUTES,
            .chat_handler = handler,
        };
    }

    pub fn setHealthChecker(self: *Self, checker: *health.HealthChecker) void {
        self.health_checker = checker;
    }

    /// Match a route by path and method.
    pub fn match(self: *const Self, path: []const u8, method: Method) ?Route {
        for (self.routes) |route| {
            if (route.method == method and std.mem.eql(u8, route.path, path)) {
                return route;
            }
        }
        return null;
    }

    /// Handle a request.
    pub fn handle(self: *Self, path: []const u8, method: Method, body: []const u8) !RouteResult {
        const route = self.match(path, method) orelse {
            return RouteResult{
                .status = 404,
                .body = "{\"error\":{\"code\":\"NOT_FOUND\",\"message\":\"Route not found\"}}",
                .content_type = "application/json",
            };
        };

        var ctx = RouteContext.init(self.allocator, self.chat_handler);
        defer ctx.deinit();

        ctx.body = body;
        ctx.health_checker = self.health_checker;

        route.handler(&ctx) catch |err| {
            return RouteResult{
                .status = 500,
                .body = try std.fmt.allocPrint(self.allocator, "{{\"error\":{{\"code\":\"INTERNAL_ERROR\",\"message\":\"{t}\"}}}}", .{err}),
                .content_type = "application/json",
            };
        };

        return RouteResult{
            .status = ctx.response_status,
            .body = try ctx.response_body.toOwnedSlice(self.allocator),
            .content_type = ctx.response_content_type,
        };
    }

    /// Get all route definitions for documentation.
    pub fn getRouteDefinitions(self: *const Self) []const Route {
        return self.routes;
    }
};

/// Result of routing a request.
pub const RouteResult = struct {
    status: u16,
    body: []const u8,
    content_type: []const u8,
};

/// Generate OpenAPI documentation for routes.
pub fn generateOpenApiSpec(allocator: std.mem.Allocator) ![]const u8 {
    var spec: std.Io.Writer.Allocating = .init(allocator);
    errdefer spec.deinit();

    try spec.writer.writeAll(
        \\{
        \\  "openapi": "3.0.0",
        \\  "info": {
        \\    "title": "Multi-Persona AI Assistant API",
        \\    "version": "1.0.0",
        \\    "description": "API for interacting with the Multi-Persona AI Assistant"
        \\  },
        \\  "paths": {
        \\
    );

    for (ROUTES, 0..) |route, i| {
        if (i > 0) try spec.writer.writeAll(",\n");

        const method_str = switch (route.method) {
            .GET => "get",
            .POST => "post",
            .PUT => "put",
            .DELETE => "delete",
            .PATCH => "patch",
            .OPTIONS => "options",
            .HEAD => "head",
        };

        try spec.writer.print(
            \\    "{s}": {{
            \\      "{s}": {{
            \\        "summary": "{s}",
            \\        "responses": {{
            \\          "200": {{
            \\            "description": "Successful response"
            \\          }}
            \\        }}
            \\      }}
            \\    }}
        , .{ route.path, method_str, route.description });
    }

    try spec.writer.writeAll(
        \\
        \\  }
        \\}
    );

    return spec.toOwnedSlice();
}

// Tests

test "route matching" {
    var handler = chat.ChatHandler.init(std.testing.allocator);
    const router = Router.init(std.testing.allocator, &handler);

    const route = router.match("/api/v1/chat", .POST);
    try std.testing.expect(route != null);
    try std.testing.expectEqualStrings("/api/v1/chat", route.?.path);

    const not_found = router.match("/api/v1/nonexistent", .GET);
    try std.testing.expect(not_found == null);
}

test "route context initialization" {
    var handler = chat.ChatHandler.init(std.testing.allocator);
    var ctx = RouteContext.init(std.testing.allocator, &handler);
    defer ctx.deinit();

    try std.testing.expectEqual(@as(u16, 200), ctx.response_status);
}

test "route definitions" {
    try std.testing.expect(ROUTES.len >= 5);

    // Check that chat route exists
    var found_chat = false;
    for (ROUTES) |route| {
        if (std.mem.eql(u8, route.path, "/api/v1/chat")) {
            found_chat = true;
            try std.testing.expectEqual(Method.POST, route.method);
        }
    }
    try std.testing.expect(found_chat);
}
