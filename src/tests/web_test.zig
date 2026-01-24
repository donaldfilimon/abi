//! Web Module Tests
//!
//! Comprehensive tests for the web module including:
//! - HTTP client functionality
//! - Chat handlers and request/response types
//! - Persona routes and router
//! - Context management
//! - OpenAPI spec generation

const std = @import("std");
const build_options = @import("build_options");
const web = @import("abi").web;

// =============================================================================
// Module Initialization Tests
// =============================================================================

test "web module enabled check" {
    // Simply verify the function exists and returns a boolean
    const enabled = web.isEnabled();
    _ = enabled;
}

test "web module init and deinit" {
    if (!web.isEnabled()) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    try web.init(allocator);
    try std.testing.expect(web.isInitialized());

    web.deinit();
    try std.testing.expect(!web.isInitialized());
}

test "web module double init is safe" {
    if (!web.isEnabled()) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    try web.init(allocator);
    try web.init(allocator); // Should not fail
    try std.testing.expect(web.isInitialized());

    web.deinit();
}

// =============================================================================
// HTTP Status Helper Tests
// =============================================================================

test "success status check" {
    try std.testing.expect(web.isSuccessStatus(200));
    try std.testing.expect(web.isSuccessStatus(201));
    try std.testing.expect(web.isSuccessStatus(204));
    try std.testing.expect(web.isSuccessStatus(299));

    try std.testing.expect(!web.isSuccessStatus(400));
    try std.testing.expect(!web.isSuccessStatus(404));
    try std.testing.expect(!web.isSuccessStatus(500));
    try std.testing.expect(!web.isSuccessStatus(100));
}

// =============================================================================
// Response Parsing Tests
// =============================================================================

test "parse json response" {
    const allocator = std.testing.allocator;

    const response = web.Response{
        .status = 200,
        .body = "{\"name\":\"test\",\"value\":42}",
    };

    var parsed = try web.parseJsonValue(allocator, response);
    defer parsed.deinit();

    try std.testing.expectEqualStrings("test", parsed.value.object.get("name").?.string);
    try std.testing.expectEqual(@as(i64, 42), parsed.value.object.get("value").?.integer);
}

test "parse json array response" {
    const allocator = std.testing.allocator;

    const response = web.Response{
        .status = 200,
        .body = "[1, 2, 3, 4, 5]",
    };

    var parsed = try web.parseJsonValue(allocator, response);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 5), parsed.value.array.items.len);
    try std.testing.expectEqual(@as(i64, 1), parsed.value.array.items[0].integer);
}

test "parse nested json response" {
    const allocator = std.testing.allocator;

    const response = web.Response{
        .status = 200,
        .body =
        \\{
        \\  "user": {
        \\    "id": 123,
        \\    "name": "Alice"
        \\  },
        \\  "active": true
        \\}
        ,
    };

    var parsed = try web.parseJsonValue(allocator, response);
    defer parsed.deinit();

    const user = parsed.value.object.get("user").?.object;
    try std.testing.expectEqual(@as(i64, 123), user.get("id").?.integer);
    try std.testing.expectEqualStrings("Alice", user.get("name").?.string);
    try std.testing.expect(parsed.value.object.get("active").?.bool);
}

// =============================================================================
// Chat Handler Tests
// =============================================================================

test "chat handler initialization" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    try std.testing.expect(handler.orchestrator == null);
}

test "chat request struct fields" {
    const request = web.ChatRequest{
        .content = "Hello, world!",
        .user_id = "user-123",
        .session_id = "session-456",
        .persona = "abbey",
        .context = "Be helpful",
        .max_tokens = 100,
        .temperature = 0.7,
    };

    try std.testing.expectEqualStrings("Hello, world!", request.content);
    try std.testing.expectEqualStrings("user-123", request.user_id.?);
    try std.testing.expectEqualStrings("abbey", request.persona.?);
    try std.testing.expectEqual(@as(u32, 100), request.max_tokens.?);
    try std.testing.expectEqual(@as(f32, 0.7), request.temperature.?);
}

test "chat response struct fields" {
    const response = web.ChatResponse{
        .content = "Hello! How can I help?",
        .persona = "abbey",
        .confidence = 0.95,
        .latency_ms = 150,
        .request_id = "req-789",
    };

    try std.testing.expectEqualStrings("Hello! How can I help?", response.content);
    try std.testing.expectEqualStrings("abbey", response.persona);
    try std.testing.expectEqual(@as(f32, 0.95), response.confidence);
    try std.testing.expectEqual(@as(u64, 150), response.latency_ms);
}

test "chat handler error format" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    const error_json = try handler.formatError("TEST_ERROR", "Test error message", "req-001");
    defer allocator.free(error_json);

    try std.testing.expect(std.mem.indexOf(u8, error_json, "TEST_ERROR") != null);
    try std.testing.expect(std.mem.indexOf(u8, error_json, "Test error message") != null);
    try std.testing.expect(std.mem.indexOf(u8, error_json, "req-001") != null);
}

test "chat handler list personas without orchestrator" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    const personas_json = try handler.listPersonas();
    defer allocator.free(personas_json);

    // Should return valid JSON even without orchestrator
    try std.testing.expect(std.mem.indexOf(u8, personas_json, "personas") != null);
}

test "chat handler get metrics without orchestrator" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    const metrics_json = try handler.getMetrics();
    defer allocator.free(metrics_json);

    // Should return valid JSON even without orchestrator
    try std.testing.expect(std.mem.indexOf(u8, metrics_json, "total_requests") != null);
    try std.testing.expect(std.mem.indexOf(u8, metrics_json, "overall_success_rate") != null);
}

// =============================================================================
// Persona Route Tests
// =============================================================================

test "persona router initialization" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    const router = web.PersonaRouter.init(allocator, &handler);

    try std.testing.expect(router.routes.len >= 5);
}

test "persona router route matching" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    const router = web.PersonaRouter.init(allocator, &handler);

    // Should find chat route
    const chat_route = router.match("/api/v1/chat", .POST);
    try std.testing.expect(chat_route != null);
    try std.testing.expectEqualStrings("/api/v1/chat", chat_route.?.path);

    // Should find personas route
    const personas_route = router.match("/api/v1/personas", .GET);
    try std.testing.expect(personas_route != null);

    // Should find health route
    const health_route = router.match("/api/v1/personas/health", .GET);
    try std.testing.expect(health_route != null);

    // Should not find non-existent route
    const not_found = router.match("/api/v1/nonexistent", .GET);
    try std.testing.expect(not_found == null);
}

test "persona router wrong method" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    const router = web.PersonaRouter.init(allocator, &handler);

    // Chat route is POST, should not match GET
    const not_found = router.match("/api/v1/chat", .GET);
    try std.testing.expect(not_found == null);
}

test "route context initialization" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    var ctx = web.RouteContext.init(allocator, &handler);
    defer ctx.deinit();

    try std.testing.expectEqual(@as(u16, 200), ctx.response_status);
    try std.testing.expectEqualStrings("application/json", ctx.response_content_type);
}

test "route context write operations" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    var ctx = web.RouteContext.init(allocator, &handler);
    defer ctx.deinit();

    try ctx.write("Hello, ");
    try ctx.write("World!");

    try std.testing.expectEqualStrings("Hello, World!", ctx.response_body.items);
}

test "route context status modification" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    var ctx = web.RouteContext.init(allocator, &handler);
    defer ctx.deinit();

    ctx.setStatus(404);
    try std.testing.expectEqual(@as(u16, 404), ctx.response_status);

    ctx.setStatus(500);
    try std.testing.expectEqual(@as(u16, 500), ctx.response_status);
}

test "route context content type modification" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    var ctx = web.RouteContext.init(allocator, &handler);
    defer ctx.deinit();

    ctx.setContentType("text/plain");
    try std.testing.expectEqualStrings("text/plain", ctx.response_content_type);

    ctx.setContentType("text/html");
    try std.testing.expectEqualStrings("text/html", ctx.response_content_type);
}

test "route context write json" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    var ctx = web.RouteContext.init(allocator, &handler);
    defer ctx.deinit();

    try ctx.writeJson("{\"status\":\"ok\"}");

    try std.testing.expectEqualStrings("{\"status\":\"ok\"}", ctx.response_body.items);
    try std.testing.expectEqualStrings("application/json", ctx.response_content_type);
}

// =============================================================================
// Route Definitions Tests
// =============================================================================

test "all routes have descriptions" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    const router = web.PersonaRouter.init(allocator, &handler);

    for (router.routes) |route| {
        try std.testing.expect(route.description.len > 0);
        try std.testing.expect(route.path.len > 0);
    }
}

test "route definitions include required endpoints" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    const router = web.PersonaRouter.init(allocator, &handler);

    const required_paths = [_][]const u8{
        "/api/v1/chat",
        "/api/v1/chat/abbey",
        "/api/v1/chat/aviva",
        "/api/v1/personas",
        "/api/v1/personas/metrics",
        "/api/v1/personas/health",
    };

    for (required_paths) |required_path| {
        var found = false;
        for (router.routes) |route| {
            if (std.mem.eql(u8, route.path, required_path)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}

test "openapi spec generation" {
    const allocator = std.testing.allocator;

    const spec = try web.routes.personas.generateOpenApiSpec(allocator);
    defer allocator.free(spec);

    // Should be valid JSON structure
    try std.testing.expect(std.mem.indexOf(u8, spec, "openapi") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "3.0.0") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "paths") != null);

    // Should include chat endpoint
    try std.testing.expect(std.mem.indexOf(u8, spec, "/api/v1/chat") != null);
}

// =============================================================================
// HTTP Method Tests
// =============================================================================

test "method enum values" {
    const Method = web.routes.personas.Method;

    try std.testing.expectEqual(Method.GET, Method.GET);
    try std.testing.expectEqual(Method.POST, Method.POST);
    try std.testing.expectEqual(Method.PUT, Method.PUT);
    try std.testing.expectEqual(Method.DELETE, Method.DELETE);
    try std.testing.expectEqual(Method.PATCH, Method.PATCH);
    try std.testing.expectEqual(Method.OPTIONS, Method.OPTIONS);
    try std.testing.expectEqual(Method.HEAD, Method.HEAD);
}

// =============================================================================
// HTTP Status Codes Tests
// =============================================================================

test "http status codes" {
    const HttpStatus = web.handlers.chat.HttpStatus;

    try std.testing.expectEqual(@as(u16, 200), HttpStatus.ok);
    try std.testing.expectEqual(@as(u16, 201), HttpStatus.created);
    try std.testing.expectEqual(@as(u16, 400), HttpStatus.bad_request);
    try std.testing.expectEqual(@as(u16, 401), HttpStatus.unauthorized);
    try std.testing.expectEqual(@as(u16, 404), HttpStatus.not_found);
    try std.testing.expectEqual(@as(u16, 405), HttpStatus.method_not_allowed);
    try std.testing.expectEqual(@as(u16, 500), HttpStatus.internal_server_error);
    try std.testing.expectEqual(@as(u16, 503), HttpStatus.service_unavailable);
}

// =============================================================================
// Persona Type Parsing Tests
// =============================================================================

test "parse persona type" {
    const parsePersonaType = web.handlers.chat.parsePersonaType;

    // Valid personas
    try std.testing.expect(parsePersonaType("abbey") != null);
    try std.testing.expect(parsePersonaType("aviva") != null);
    try std.testing.expect(parsePersonaType("abi") != null);

    // Invalid personas
    try std.testing.expect(parsePersonaType("unknown") == null);
    try std.testing.expect(parsePersonaType("") == null);
    try std.testing.expect(parsePersonaType("ABBEY") == null); // Case sensitive
}

// =============================================================================
// Web Context Tests
// =============================================================================

test "web context initialization" {
    if (!web.isEnabled()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const config_module = @import("../config/mod.zig");

    const ctx = try web.Context.init(allocator, config_module.WebConfig{});
    defer ctx.deinit();

    try std.testing.expect(ctx.http_client != null);
}

// =============================================================================
// Integration Tests
// =============================================================================

test "full router request handling for not found" {
    const allocator = std.testing.allocator;

    var handler = web.ChatHandler.init(allocator);
    var router = web.PersonaRouter.init(allocator, &handler);

    const result = try router.handle("/api/v1/nonexistent", .GET, "");
    defer if (result.status == 500) allocator.free(result.body);

    try std.testing.expectEqual(@as(u16, 404), result.status);
    try std.testing.expect(std.mem.indexOf(u8, result.body, "NOT_FOUND") != null);
}

test "route result struct" {
    const result = web.routes.personas.RouteResult{
        .status = 200,
        .body = "{\"ok\":true}",
        .content_type = "application/json",
    };

    try std.testing.expectEqual(@as(u16, 200), result.status);
    try std.testing.expectEqualStrings("application/json", result.content_type);
}
