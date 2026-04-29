//! Integration Tests: ACP OpenAPI Specification Generator
//!
//! Verifies the OpenAPI 3.1.0 spec generator from a consumer perspective:
//! JSON validity, route coverage, component schemas, x-state-machine extension,
//! card configuration reflection, and Server.getOrBuildOpenApiSpec caching.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const acp = abi.acp;

// ============================================================================
// Helpers
// ============================================================================

fn defaultCard() acp.AgentCard {
    return .{
        .name = "openapi-test-agent",
        .description = "Integration test agent for OpenAPI spec",
        .version = "1.0.0",
        .url = "http://localhost:8080",
        .capabilities = .{},
    };
}

/// Generate the spec and parse it as JSON, returning the parsed tree.
/// Caller must call parsed.deinit() and allocator.free(raw).
fn generateAndParse(allocator: std.mem.Allocator, card: acp.AgentCard) !struct {
    parsed: std.json.Parsed(std.json.Value),
    raw: []u8,
} {
    const raw = try acp.openapi.generate(allocator, card);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    return .{ .parsed = parsed, .raw = raw };
}

// ============================================================================
// JSON validity and top-level structure
// ============================================================================

test "acp openapi: generate produces valid JSON with top-level keys" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const root = result.parsed.value.object;

    // Must have the required top-level OpenAPI keys
    try std.testing.expect(root.contains("openapi"));
    try std.testing.expect(root.contains("info"));
    try std.testing.expect(root.contains("servers"));
    try std.testing.expect(root.contains("paths"));
    try std.testing.expect(root.contains("components"));
    try std.testing.expect(root.contains("tags"));
}

test "acp openapi: openapi version is 3.1.0" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const version = result.parsed.value.object.get("openapi").?.string;
    try std.testing.expectEqualStrings("3.1.0", version);
}

// ============================================================================
// Info section reflects card fields
// ============================================================================

test "acp openapi: info section reflects agent card" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const card = acp.AgentCard{
        .name = "my-special-agent",
        .description = "A very special agent for testing",
        .version = "2.5.0",
        .url = "https://agents.example.com:9090",
        .capabilities = .{ .streaming = true },
    };

    const result = try generateAndParse(allocator, card);
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const info = result.parsed.value.object.get("info").?.object;
    try std.testing.expectEqualStrings("my-special-agent", info.get("title").?.string);
    try std.testing.expectEqualStrings("A very special agent for testing", info.get("description").?.string);
    try std.testing.expectEqualStrings("2.5.0", info.get("version").?.string);
}

test "acp openapi: servers section reflects card URL" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const card = acp.AgentCard{
        .name = "test",
        .description = "test",
        .version = "0.1.0",
        .url = "https://my-server.example.com:4443",
        .capabilities = .{},
    };

    const result = try generateAndParse(allocator, card);
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const servers = result.parsed.value.object.get("servers").?.array;
    try std.testing.expect(servers.items.len >= 1);
    const server_url = servers.items[0].object.get("url").?.string;
    try std.testing.expectEqualStrings("https://my-server.example.com:4443", server_url);
}

// ============================================================================
// Route coverage — all ROUTE_TABLE paths present
// ============================================================================

test "acp openapi: paths contains all route table entries" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const paths = result.parsed.value.object.get("paths").?.object;

    // Verify every route from the comptime ROUTE_TABLE is present
    for (acp.openapi.ROUTE_TABLE) |route| {
        const entry = paths.get(route.path);
        if (entry == null) {
            std.debug.print("Missing path in OpenAPI spec: {s}\n", .{route.path});
        }
        try std.testing.expect(entry != null);
    }

    // Total path count should match
    try std.testing.expectEqual(acp.openapi.ROUTE_TABLE.len, paths.count());
}

test "acp openapi: each path has correct HTTP method" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const paths = result.parsed.value.object.get("paths").?.object;

    for (acp.openapi.ROUTE_TABLE) |route| {
        const path_obj = paths.get(route.path).?.object;
        const method_str = switch (route.method) {
            .GET => "get",
            .POST => "post",
        };
        try std.testing.expect(path_obj.contains(method_str));
    }
}

test "acp openapi: POST routes have requestBody" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const paths = result.parsed.value.object.get("paths").?.object;

    for (acp.openapi.ROUTE_TABLE) |route| {
        if (route.method == .POST) {
            const path_obj = paths.get(route.path).?.object;
            const method_obj = path_obj.get("post").?.object;
            try std.testing.expect(method_obj.contains("requestBody"));
        }
    }
}

test "acp openapi: parameterized paths have parameters array" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const paths = result.parsed.value.object.get("paths").?.object;

    for (acp.openapi.ROUTE_TABLE) |route| {
        const has_param = std.mem.indexOf(u8, route.path, "{") != null;
        if (has_param) {
            const path_obj = paths.get(route.path).?.object;
            const method_str = switch (route.method) {
                .GET => "get",
                .POST => "post",
            };
            const method_obj = path_obj.get(method_str).?.object;
            try std.testing.expect(method_obj.contains("parameters"));
        }
    }
}

// ============================================================================
// Component schemas
// ============================================================================

test "acp openapi: components contains all expected schemas" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const schemas = result.parsed.value.object
        .get("components").?.object
        .get("schemas").?.object;

    const expected_schemas = [_][]const u8{
        "Task",
        "TaskMessage",
        "TaskStatus",
        "Session",
        "AgentCard",
        "AgentCapabilities",
        "DiscordSendRequest",
        "DiscordWebhookRequest",
        "ErrorResponse",
    };

    for (expected_schemas) |name| {
        if (!schemas.contains(name)) {
            std.debug.print("Missing schema: {s}\n", .{name});
        }
        try std.testing.expect(schemas.contains(name));
    }
}

test "acp openapi: Task schema has required fields" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const task_schema = result.parsed.value.object
        .get("components").?.object
        .get("schemas").?.object
        .get("Task").?.object;

    // Must be type: object
    try std.testing.expectEqualStrings("object", task_schema.get("type").?.string);

    // Must have properties
    const props = task_schema.get("properties").?.object;
    try std.testing.expect(props.contains("id"));
    try std.testing.expect(props.contains("status"));
    try std.testing.expect(props.contains("messages"));

    // Must have required array
    const required = task_schema.get("required").?.array;
    try std.testing.expect(required.items.len >= 2);
}

// ============================================================================
// x-state-machine extension
// ============================================================================

test "acp openapi: TaskStatus has x-state-machine extension" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const status_schema = result.parsed.value.object
        .get("components").?.object
        .get("schemas").?.object
        .get("TaskStatus").?.object;

    // Must have x-state-machine
    try std.testing.expect(status_schema.contains("x-state-machine"));

    const state_machine = status_schema.get("x-state-machine").?.object;

    // All status values should be keys in the state machine
    try std.testing.expect(state_machine.contains("submitted"));
    try std.testing.expect(state_machine.contains("working"));
    try std.testing.expect(state_machine.contains("input-required"));
    try std.testing.expect(state_machine.contains("completed"));
    try std.testing.expect(state_machine.contains("failed"));
    try std.testing.expect(state_machine.contains("canceled"));
}

test "acp openapi: TaskStatus enum values present" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const status_schema = result.parsed.value.object
        .get("components").?.object
        .get("schemas").?.object
        .get("TaskStatus").?.object;

    try std.testing.expectEqualStrings("string", status_schema.get("type").?.string);

    const enum_values = status_schema.get("enum").?.array;
    try std.testing.expectEqual(@as(usize, 6), enum_values.items.len);
}

// ============================================================================
// Tags
// ============================================================================

test "acp openapi: tags array contains expected tag groups" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const result = try generateAndParse(allocator, defaultCard());
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    const tags = result.parsed.value.object.get("tags").?.array;

    // Collect tag names
    var found_agent = false;
    var found_tasks = false;
    var found_sessions = false;
    var found_discord = false;
    var found_meta = false;

    for (tags.items) |tag| {
        const name = tag.object.get("name").?.string;
        if (std.mem.eql(u8, name, "Agent")) found_agent = true;
        if (std.mem.eql(u8, name, "Tasks")) found_tasks = true;
        if (std.mem.eql(u8, name, "Sessions")) found_sessions = true;
        if (std.mem.eql(u8, name, "Discord")) found_discord = true;
        if (std.mem.eql(u8, name, "Meta")) found_meta = true;
    }

    try std.testing.expect(found_agent);
    try std.testing.expect(found_tasks);
    try std.testing.expect(found_sessions);
    try std.testing.expect(found_discord);
    try std.testing.expect(found_meta);
}

// ============================================================================
// Server.getOrBuildOpenApiSpec caching
// ============================================================================

test "acp openapi: Server.getOrBuildOpenApiSpec returns cached spec" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    var server = acp.Server.init(allocator, defaultCard());
    defer server.deinit();

    const spec1 = try server.getOrBuildOpenApiSpec();
    const spec2 = try server.getOrBuildOpenApiSpec();

    // Both calls should return the exact same pointer (cached)
    try std.testing.expectEqual(spec1.ptr, spec2.ptr);
    try std.testing.expectEqual(spec1.len, spec2.len);

    // And it should be valid JSON
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, spec1, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value == .object);
}

// ============================================================================
// Different card configurations
// ============================================================================

test "acp openapi: spec with special characters in card fields" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const card = acp.AgentCard{
        .name = "agent \"alpha\" & beta",
        .description = "Line1\nLine2\\End",
        .version = "0.0.1",
        .url = "http://localhost:8080/path?q=1&r=2",
        .capabilities = .{},
    };

    const result = try generateAndParse(allocator, card);
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    // If we got here, the JSON is valid despite special characters
    const info = result.parsed.value.object.get("info").?.object;
    try std.testing.expect(info.contains("title"));
    try std.testing.expect(info.contains("description"));
}

test "acp openapi: spec with all capabilities enabled" {
    if (!build_options.feat_acp) return;
    const allocator = std.testing.allocator;

    const card = acp.AgentCard{
        .name = "full-caps",
        .description = "All capabilities enabled",
        .version = "3.0.0",
        .url = "https://example.com",
        .capabilities = .{
            .streaming = true,
            .pushNotifications = true,
            .stateTransitionHistory = true,
            .extensions = true,
        },
    };

    const result = try generateAndParse(allocator, card);
    defer result.parsed.deinit();
    defer allocator.free(result.raw);

    // Spec should still be valid — capabilities don't affect spec structure
    // but the card name should appear in info.title
    const info = result.parsed.value.object.get("info").?.object;
    try std.testing.expectEqualStrings("full-caps", info.get("title").?.string);
}

test {
    std.testing.refAllDecls(@This());
}
