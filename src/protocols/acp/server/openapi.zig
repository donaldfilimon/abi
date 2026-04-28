//! OpenAPI 3.1.0 Specification Generator
//!
//! Generates a complete OpenAPI spec from a comptime route table and the
//! agent card metadata. The spec is lazily generated on first request and
//! cached on the Server struct.

const std = @import("std");
const parity_gate = @import("../../../common/parity_gate.zig");
const json_utils = @import("json_utils.zig");
const appendEscaped = json_utils.appendEscaped;
const AgentCard = @import("agent_card.zig").AgentCard;

// ============================================================================
// Route Table (comptime)
// ============================================================================

const Method = enum { GET, POST };

const RouteSpec = struct {
    method: Method,
    path: []const u8,
    summary: []const u8,
    tag: []const u8,
};

pub const ROUTE_TABLE = [_]RouteSpec{
    // ACP core endpoints
    .{ .method = .GET, .path = "/.well-known/agent.json", .summary = "Get agent card", .tag = "Agent" },
    .{ .method = .POST, .path = "/tasks/send", .summary = "Create a new task", .tag = "Tasks" },
    .{ .method = .GET, .path = "/tasks/{taskId}", .summary = "Get task by ID", .tag = "Tasks" },
    .{ .method = .POST, .path = "/tasks/{taskId}/status", .summary = "Update task status", .tag = "Tasks" },
    .{ .method = .POST, .path = "/sessions", .summary = "Create a new session", .tag = "Sessions" },
    .{ .method = .GET, .path = "/sessions/{sessionId}", .summary = "Get session by ID", .tag = "Sessions" },
    .{ .method = .POST, .path = "/sessions/{sessionId}/tasks", .summary = "Add task to session", .tag = "Sessions" },
    // Discord endpoints
    .{ .method = .POST, .path = "/discord/send", .summary = "Send a Discord message", .tag = "Discord" },
    .{ .method = .GET, .path = "/discord/channels/{channelId}", .summary = "Get a Discord channel", .tag = "Discord" },
    .{ .method = .POST, .path = "/discord/webhook", .summary = "Execute a Discord webhook", .tag = "Discord" },
    .{ .method = .GET, .path = "/discord/guilds", .summary = "List bot guilds", .tag = "Discord" },
    .{ .method = .GET, .path = "/discord/bot", .summary = "Get bot user info", .tag = "Discord" },
    // OpenAPI endpoint itself
    .{ .method = .GET, .path = "/openapi.json", .summary = "Get OpenAPI specification", .tag = "Meta" },
};

// ============================================================================
// Generator
// ============================================================================

/// Generate a complete OpenAPI 3.1.0 JSON spec from the route table and card.
/// Caller owns the returned slice.
pub fn generate(allocator: std.mem.Allocator, card: AgentCard) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8).empty;
    errdefer buf.deinit(allocator);

    // Header
    try buf.appendSlice(allocator, "{\"openapi\":\"3.1.0\",\"info\":{\"title\":\"");
    try appendEscaped(allocator, &buf, card.name);
    try buf.appendSlice(allocator, "\",\"description\":\"");
    try appendEscaped(allocator, &buf, card.description);
    try buf.appendSlice(allocator, "\",\"version\":\"");
    try appendEscaped(allocator, &buf, card.version);
    try buf.appendSlice(allocator, "\"},\"servers\":[{\"url\":\"");
    try appendEscaped(allocator, &buf, card.url);
    try buf.appendSlice(allocator, "\"}],\"paths\":{");

    // Paths
    var first_path = true;
    for (ROUTE_TABLE) |route| {
        if (!first_path) try buf.append(allocator, ',');
        first_path = false;

        try buf.append(allocator, '"');
        try appendEscaped(allocator, &buf, route.path);
        try buf.appendSlice(allocator, "\":{\"");
        try buf.appendSlice(allocator, switch (route.method) {
            .GET => "get",
            .POST => "post",
        });
        try buf.appendSlice(allocator, "\":{\"summary\":\"");
        try appendEscaped(allocator, &buf, route.summary);
        try buf.appendSlice(allocator, "\",\"tags\":[\"");
        try appendEscaped(allocator, &buf, route.tag);
        try buf.appendSlice(allocator, "\"],\"responses\":{\"200\":{\"description\":\"Success\",\"content\":{\"application/json\":{\"schema\":{\"type\":\"object\"}}}}");

        // Add error responses
        if (route.method == .POST) {
            try buf.appendSlice(allocator, ",\"400\":{\"description\":\"Bad Request\"}");
        }
        try buf.appendSlice(allocator, ",\"404\":{\"description\":\"Not Found\"}");
        try buf.appendSlice(allocator, "}");

        // Add requestBody for POST routes
        if (route.method == .POST) {
            try buf.appendSlice(allocator, ",\"requestBody\":{\"required\":true,\"content\":{\"application/json\":{\"schema\":{\"type\":\"object\"}}}}");
        }

        // Add path parameters
        if (std.mem.indexOf(u8, route.path, "{")) |_| {
            try appendPathParams(allocator, &buf, route.path);
        }

        try buf.appendSlice(allocator, "}}");
    }

    // Components
    try buf.appendSlice(allocator, "},\"components\":{\"schemas\":{");
    try appendSchemas(allocator, &buf);
    try buf.appendSlice(allocator, "}}");

    // Tags
    try buf.appendSlice(allocator,
        \\,"tags":[{"name":"Agent","description":"Agent card and capabilities"},{"name":"Tasks","description":"Task management"},{"name":"Sessions","description":"Session management"},{"name":"Discord","description":"Discord integration"},{"name":"Meta","description":"API metadata"}]
    );

    try buf.append(allocator, '}');

    return buf.toOwnedSlice(allocator);
}

fn appendPathParams(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), path: []const u8) !void {
    try buf.appendSlice(allocator, ",\"parameters\":[");
    var first = true;
    var i: usize = 0;
    while (i < path.len) {
        if (path[i] == '{') {
            const end = std.mem.indexOfScalarPos(u8, path, i + 1, '}') orelse break;
            const param_name = path[i + 1 .. end];

            if (!first) try buf.append(allocator, ',');
            first = false;

            try buf.appendSlice(allocator, "{\"name\":\"");
            try appendEscaped(allocator, buf, param_name);
            try buf.appendSlice(allocator, "\",\"in\":\"path\",\"required\":true,\"schema\":{\"type\":\"string\"}}");

            i = end + 1;
        } else {
            i += 1;
        }
    }
    try buf.append(allocator, ']');
}

fn appendSchemas(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8)) !void {
    // Task schema with x-state-machine extension
    try buf.appendSlice(allocator,
        \\"Task":{"type":"object","properties":{"id":{"type":"string"},"status":{"$ref":"#/components/schemas/TaskStatus"},"messages":{"type":"array","items":{"$ref":"#/components/schemas/TaskMessage"}}},"required":["id","status"]},
    );
    try buf.appendSlice(allocator,
        \\"TaskMessage":{"type":"object","properties":{"role":{"type":"string","enum":["user","agent"]},"content":{"type":"string"}},"required":["role","content"]},
    );
    try buf.appendSlice(allocator,
        \\"TaskStatus":{"type":"string","enum":["submitted","working","input-required","completed","failed","canceled"],"x-state-machine":{"submitted":["working"],"working":["completed","failed","input-required"],"input-required":["working"],"completed":["canceled"],"failed":["canceled"],"canceled":[]}},
    );
    try buf.appendSlice(allocator,
        \\"Session":{"type":"object","properties":{"id":{"type":"string"},"created_at":{"type":"integer"},"metadata":{"type":"string","nullable":true},"task_ids":{"type":"array","items":{"type":"string"}}},"required":["id","created_at"]},
    );
    try buf.appendSlice(allocator,
        \\"AgentCard":{"type":"object","properties":{"name":{"type":"string"},"description":{"type":"string"},"version":{"type":"string"},"url":{"type":"string","format":"uri"},"capabilities":{"$ref":"#/components/schemas/AgentCapabilities"}},"required":["name","description","version","url","capabilities"]},
    );
    try buf.appendSlice(allocator,
        \\"AgentCapabilities":{"type":"object","properties":{"streaming":{"type":"boolean"},"pushNotifications":{"type":"boolean"},"stateTransitionHistory":{"type":"boolean"},"extensions":{"type":"boolean"}}},
    );
    try buf.appendSlice(allocator,
        \\"DiscordSendRequest":{"type":"object","properties":{"channel_id":{"type":"string"},"content":{"type":"string"}},"required":["channel_id","content"]},
    );
    try buf.appendSlice(allocator,
        \\"DiscordWebhookRequest":{"type":"object","properties":{"webhook_id":{"type":"string"},"webhook_token":{"type":"string"},"content":{"type":"string"}},"required":["webhook_id","webhook_token","content"]},
    );
    try buf.appendSlice(allocator,
        \\"ErrorResponse":{"type":"object","properties":{"error":{"type":"string"}},"required":["error"]}
    );
}

// ============================================================================
// Tests
// ============================================================================

test "openapi: generate produces valid JSON" {
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test-agent",
        .description = "A test agent",
        .version = "0.1.0",
        .url = "http://localhost:8080",
        .capabilities = .{},
    };

    const spec = try generate(allocator, card);
    defer allocator.free(spec);

    // Verify it's parseable JSON
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, spec, .{});
    defer parsed.deinit();

    try std.testing.expect(parsed.value == .object);
}

test "openapi: spec contains openapi version" {
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test",
        .description = "test",
        .version = "0.0.1",
        .url = "http://localhost",
        .capabilities = .{},
    };

    const spec = try generate(allocator, card);
    defer allocator.free(spec);

    try std.testing.expect(std.mem.indexOf(u8, spec, "\"openapi\":\"3.1.0\"") != null);
}

test "openapi: spec contains all route paths" {
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test",
        .description = "test",
        .version = "0.0.1",
        .url = "http://localhost",
        .capabilities = .{},
    };

    const spec = try generate(allocator, card);
    defer allocator.free(spec);

    // Check all 13 routes are present
    for (ROUTE_TABLE) |route| {
        try std.testing.expect(std.mem.indexOf(u8, spec, route.path) != null);
    }
}

test "openapi: spec contains TaskStatus values" {
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test",
        .description = "test",
        .version = "0.0.1",
        .url = "http://localhost",
        .capabilities = .{},
    };

    const spec = try generate(allocator, card);
    defer allocator.free(spec);

    try std.testing.expect(std.mem.indexOf(u8, spec, "submitted") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "working") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "completed") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "failed") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "canceled") != null);
}

test "openapi: spec contains x-state-machine" {
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test",
        .description = "test",
        .version = "0.0.1",
        .url = "http://localhost",
        .capabilities = .{},
    };

    const spec = try generate(allocator, card);
    defer allocator.free(spec);

    try std.testing.expect(std.mem.indexOf(u8, spec, "x-state-machine") != null);
}

test "openapi: spec reflects card fields" {
    if (!parity_gate.canRunTest()) return;
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "my-custom-agent",
        .description = "Custom description here",
        .version = "2.3.4",
        .url = "https://example.com:9090",
        .capabilities = .{},
    };

    const spec = try generate(allocator, card);
    defer allocator.free(spec);

    try std.testing.expect(std.mem.indexOf(u8, spec, "my-custom-agent") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "Custom description here") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "2.3.4") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "https://example.com:9090") != null);
}

test "openapi: spec contains discord routes" {
    if (!parity_gate.canRunTest()) return;
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test",
        .description = "test",
        .version = "0.0.1",
        .url = "http://localhost",
        .capabilities = .{},
    };

    const spec = try generate(allocator, card);
    defer allocator.free(spec);

    try std.testing.expect(std.mem.indexOf(u8, spec, "/discord/send") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "/discord/channels/") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "/discord/webhook") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "/discord/guilds") != null);
    try std.testing.expect(std.mem.indexOf(u8, spec, "/discord/bot") != null);
}

test {
    if (!parity_gate.canRunTest()) return;
    std.testing.refAllDecls(@This());
}
