//! ACP Agent Card — describes an agent's capabilities and skills.

const std = @import("std");
const json_utils = @import("json_utils.zig");
const appendEscaped = json_utils.appendEscaped;

/// ACP Agent Card — describes this agent's capabilities
pub const AgentCard = struct {
    name: []const u8,
    description: []const u8,
    version: []const u8,
    url: []const u8,
    capabilities: Capabilities,

    pub const Capabilities = struct {
        streaming: bool = false,
        pushNotifications: bool = false,
        stateTransitionHistory: bool = false,
        extensions: bool = false,
    };

    /// Serialize to JSON (escapes all string fields for safety)
    pub fn toJson(self: AgentCard, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\"name\":\"");
        try appendEscaped(allocator, &buf, self.name);
        try buf.appendSlice(allocator, "\",\"description\":\"");
        try appendEscaped(allocator, &buf, self.description);
        try buf.appendSlice(allocator, "\",\"version\":\"");
        try appendEscaped(allocator, &buf, self.version);
        try buf.appendSlice(allocator, "\",\"url\":\"");
        try appendEscaped(allocator, &buf, self.url);
        try buf.appendSlice(allocator, "\",\"capabilities\":{\"streaming\":");
        try buf.appendSlice(allocator, if (self.capabilities.streaming) "true" else "false");
        try buf.appendSlice(allocator, ",\"pushNotifications\":");
        try buf.appendSlice(allocator, if (self.capabilities.pushNotifications) "true" else "false");
        try buf.appendSlice(allocator, ",\"stateTransitionHistory\":");
        try buf.appendSlice(allocator, if (self.capabilities.stateTransitionHistory) "true" else "false");
        try buf.appendSlice(allocator, ",\"extensions\":");
        try buf.appendSlice(allocator, if (self.capabilities.extensions) "true" else "false");
        try buf.appendSlice(allocator, "},\"skills\":[{\"id\":\"db_query\",\"name\":\"Vector Search\",\"description\":\"Search the WDBX vector database\"},{\"id\":\"db_insert\",\"name\":\"Vector Insert\",\"description\":\"Insert vectors with metadata\"},{\"id\":\"agent_chat\",\"name\":\"Chat\",\"description\":\"Conversational interaction\"}]}");

        return buf.toOwnedSlice(allocator);
    }
};

test "AgentCard toJson" {
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test-agent",
        .description = "A test agent",
        .version = "0.1.0",
        .url = "http://localhost:8080",
        .capabilities = .{},
    };
    const json = try card.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "test-agent") != null);
}

test "AgentCard toJson escapes special characters" {
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test\"agent",
        .description = "line1\nline2\\end",
        .version = "0.1.0",
        .url = "http://localhost:8080",
        .capabilities = .{ .streaming = true },
    };
    const json = try card.toJson(allocator);
    defer allocator.free(json);

    // Verify special chars are escaped
    try std.testing.expect(std.mem.indexOf(u8, json, "test\\\"agent") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "line1\\nline2\\\\end") != null);
    // Verify streaming capability
    try std.testing.expect(std.mem.indexOf(u8, json, "\"streaming\":true") != null);
}

test "AgentCard toJson includes new capabilities" {
    const allocator = std.testing.allocator;
    const card = AgentCard{
        .name = "test-agent",
        .description = "A test agent",
        .version = "0.1.0",
        .url = "http://localhost:8080",
        .capabilities = .{ .streaming = true, .stateTransitionHistory = true },
    };
    const json = try card.toJson(allocator);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"stateTransitionHistory\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"extensions\":false") != null);
}

const env_gate = @import("common");
test {
    if (!env_gate.canRunTest()) return;
    std.testing.refAllDecls(@This());
}
