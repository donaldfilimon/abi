const std = @import("std");
const handlers = @import("mcp_handlers");

fn expectToolJsonContains(allocator: std.mem.Allocator, json_text: []const u8, needle: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();
    const response = try handlers.handleToolsCallJson(allocator, parsed.value);
    defer allocator.free(response);
    try std.testing.expect(std.mem.indexOf(u8, response, needle) != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"type\":\"text\"") != null);
}

fn expectToolJsonContainsEither(allocator: std.mem.Allocator, json_text: []const u8, a: []const u8, b: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();
    const response = try handlers.handleToolsCallJson(allocator, parsed.value);
    defer allocator.free(response);
    try std.testing.expect(std.mem.indexOf(u8, response, a) != null or std.mem.indexOf(u8, response, b) != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"type\":\"text\"") != null);
}

test "MCP ai_run tool contract" {
    const allocator = std.testing.allocator;
    try expectToolJsonContainsEither(
        allocator,
        \\{"name":"ai_run","arguments":{"input":"execute deploy quickly"}}
    ,
        "Abi action",
        "AI feature is disabled",
    );
}

test "MCP ai_complete tool contract" {
    const allocator = std.testing.allocator;
    try expectToolJsonContains(
        allocator,
        \\{"name":"ai_complete","arguments":{"input":"hello","model":"abi-local"}}
    ,
        "model=abi-local",
    );
}

test "MCP ai_train tool contract" {
    const allocator = std.testing.allocator;
    try expectToolJsonContains(
        allocator,
        \\{"name":"ai_train","arguments":{"profile":"abi","dataset":"data.jsonl"}}
    ,
        "training accepted",
    );
}

test "MCP wdbx_query tool contract" {
    const allocator = std.testing.allocator;
    try expectToolJsonContains(
        allocator,
        \\{"name":"wdbx_query","arguments":{"query":"creative explore ideas"}}
    ,
        "wdbx local match",
    );
}
