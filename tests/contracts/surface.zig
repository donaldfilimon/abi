const std = @import("std");
const usage = @import("cli_usage");
const handlers = @import("mcp_handlers");

test "CLI command surface is frozen" {
    const expected = [_][]const u8{
        "help",   "complete", "train",  "agent", "backends",
        "plugin", "auth",     "twilio", "tui",   "dashboard",
    };
    try std.testing.expectEqual(expected.len, usage.commands.len);
    for (expected, usage.commands) |name, cmd| {
        try std.testing.expectEqualStrings(name, cmd.name);
    }
}

test "MCP tools/list includes contract tools" {
    const json = try handlers.handleToolsListJson(std.testing.allocator);
    defer std.testing.allocator.free(json);
    for ([_][]const u8{ "ai_run", "ai_complete", "ai_train", "wdbx_query" }) |tool| {
        try std.testing.expect(std.mem.indexOf(u8, json, tool) != null);
    }
}

test "MCP initialize advertises abi-mcp" {
    const json = try handlers.handleInitializeJson(std.testing.allocator, null);
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "abi-mcp") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "2024-11-05") != null);
}
