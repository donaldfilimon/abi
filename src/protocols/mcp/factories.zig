//! MCP Server Factory Functions
//!
//! Pre-configured server constructors that register tool sets for
//! status/diagnostics, database operations, ZLS tooling, and combined servers.

const std = @import("std");
const Server = @import("server.zig").Server;
const zls_bridge = @import("zls_bridge.zig");
const status = @import("handlers/status.zig");
const database = @import("handlers/database.zig");
const ai = @import("handlers/ai.zig");
const discord_handlers = @import("handlers/discord.zig");
const registry = @import("registry.zig");

/// Create an MCP server pre-configured with status/diagnostics tools
/// Uses registry pattern for modular tool registration
pub fn createStatusServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-status", version);
    try registry.registerModules(&server, .{status});
    return server;
}

/// Create an MCP server pre-configured with all tools (status + database + ZLS + Discord)
/// Uses registry pattern for declarative tool registration
pub fn createCombinedServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-full", version);

    // Aggregate tools using registry pattern
    try registry.registerModules(&server, .{
        status,
        database,
        ai,
        discord_handlers,
    });

    // Merge ZLS tools (ZLS bridge doesn't yet use registry pattern)
    var zls_server = try zls_bridge.createZlsServer(allocator, version);
    defer zls_server.deinit();
    for (zls_server.tools.items) |tool| {
        try server.addTool(tool);
    }

    return server;
}

/// Create an MCP server pre-configured with Discord REST API tools
pub fn createDiscordServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-discord", version);
    try registry.registerModules(&server, .{discord_handlers});
    return server;
}

/// Create an MCP server pre-configured with database tools
pub fn createDatabaseServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-database", version);
    try registry.registerModules(&server, .{database});
    return server;
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

test "createDatabaseServer registers tools" {
    const allocator = std.testing.allocator;
    var server = try createDatabaseServer(allocator, "0.4.0");
    defer server.deinit();

    try std.testing.expectEqual(@as(usize, 9), server.tools.items.len);
    try std.testing.expectEqualStrings("db_query", server.tools.items[0].def.name);
    try std.testing.expectEqualStrings("db_insert", server.tools.items[1].def.name);
    try std.testing.expectEqualStrings("db_stats", server.tools.items[2].def.name);
    try std.testing.expectEqualStrings("db_list", server.tools.items[3].def.name);
    try std.testing.expectEqualStrings("db_delete", server.tools.items[4].def.name);
    try std.testing.expectEqualStrings("db_get", server.tools.items[5].def.name);
    try std.testing.expectEqualStrings("db_update", server.tools.items[6].def.name);
    try std.testing.expectEqualStrings("db_backup", server.tools.items[7].def.name);
    try std.testing.expectEqualStrings("db_diagnostics", server.tools.items[8].def.name);
}

test "createCombinedServer registers database and ZLS tools" {
    const allocator = std.testing.allocator;
    var server = try createCombinedServer(allocator, "0.4.0");
    defer server.deinit();

    var saw_abi_chat = false;
    var saw_db_query = false;
    var saw_zls_hover = false;
    var saw_abi_status = false;
    var saw_abi_health = false;
    var saw_abi_features = false;
    var saw_abi_version = false;
    var saw_hardware_status = false;
    var saw_discord_send = false;
    var saw_discord_get_bot = false;
    for (server.tools.items) |tool| {
        if (std.mem.eql(u8, tool.def.name, "abi_chat")) saw_abi_chat = true;
        if (std.mem.eql(u8, tool.def.name, "db_query")) saw_db_query = true;
        if (std.mem.eql(u8, tool.def.name, "zls_hover")) saw_zls_hover = true;
        if (std.mem.eql(u8, tool.def.name, "abi_status")) saw_abi_status = true;
        if (std.mem.eql(u8, tool.def.name, "abi_health")) saw_abi_health = true;
        if (std.mem.eql(u8, tool.def.name, "abi_features")) saw_abi_features = true;
        if (std.mem.eql(u8, tool.def.name, "abi_version")) saw_abi_version = true;
        if (std.mem.eql(u8, tool.def.name, "hardware_status")) saw_hardware_status = true;
        if (std.mem.eql(u8, tool.def.name, "discord_send_message")) saw_discord_send = true;
        if (std.mem.eql(u8, tool.def.name, "discord_get_bot")) saw_discord_get_bot = true;
    }

    try std.testing.expect(saw_abi_chat);
    try std.testing.expect(saw_db_query);
    try std.testing.expect(saw_zls_hover);
    try std.testing.expect(saw_abi_status);
    try std.testing.expect(saw_abi_health);
    try std.testing.expect(saw_abi_features);
    try std.testing.expect(saw_abi_version);
    try std.testing.expect(saw_hardware_status);
    try std.testing.expect(saw_discord_send);
    try std.testing.expect(saw_discord_get_bot);
}

test "createDiscordServer registers 19 tools" {
    const allocator = std.testing.allocator;
    var server = try createDiscordServer(allocator, "0.4.0");
    defer server.deinit();

    try std.testing.expectEqual(@as(usize, 19), server.tools.items.len);
    try std.testing.expectEqualStrings("discord_send_message", server.tools.items[0].def.name);
    try std.testing.expectEqualStrings("discord_get_bot", server.tools.items[11].def.name);
    try std.testing.expectEqualStrings("discord_register_command", server.tools.items[15].def.name);
    try std.testing.expectEqualStrings("discord_list_commands", server.tools.items[16].def.name);
    try std.testing.expectEqualStrings("discord_delete_command", server.tools.items[17].def.name);
    try std.testing.expectEqualStrings("discord_get_message", server.tools.items[18].def.name);
}

test "createStatusServer registers 6 tools" {
    const allocator = std.testing.allocator;
    var server = try createStatusServer(allocator, "0.4.0");
    defer server.deinit();

    try std.testing.expectEqual(@as(usize, 6), server.tools.items.len);
    try std.testing.expectEqualStrings("abi_status", server.tools.items[0].def.name);
    try std.testing.expectEqualStrings("abi_health", server.tools.items[1].def.name);
    try std.testing.expectEqualStrings("abi_features", server.tools.items[2].def.name);
    try std.testing.expectEqualStrings("abi_version", server.tools.items[3].def.name);
    try std.testing.expectEqualStrings("hardware_status", server.tools.items[4].def.name);
    try std.testing.expectEqualStrings("abi_chat", server.tools.items[5].def.name);
}

test {
    std.testing.refAllDecls(@This());
}
