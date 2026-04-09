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

/// Create an MCP server pre-configured with status/diagnostics tools
pub fn createStatusServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-status", version);

    try server.addTool(.{
        .def = .{
            .name = "abi_chat",
            .description = "Route a message through the ABI multi-profile pipeline and get an AI response",
            .input_schema =
            \\{"type":"object","properties":{"message":{"type":"string","description":"User message to process"}},"required":["message"]}
            ,
        },
        .handler = ai.handleAbiChat,
    });

    try server.addTool(.{
        .def = .{
            .name = "abi_status",
            .description = "Report ABI server status including name, version, tool count, and uptime",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = status.handleAbiStatus,
    });

    try server.addTool(.{
        .def = .{
            .name = "abi_health",
            .description = "Health check — returns ok if the server is responsive",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = status.handleAbiHealth,
    });

    try server.addTool(.{
        .def = .{
            .name = "abi_features",
            .description = "List all compile-time enabled feature flags for this ABI build",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = status.handleAbiFeatures,
    });

    try server.addTool(.{
        .def = .{
            .name = "abi_version",
            .description = "Return ABI version string, protocol version, and Zig compiler version",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = status.handleAbiVersion,
    });

    try server.addTool(.{
        .def = .{
            .name = "hardware_status",
            .description = "Query system hardware capabilities including CPU cores, RAM, and GPU/VRAM details",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = status.handleHardwareStatus,
    });

    return server;
}

/// Copy all tools from `src_server` into `dst`.
fn mergeServerTools(dst: *Server, src_server: *Server) !void {
    for (src_server.tools.items) |tool| {
        try dst.addTool(tool);
    }
}

/// Create an MCP server pre-configured with all tools (status + database + ZLS + Discord)
pub fn createCombinedServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-full", version);

    // Unpack status tools
    var status_server = try createStatusServer(allocator, version);
    defer status_server.deinit();
    try mergeServerTools(&server, &status_server);

    // Unpack database tools
    var database_server = try createDatabaseServer(allocator, version);
    defer database_server.deinit();
    try mergeServerTools(&server, &database_server);

    // Unpack ZLS tools
    var zls_server = try zls_bridge.createZlsServer(allocator, version);
    defer zls_server.deinit();
    try mergeServerTools(&server, &zls_server);

    // Unpack Discord tools
    var discord_server = try createDiscordServer(allocator, version);
    defer discord_server.deinit();
    try mergeServerTools(&server, &discord_server);

    return server;
}

/// Create an MCP server pre-configured with Discord REST API tools
pub fn createDiscordServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-discord", version);

    const tools = [_]struct { name: []const u8, desc: []const u8, schema: []const u8, handler: @TypeOf(discord_handlers.handleDiscordSendMessage) }{
        .{ .name = "discord_send_message", .desc = "Send a message to a Discord channel", .schema = 
        \\{"type":"object","properties":{"channel_id":{"type":"string","description":"Discord channel ID"},"content":{"type":"string","description":"Message content"}},"required":["channel_id","content"]}
        , .handler = discord_handlers.handleDiscordSendMessage },
        .{ .name = "discord_send_embed", .desc = "Send a rich embed message to a Discord channel", .schema = 
        \\{"type":"object","properties":{"channel_id":{"type":"string","description":"Discord channel ID"},"title":{"type":"string","description":"Embed title"},"description":{"type":"string","description":"Embed description"},"content":{"type":"string","description":"Optional text content"},"color":{"type":"integer","description":"Embed color (decimal)"}},"required":["channel_id","title"]}
        , .handler = discord_handlers.handleDiscordSendEmbed },
        .{ .name = "discord_edit_message", .desc = "Edit an existing Discord message", .schema = 
        \\{"type":"object","properties":{"channel_id":{"type":"string"},"message_id":{"type":"string"},"content":{"type":"string","description":"New content"}},"required":["channel_id","message_id","content"]}
        , .handler = discord_handlers.handleDiscordEditMessage },
        .{ .name = "discord_delete_message", .desc = "Delete a Discord message", .schema = 
        \\{"type":"object","properties":{"channel_id":{"type":"string"},"message_id":{"type":"string"}},"required":["channel_id","message_id"]}
        , .handler = discord_handlers.handleDiscordDeleteMessage },
        .{ .name = "discord_get_messages", .desc = "Get recent messages from a Discord channel", .schema = 
        \\{"type":"object","properties":{"channel_id":{"type":"string","description":"Discord channel ID"},"limit":{"type":"integer","description":"Max messages (1-100, default 50)","default":50}},"required":["channel_id"]}
        , .handler = discord_handlers.handleDiscordGetMessages },
        .{ .name = "discord_get_channel", .desc = "Get Discord channel details", .schema = 
        \\{"type":"object","properties":{"channel_id":{"type":"string","description":"Discord channel ID"}},"required":["channel_id"]}
        , .handler = discord_handlers.handleDiscordGetChannel },
        .{ .name = "discord_react", .desc = "Add a reaction to a Discord message", .schema = 
        \\{"type":"object","properties":{"channel_id":{"type":"string"},"message_id":{"type":"string"},"emoji":{"type":"string","description":"Emoji (e.g. %F0%9F%91%8D or custom:name:id)"}},"required":["channel_id","message_id","emoji"]}
        , .handler = discord_handlers.handleDiscordReact },
        .{ .name = "discord_typing", .desc = "Show typing indicator in a Discord channel", .schema = 
        \\{"type":"object","properties":{"channel_id":{"type":"string"}},"required":["channel_id"]}
        , .handler = discord_handlers.handleDiscordTyping },
        .{ .name = "discord_get_guild", .desc = "Get Discord server (guild) details", .schema = 
        \\{"type":"object","properties":{"guild_id":{"type":"string","description":"Discord guild/server ID"}},"required":["guild_id"]}
        , .handler = discord_handlers.handleDiscordGetGuild },
        .{ .name = "discord_get_guild_channels", .desc = "List all channels in a Discord server", .schema = 
        \\{"type":"object","properties":{"guild_id":{"type":"string","description":"Discord guild/server ID"}},"required":["guild_id"]}
        , .handler = discord_handlers.handleDiscordGetGuildChannels },
        .{ .name = "discord_list_guilds", .desc = "List all Discord servers the bot is in", .schema = 
        \\{"type":"object","properties":{},"required":[]}
        , .handler = discord_handlers.handleDiscordListGuilds },
        .{ .name = "discord_get_bot", .desc = "Get the bot's own Discord user info", .schema = 
        \\{"type":"object","properties":{},"required":[]}
        , .handler = discord_handlers.handleDiscordGetBot },
        .{ .name = "discord_create_dm", .desc = "Open a DM channel with a Discord user", .schema = 
        \\{"type":"object","properties":{"user_id":{"type":"string","description":"Discord user ID"}},"required":["user_id"]}
        , .handler = discord_handlers.handleDiscordCreateDM },
        .{ .name = "discord_execute_webhook", .desc = "Execute a Discord webhook", .schema = 
        \\{"type":"object","properties":{"webhook_id":{"type":"string"},"token":{"type":"string"},"content":{"type":"string"}},"required":["webhook_id","token","content"]}
        , .handler = discord_handlers.handleDiscordExecuteWebhook },
        .{ .name = "discord_get_member", .desc = "Get a member's details in a Discord server", .schema = 
        \\{"type":"object","properties":{"guild_id":{"type":"string"},"user_id":{"type":"string"}},"required":["guild_id","user_id"]}
        , .handler = discord_handlers.handleDiscordGetMember },
        .{ .name = "discord_register_command", .desc = "Register a global slash command for the bot", .schema = 
        \\{"type":"object","properties":{"application_id":{"type":"string","description":"Discord application ID"},"name":{"type":"string","description":"Command name (lowercase, 1-32 chars)"},"description":{"type":"string","description":"Command description (1-100 chars)"}},"required":["application_id","name","description"]}
        , .handler = discord_handlers.handleDiscordRegisterCommand },
        .{ .name = "discord_list_commands", .desc = "List all registered global slash commands", .schema = 
        \\{"type":"object","properties":{"application_id":{"type":"string","description":"Discord application ID"}},"required":["application_id"]}
        , .handler = discord_handlers.handleDiscordListCommands },
        .{ .name = "discord_delete_command", .desc = "Delete a global slash command by ID", .schema = 
        \\{"type":"object","properties":{"application_id":{"type":"string","description":"Discord application ID"},"command_id":{"type":"string","description":"Command ID to delete"}},"required":["application_id","command_id"]}
        , .handler = discord_handlers.handleDiscordDeleteCommand },
        .{ .name = "discord_get_message", .desc = "Get a specific Discord message by ID", .schema = 
        \\{"type":"object","properties":{"channel_id":{"type":"string"},"message_id":{"type":"string"}},"required":["channel_id","message_id"]}
        , .handler = discord_handlers.handleDiscordGetMessage },
    };

    inline for (tools) |t| {
        try server.addTool(.{
            .def = .{
                .name = t.name,
                .description = t.desc,
                .input_schema = t.schema,
            },
            .handler = t.handler,
        });
    }

    return server;
}

/// Create an MCP server pre-configured with database tools
pub fn createDatabaseServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-database", version);

    try server.addTool(.{
        .def = .{
            .name = "db_query",
            .description = "Search for similar vectors in the database using cosine similarity",
            .input_schema =
            \\{"type":"object","properties":{"vector":{"type":"array","items":{"type":"number"},"description":"Query vector (float32 array)"},"top_k":{"type":"integer","description":"Number of results to return (default: 5)","default":5},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["vector"]}
            ,
        },
        .handler = database.handleDbQuery,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_insert",
            .description = "Insert a vector with optional metadata into the database",
            .input_schema =
            \\{"type":"object","properties":{"id":{"type":"integer","description":"Unique vector ID"},"vector":{"type":"array","items":{"type":"number"},"description":"Vector data (float32 array)"},"metadata":{"type":"string","description":"Optional metadata string"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["id","vector"]}
            ,
        },
        .handler = database.handleDbInsert,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_stats",
            .description = "Get statistics about the database (vector count, dimensions, memory usage)",
            .input_schema =
            \\{"type":"object","properties":{"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":[]}
            ,
        },
        .handler = database.handleDbStats,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_list",
            .description = "List vectors stored in the database",
            .input_schema =
            \\{"type":"object","properties":{"limit":{"type":"integer","description":"Max vectors to return (default: 10)","default":10},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":[]}
            ,
        },
        .handler = database.handleDbList,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_delete",
            .description = "Delete a vector by ID from the database",
            .input_schema =
            \\{"type":"object","properties":{"id":{"type":"integer","description":"Vector ID to delete"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["id"]}
            ,
        },
        .handler = database.handleDbDelete,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_get",
            .description = "Retrieve a single vector by ID from the database",
            .input_schema =
            \\{"type":"object","properties":{"id":{"type":"integer","description":"Vector ID to retrieve"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["id"]}
            ,
        },
        .handler = database.handleDbGet,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_update",
            .description = "Update an existing vector's data in the database",
            .input_schema =
            \\{"type":"object","properties":{"id":{"type":"integer","description":"Vector ID to update"},"vector":{"type":"array","items":{"type":"number"},"description":"New vector data (float32 array)"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["id","vector"]}
            ,
        },
        .handler = database.handleDbUpdate,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_backup",
            .description = "Save the database to a file for persistence or recovery",
            .input_schema =
            \\{"type":"object","properties":{"path":{"type":"string","description":"File path to save the backup"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["path"]}
            ,
        },
        .handler = database.handleDbBackup,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_diagnostics",
            .description = "Get detailed performance diagnostics for the database",
            .input_schema =
            \\{"type":"object","properties":{"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":[]}
            ,
        },
        .handler = database.handleDbDiagnostics,
    });

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
    try std.testing.expectEqualStrings("abi_chat", server.tools.items[0].def.name);
    try std.testing.expectEqualStrings("abi_status", server.tools.items[1].def.name);
    try std.testing.expectEqualStrings("abi_health", server.tools.items[2].def.name);
    try std.testing.expectEqualStrings("abi_features", server.tools.items[3].def.name);
    try std.testing.expectEqualStrings("abi_version", server.tools.items[4].def.name);
    try std.testing.expectEqualStrings("hardware_status", server.tools.items[5].def.name);
}

test {
    std.testing.refAllDecls(@This());
}
