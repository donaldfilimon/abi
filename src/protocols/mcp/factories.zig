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

/// Create an MCP server pre-configured with all tools (status + database + ZLS)
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
    for (server.tools.items) |tool| {
        if (std.mem.eql(u8, tool.def.name, "abi_chat")) saw_abi_chat = true;
        if (std.mem.eql(u8, tool.def.name, "db_query")) saw_db_query = true;
        if (std.mem.eql(u8, tool.def.name, "zls_hover")) saw_zls_hover = true;
        if (std.mem.eql(u8, tool.def.name, "abi_status")) saw_abi_status = true;
        if (std.mem.eql(u8, tool.def.name, "abi_health")) saw_abi_health = true;
        if (std.mem.eql(u8, tool.def.name, "abi_features")) saw_abi_features = true;
        if (std.mem.eql(u8, tool.def.name, "abi_version")) saw_abi_version = true;
        if (std.mem.eql(u8, tool.def.name, "hardware_status")) saw_hardware_status = true;
    }

    try std.testing.expect(saw_abi_chat);
    try std.testing.expect(saw_db_query);
    try std.testing.expect(saw_zls_hover);
    try std.testing.expect(saw_abi_status);
    try std.testing.expect(saw_abi_health);
    try std.testing.expect(saw_abi_features);
    try std.testing.expect(saw_abi_version);
    try std.testing.expect(saw_hardware_status);
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
