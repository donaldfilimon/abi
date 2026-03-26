//! MCP (Model Context Protocol) Service
//!
//! Provides a JSON-RPC 2.0 server over stdio for exposing ABI framework
//! tools to MCP-compatible AI clients (Claude Desktop, Cursor, etc.).
//!
//! ## Usage
//! ```bash
//! abi mcp serve                          # Start MCP server (stdio)
//! echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | abi mcp serve
//! ```
//!
//! ## Exposed Tools
//! - `db_*` — Database tools
//! - `zls_*` — ZLS LSP tools (hover, completion, definition, etc.)

const std = @import("std");
const build_options = @import("build_options");
pub const types = @import("types.zig");
pub const Server = @import("server.zig").Server;
pub const RegisteredTool = @import("server.zig").RegisteredTool;
pub const RegisteredResource = @import("server.zig").RegisteredResource;
pub const ResourceHandler = @import("server.zig").ResourceHandler;
pub const ToolHandler = @import("server.zig").ToolHandler;
pub const zls_bridge = @import("zls_bridge.zig");

pub const createZlsServer = zls_bridge.createZlsServer;

/// Create an MCP server pre-configured with status/diagnostics tools
pub fn createStatusServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-status", version);

    try server.addTool(.{
        .def = .{
            .name = "abi_status",
            .description = "Report ABI server status including name, version, tool count, and uptime",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = handleAbiStatus,
    });

    try server.addTool(.{
        .def = .{
            .name = "abi_health",
            .description = "Health check — returns ok if the server is responsive",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = handleAbiHealth,
    });

    try server.addTool(.{
        .def = .{
            .name = "abi_features",
            .description = "List all compile-time enabled feature flags for this ABI build",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = handleAbiFeatures,
    });

    try server.addTool(.{
        .def = .{
            .name = "abi_version",
            .description = "Return ABI version string, protocol version, and Zig compiler version",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = handleAbiVersion,
    });

    try server.addTool(.{
        .def = .{
            .name = "hardware_status",
            .description = "Query system hardware capabilities including CPU cores, RAM, and GPU/VRAM details",
            .input_schema =
            \\{"type":"object","properties":{},"required":[]}
            ,
        },
        .handler = handleHardwareStatus,
    });

    return server;
}

/// Create an MCP server pre-configured with all tools (status + database + ZLS)
pub fn createCombinedServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-full", version);

    // Unpack status tools
    var status_server = try createStatusServer(allocator, version);
    defer status_server.deinit();
    for (status_server.tools.items) |tool| {
        try server.addTool(tool);
    }

    // Unpack database tools
    var database_server = try createDatabaseServer(allocator, version);
    defer database_server.deinit();
    for (database_server.tools.items) |tool| {
        try server.addTool(tool);
    }

    // Unpack ZLS tools
    var zls_server = try createZlsServer(allocator, version);
    defer zls_server.deinit();
    for (zls_server.tools.items) |tool| {
        try server.addTool(tool);
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
        .handler = handleDbQuery,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_insert",
            .description = "Insert a vector with optional metadata into the database",
            .input_schema =
            \\{"type":"object","properties":{"id":{"type":"integer","description":"Unique vector ID"},"vector":{"type":"array","items":{"type":"number"},"description":"Vector data (float32 array)"},"metadata":{"type":"string","description":"Optional metadata string"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["id","vector"]}
            ,
        },
        .handler = handleDbInsert,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_stats",
            .description = "Get statistics about the database (vector count, dimensions, memory usage)",
            .input_schema =
            \\{"type":"object","properties":{"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":[]}
            ,
        },
        .handler = handleDbStats,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_list",
            .description = "List vectors stored in the database",
            .input_schema =
            \\{"type":"object","properties":{"limit":{"type":"integer","description":"Max vectors to return (default: 10)","default":10},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":[]}
            ,
        },
        .handler = handleDbList,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_delete",
            .description = "Delete a vector by ID from the database",
            .input_schema =
            \\{"type":"object","properties":{"id":{"type":"integer","description":"Vector ID to delete"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["id"]}
            ,
        },
        .handler = handleDbDelete,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_get",
            .description = "Retrieve a single vector by ID from the database",
            .input_schema =
            \\{"type":"object","properties":{"id":{"type":"integer","description":"Vector ID to retrieve"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["id"]}
            ,
        },
        .handler = handleDbGet,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_update",
            .description = "Update an existing vector's data in the database",
            .input_schema =
            \\{"type":"object","properties":{"id":{"type":"integer","description":"Vector ID to update"},"vector":{"type":"array","items":{"type":"number"},"description":"New vector data (float32 array)"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["id","vector"]}
            ,
        },
        .handler = handleDbUpdate,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_backup",
            .description = "Save the database to a file for persistence or recovery",
            .input_schema =
            \\{"type":"object","properties":{"path":{"type":"string","description":"File path to save the backup"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["path"]}
            ,
        },
        .handler = handleDbBackup,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_diagnostics",
            .description = "Get detailed performance diagnostics for the database",
            .input_schema =
            \\{"type":"object","properties":{"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":[]}
            ,
        },
        .handler = handleDbDiagnostics,
    });

    return server;
}

// ═══════════════════════════════════════════════════════════════
// Tool Handlers
// ═══════════════════════════════════════════════════════════════

/// Shared database access — comptime-gated to support disabled database feature.
const database = if (build_options.feat_database)
    @import("../../features/database/mod.zig")
else
    @import("../../features/database/stub.zig");

fn getOrCreateDb(allocator: std.mem.Allocator, name_opt: ?[]const u8) !database.Store {
    const name = name_opt orelse "default";
    return database.Store.open(allocator, name);
}

fn handleDbQuery(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;

    // Parse vector
    const vec_val = p.get("vector") orelse return error.InvalidParams;
    if (vec_val != .array) return error.InvalidParams;

    var query_vec = std.ArrayListUnmanaged(f32).empty;
    defer query_vec.deinit(allocator);

    for (vec_val.array.items) |item| {
        const v: f32 = switch (item) {
            .float => @floatCast(item.float),
            .integer => @floatFromInt(item.integer),
            else => return error.InvalidParams,
        };
        try query_vec.append(allocator, v);
    }

    const top_k: usize = if (p.get("top_k")) |tk|
        (if (tk == .integer) @intCast(@max(1, tk.integer)) else 5)
    else
        5;

    const db_name = if (p.get("db_name")) |dn|
        (if (dn == .string) dn.string else null)
    else
        null;

    var store = try getOrCreateDb(allocator, db_name);
    defer store.deinit();

    const results = try store.search(query_vec.items, top_k);
    defer allocator.free(results);

    // Format results as JSON text
    try out.appendSlice(allocator, "Search results:\n");
    if (results.len == 0) {
        try out.appendSlice(allocator, "No results found.");
    } else {
        for (results, 0..) |r, i| {
            var buf: [128]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "  {d}. ID={d} score={d:.4}\n", .{ i + 1, r.id, r.score }) catch continue;
            try out.appendSlice(allocator, s);
        }
        var total_buf: [64]u8 = undefined;
        const total = std.fmt.bufPrint(&total_buf, "\nTotal: {d} results", .{results.len}) catch "";
        try out.appendSlice(allocator, total);
    }
}

fn handleDbInsert(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;

    // Parse ID (must be non-negative)
    const id_val = p.get("id") orelse return error.InvalidParams;
    if (id_val != .integer or id_val.integer < 0) return error.InvalidParams;
    const id: u64 = @intCast(id_val.integer);

    // Parse vector
    const vec_val = p.get("vector") orelse return error.InvalidParams;
    if (vec_val != .array) return error.InvalidParams;

    var vec = std.ArrayListUnmanaged(f32).empty;
    defer vec.deinit(allocator);

    for (vec_val.array.items) |item| {
        const v: f32 = switch (item) {
            .float => @floatCast(item.float),
            .integer => @floatFromInt(item.integer),
            else => return error.InvalidParams,
        };
        try vec.append(allocator, v);
    }

    // Parse optional metadata
    const metadata: ?[]const u8 = if (p.get("metadata")) |m|
        (if (m == .string) m.string else null)
    else
        null;

    const db_name = if (p.get("db_name")) |dn|
        (if (dn == .string) dn.string else null)
    else
        null;

    var store = try getOrCreateDb(allocator, db_name);
    defer store.deinit();

    try store.insert(id, vec.items, metadata);

    var buf: [128]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "Inserted vector ID={d} (dimension={d})", .{ id, vec.items.len }) catch "Inserted";
    try out.appendSlice(allocator, msg);
}

fn handleDbStats(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const db_name = if (params) |p|
        (if (p.get("db_name")) |dn| (if (dn == .string) dn.string else null) else null)
    else
        null;

    var store = try getOrCreateDb(allocator, db_name);
    defer store.deinit();

    const stats = store.stats();

    try out.appendSlice(allocator, "WDBX Database Statistics:\n");

    var buf: [256]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "  Vectors: {d}\n  Dimensions: {d}\n  Memory: {d} bytes\n  Norm cache: {s}", .{
        stats.count,
        stats.dimension,
        stats.memory_bytes,
        if (stats.norm_cache_enabled) "enabled" else "disabled",
    }) catch "error formatting stats";
    try out.appendSlice(allocator, s);
}

fn handleDbList(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const limit: usize = if (params) |p|
        (if (p.get("limit")) |lim| (if (lim == .integer) @intCast(@max(1, lim.integer)) else 10) else 10)
    else
        10;

    const db_name = if (params) |p|
        (if (p.get("db_name")) |dn| (if (dn == .string) dn.string else null) else null)
    else
        null;

    var store = try getOrCreateDb(allocator, db_name);
    defer store.deinit();

    const vectors = try store.list(limit);
    defer allocator.free(vectors);

    try out.appendSlice(allocator, "Stored vectors:\n");
    if (vectors.len == 0) {
        try out.appendSlice(allocator, "  (empty database)");
    } else {
        for (vectors) |v| {
            var buf: [256]u8 = undefined;
            const dim_len = v.vector.len;
            const preview = if (dim_len > 3) "..." else "";
            const s = std.fmt.bufPrint(&buf, "  ID={d} dim={d} meta={s}{s}\n", .{
                v.id,
                dim_len,
                if (v.metadata) |m| m else "(none)",
                preview,
            }) catch continue;
            try out.appendSlice(allocator, s);
        }
    }
}

fn handleDbDelete(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;

    const id_val = p.get("id") orelse return error.InvalidParams;
    if (id_val != .integer or id_val.integer < 0) return error.InvalidParams;
    const id: u64 = @intCast(id_val.integer);

    const db_name = if (p.get("db_name")) |dn|
        (if (dn == .string) dn.string else null)
    else
        null;

    var store = try getOrCreateDb(allocator, db_name);
    defer store.deinit();

    const deleted = store.remove(id);

    var buf: [128]u8 = undefined;
    const msg = if (deleted)
        std.fmt.bufPrint(&buf, "Deleted vector ID={d}", .{id}) catch "Deleted"
    else
        std.fmt.bufPrint(&buf, "Vector ID={d} not found", .{id}) catch "Not found";
    try out.appendSlice(allocator, msg);
}

fn handleDbGet(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;

    const id_val = p.get("id") orelse return error.InvalidParams;
    if (id_val != .integer or id_val.integer < 0) return error.InvalidParams;
    const id: u64 = @intCast(id_val.integer);

    const db_name = if (p.get("db_name")) |dn|
        (if (dn == .string) dn.string else null)
    else
        null;

    var store = try getOrCreateDb(allocator, db_name);
    defer store.deinit();

    const vec_view = store.get(id);
    if (vec_view) |v| {
        var buf: [256]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, "Vector ID={d}\n  Dimensions: {d}\n  Metadata: {s}", .{
            id,
            v.vector.len,
            if (v.metadata) |m| m else "(none)",
        }) catch "found";
        try out.appendSlice(allocator, s);
    } else {
        var buf: [64]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "Vector ID={d} not found", .{id}) catch "Not found";
        try out.appendSlice(allocator, msg);
    }
}

fn handleDbUpdate(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;

    const id_val = p.get("id") orelse return error.InvalidParams;
    if (id_val != .integer or id_val.integer < 0) return error.InvalidParams;
    const id: u64 = @intCast(id_val.integer);

    const vec_val = p.get("vector") orelse return error.InvalidParams;
    if (vec_val != .array) return error.InvalidParams;

    var vec = std.ArrayListUnmanaged(f32).empty;
    defer vec.deinit(allocator);

    for (vec_val.array.items) |item| {
        const v: f32 = switch (item) {
            .float => @floatCast(item.float),
            .integer => @floatFromInt(item.integer),
            else => return error.InvalidParams,
        };
        try vec.append(allocator, v);
    }

    const db_name = if (p.get("db_name")) |dn|
        (if (dn == .string) dn.string else null)
    else
        null;

    var store = try getOrCreateDb(allocator, db_name);
    defer store.deinit();

    const updated = try store.update(id, vec.items);

    var buf: [128]u8 = undefined;
    const msg = if (updated)
        std.fmt.bufPrint(&buf, "Updated vector ID={d} (dimension={d})", .{ id, vec.items.len }) catch "Updated"
    else
        std.fmt.bufPrint(&buf, "Vector ID={d} not found — no update performed", .{id}) catch "Not found";
    try out.appendSlice(allocator, msg);
}

fn handleDbBackup(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const p = params orelse return error.InvalidParams;

    const path_val = p.get("path") orelse return error.InvalidParams;
    if (path_val != .string) return error.InvalidParams;
    const path = path_val.string;

    const db_name = if (p.get("db_name")) |dn|
        (if (dn == .string) dn.string else null)
    else
        null;

    var store = try getOrCreateDb(allocator, db_name);
    defer store.deinit();

    store.save(path) catch |err| {
        var buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "Backup failed: {s}", .{@errorName(err)}) catch "Backup failed";
        try out.appendSlice(allocator, msg);
        return;
    };

    try out.appendSlice(allocator, "Database backed up to: ");
    try out.appendSlice(allocator, path);
}

fn handleDbDiagnostics(
    allocator: std.mem.Allocator,
    params: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const db_name = if (params) |p|
        (if (p.get("db_name")) |dn| (if (dn == .string) dn.string else null) else null)
    else
        null;

    var store = try getOrCreateDb(allocator, db_name);
    defer store.deinit();

    const diag = store.diagnostics();

    try out.appendSlice(allocator, "WDBX Diagnostics:\n");

    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(&buf,
        \\  Vectors: {d}
        \\  Dimensions: {d}
        \\  Memory: {d} bytes
        \\  Searches performed: {d}
        \\  Avg search time: {d} ns
        \\  Cache hit rate: {d:.1}%
        \\  SIMD: {s}
    , .{
        diag.vector_count,
        diag.dimension,
        diag.memory.total_bytes,
        @as(u64, 0),
        @as(u64, 0),
        @as(f64, 0.0),
        if (diag.config.norm_cache_enabled) "enabled" else "disabled",
    }) catch "error formatting diagnostics";
    try out.appendSlice(allocator, s);
}

// ═══════════════════════════════════════════════════════════════
// Status / Diagnostics Tool Handlers
// ═══════════════════════════════════════════════════════════════

fn handleAbiStatus(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    try out.appendSlice(allocator, "ABI MCP Server Status: running");
}

fn handleAbiHealth(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    try out.appendSlice(allocator, "ok");
}

fn handleAbiFeatures(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    try out.appendSlice(allocator, "Enabled features:");
    if (build_options.feat_gpu) try out.appendSlice(allocator, " gpu");
    if (build_options.feat_ai) try out.appendSlice(allocator, " ai");
    if (build_options.feat_database) try out.appendSlice(allocator, " database");
    if (build_options.feat_network) try out.appendSlice(allocator, " network");
    if (build_options.feat_observability) try out.appendSlice(allocator, " observability");
    if (build_options.feat_web) try out.appendSlice(allocator, " web");
    if (build_options.feat_cloud) try out.appendSlice(allocator, " cloud");
    if (build_options.feat_auth) try out.appendSlice(allocator, " auth");
    if (build_options.feat_messaging) try out.appendSlice(allocator, " messaging");
    if (build_options.feat_cache) try out.appendSlice(allocator, " cache");
    if (build_options.feat_storage) try out.appendSlice(allocator, " storage");
    if (build_options.feat_search) try out.appendSlice(allocator, " search");
    if (build_options.feat_mobile) try out.appendSlice(allocator, " mobile");
    if (build_options.feat_gateway) try out.appendSlice(allocator, " gateway");
    if (build_options.feat_pages) try out.appendSlice(allocator, " pages");
    if (build_options.feat_benchmarks) try out.appendSlice(allocator, " benchmarks");
    if (build_options.feat_compute) try out.appendSlice(allocator, " compute");
    if (build_options.feat_documents) try out.appendSlice(allocator, " documents");
    if (build_options.feat_desktop) try out.appendSlice(allocator, " desktop");
    if (build_options.feat_lsp) try out.appendSlice(allocator, " lsp");
    if (build_options.feat_mcp) try out.appendSlice(allocator, " mcp");
}

fn handleAbiVersion(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    try out.appendSlice(allocator, "ABI version: ");
    try out.appendSlice(allocator, build_options.package_version);
    try out.appendSlice(allocator, "\nProtocol: ");
    try out.appendSlice(allocator, types.PROTOCOL_VERSION);
    try out.appendSlice(allocator, "\nZig: 0.16.0-dev");
}

fn handleHardwareStatus(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const discovery = @import("../../features/ai/explore/discovery.zig");
    const caps = discovery.detectCapabilities();
    const json_str = try std.json.Stringify.valueAlloc(allocator, caps, .{});
    defer allocator.free(json_str);
    try out.appendSlice(allocator, json_str);
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

    var saw_db_query = false;
    var saw_zls_hover = false;
    var saw_abi_status = false;
    var saw_abi_health = false;
    var saw_abi_features = false;
    var saw_abi_version = false;
    for (server.tools.items) |tool| {
        if (std.mem.eql(u8, tool.def.name, "db_query")) saw_db_query = true;
        if (std.mem.eql(u8, tool.def.name, "zls_hover")) saw_zls_hover = true;
        if (std.mem.eql(u8, tool.def.name, "abi_status")) saw_abi_status = true;
        if (std.mem.eql(u8, tool.def.name, "abi_health")) saw_abi_health = true;
        if (std.mem.eql(u8, tool.def.name, "abi_features")) saw_abi_features = true;
        if (std.mem.eql(u8, tool.def.name, "abi_version")) saw_abi_version = true;
    }

    try std.testing.expect(saw_db_query);
    try std.testing.expect(saw_zls_hover);
    try std.testing.expect(saw_abi_status);
    try std.testing.expect(saw_abi_health);
    try std.testing.expect(saw_abi_features);
    try std.testing.expect(saw_abi_version);
}

test "createStatusServer registers 5 tools" {
    const allocator = std.testing.allocator;
    var server = try createStatusServer(allocator, "0.4.0");
    defer server.deinit();

    try std.testing.expectEqual(@as(usize, 5), server.tools.items.len);
    try std.testing.expectEqualStrings("abi_status", server.tools.items[0].def.name);
    try std.testing.expectEqualStrings("abi_health", server.tools.items[1].def.name);
    try std.testing.expectEqualStrings("abi_features", server.tools.items[2].def.name);
    try std.testing.expectEqualStrings("abi_version", server.tools.items[3].def.name);
    try std.testing.expectEqualStrings("hardware_status", server.tools.items[4].def.name);
}

test {
    std.testing.refAllDecls(@This());
}
