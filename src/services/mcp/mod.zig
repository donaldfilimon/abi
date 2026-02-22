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
//! - `db_query` — Vector similarity search
//! - `db_insert` — Insert vectors with metadata
//! - `db_stats` — Database statistics
//! - `db_list` — List stored vectors
//! - `db_delete` — Delete a vector by ID
//! - `zls_*` — ZLS LSP tools (hover, completion, definition, etc.)

const std = @import("std");
pub const types = @import("types.zig");
pub const Server = @import("server.zig").Server;
pub const RegisteredTool = @import("server.zig").RegisteredTool;
pub const zls_bridge = @import("zls_bridge.zig");

pub const createZlsServer = zls_bridge.createZlsServer;

/// Create an MCP server pre-configured with WDBX database tools
pub fn createWdbxServer(allocator: std.mem.Allocator, version: []const u8) !Server {
    var server = Server.init(allocator, "abi-wdbx", version);

    try server.addTool(.{
        .def = .{
            .name = "db_query",
            .description = "Search for similar vectors in the WDBX database using cosine similarity",
            .input_schema =
            \\{"type":"object","properties":{"vector":{"type":"array","items":{"type":"number"},"description":"Query vector (float32 array)"},"top_k":{"type":"integer","description":"Number of results to return (default: 5)","default":5},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["vector"]}
            ,
        },
        .handler = handleDbQuery,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_insert",
            .description = "Insert a vector with optional metadata into the WDBX database",
            .input_schema =
            \\{"type":"object","properties":{"id":{"type":"integer","description":"Unique vector ID"},"vector":{"type":"array","items":{"type":"number"},"description":"Vector data (float32 array)"},"metadata":{"type":"string","description":"Optional metadata string"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["id","vector"]}
            ,
        },
        .handler = handleDbInsert,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_stats",
            .description = "Get statistics about the WDBX database (vector count, dimensions, memory usage)",
            .input_schema =
            \\{"type":"object","properties":{"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":[]}
            ,
        },
        .handler = handleDbStats,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_list",
            .description = "List vectors stored in the WDBX database",
            .input_schema =
            \\{"type":"object","properties":{"limit":{"type":"integer","description":"Max vectors to return (default: 10)","default":10},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":[]}
            ,
        },
        .handler = handleDbList,
    });

    try server.addTool(.{
        .def = .{
            .name = "db_delete",
            .description = "Delete a vector by ID from the WDBX database",
            .input_schema =
            \\{"type":"object","properties":{"id":{"type":"integer","description":"Vector ID to delete"},"db_name":{"type":"string","description":"Database name (default: default)","default":"default"}},"required":["id"]}
            ,
        },
        .handler = handleDbDelete,
    });

    return server;
}

// ═══════════════════════════════════════════════════════════════
// Tool Handlers
// ═══════════════════════════════════════════════════════════════

/// Shared database access — uses the database feature module
const database = @import("../../features/database/mod.zig");

fn getOrCreateDb(allocator: std.mem.Allocator, name_opt: ?[]const u8) !database.DatabaseHandle {
    const name = name_opt orelse "default";
    return database.open(allocator, name) catch |err| {
        if (err == error.DatabaseAlreadyExists) {
            return database.connect(allocator, name);
        }
        return err;
    };
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

    var handle = try getOrCreateDb(allocator, db_name);
    defer database.close(&handle);

    const results = try database.search(&handle, allocator, query_vec.items, top_k);
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

    var handle = try getOrCreateDb(allocator, db_name);
    defer database.close(&handle);

    try database.insert(&handle, id, vec.items, metadata);

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

    var handle = try getOrCreateDb(allocator, db_name);
    defer database.close(&handle);

    const stats = database.stats(&handle);

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

    var handle = try getOrCreateDb(allocator, db_name);
    defer database.close(&handle);

    const vectors = try database.list(&handle, allocator, limit);
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

    var handle = try getOrCreateDb(allocator, db_name);
    defer database.close(&handle);

    const deleted = database.remove(&handle, id);

    var buf: [128]u8 = undefined;
    const msg = if (deleted)
        std.fmt.bufPrint(&buf, "Deleted vector ID={d}", .{id}) catch "Deleted"
    else
        std.fmt.bufPrint(&buf, "Vector ID={d} not found", .{id}) catch "Not found";
    try out.appendSlice(allocator, msg);
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

test "createWdbxServer registers tools" {
    const allocator = std.testing.allocator;
    var server = try createWdbxServer(allocator, "0.4.0");
    defer server.deinit();

    try std.testing.expectEqual(@as(usize, 5), server.tools.items.len);
    try std.testing.expectEqualStrings("db_query", server.tools.items[0].def.name);
    try std.testing.expectEqualStrings("db_insert", server.tools.items[1].def.name);
    try std.testing.expectEqualStrings("db_stats", server.tools.items[2].def.name);
    try std.testing.expectEqualStrings("db_list", server.tools.items[3].def.name);
    try std.testing.expectEqualStrings("db_delete", server.tools.items[4].def.name);
}
