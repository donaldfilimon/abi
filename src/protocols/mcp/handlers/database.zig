//! MCP Database Tool Handlers
//!
//! Handlers for `db_query`, `db_insert`, `db_stats`, `db_list`, `db_delete`,
//! `db_get`, `db_update`, `db_backup`, and `db_diagnostics` tools.

const std = @import("std");
const build_options = @import("build_options");

/// Shared database access — comptime-gated to support disabled database feature.
const database = if (build_options.feat_database)
    @import("../../../features/database/mod.zig")
else
    @import("../../../features/database/stub.zig");

fn getOrCreateDb(allocator: std.mem.Allocator, name_opt: ?[]const u8) !database.Store {
    const name = name_opt orelse "default";
    return database.Store.open(allocator, name);
}

pub fn handleDbQuery(
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

pub fn handleDbInsert(
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

pub fn handleDbStats(
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

pub fn handleDbList(
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

pub fn handleDbDelete(
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

pub fn handleDbGet(
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

pub fn handleDbUpdate(
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

pub fn handleDbBackup(
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

pub fn handleDbDiagnostics(
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

test {
    std.testing.refAllDecls(@This());
}
