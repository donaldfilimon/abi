const std = @import("std");
const abi = @import("../../abi.zig");

const Db = abi.database.database.Db;

pub const WdbxResult = extern struct {
    index: u64,
    score: f32,
};

/// Global allocator for C API functions. Can be set via wdbx_set_allocator().
var global_allocator: std.mem.Allocator = std.heap.page_allocator;

fn toDb(handle: ?*anyopaque) ?*Db {
    if (handle == null) return null;
    const ptr = handle.?;
    return @ptrCast(@alignCast(ptr));
}

/// Set the global allocator for C API functions
export fn wdbx_set_allocator(allocator: std.mem.Allocator) callconv(.c) void {
    global_allocator = allocator;
}

export fn wdbx_open(path: [*c]const u8, create_if_missing: bool) callconv(.c) ?*anyopaque {
    const slice = std.mem.span(path);
    const db = Db.open(global_allocator, slice, create_if_missing) catch return null;
    return @ptrCast(db);
}

export fn wdbx_close(handle: ?*anyopaque) callconv(.c) void {
    const db = toDb(handle) orelse return;
    db.close();
}

export fn wdbx_init_db(handle: ?*anyopaque, dim: u16) callconv(.c) c_int {
    const db = toDb(handle) orelse return -1;
    db.init(dim) catch {
        return 1;
    };
    return 0;
}

export fn wdbx_add_embedding(handle: ?*anyopaque, embedding: [*c]const f32, len: usize, out_id: ?*u64) callconv(.c) c_int {
    const db = toDb(handle) orelse return -1;
    if (@intFromPtr(out_id) == 0) return -2;
    const slice: []const f32 = embedding[0..len];
    const id = db.addEmbedding(slice) catch {
        return 1;
    };
    out_id.?.* = id;
    return 0;
}

export fn wdbx_search_alloc(
    handle: ?*anyopaque,
    query: [*c]const f32,
    len: usize,
    top_k: usize,
    out_len: ?*usize,
) callconv(.c) ?[*]WdbxResult {
    const db = toDb(handle) orelse return null;

    const qslice: []const f32 = query[0..len];

    const allocator = global_allocator;
    const results = db.search(qslice, top_k, allocator) catch return null;
    defer allocator.free(results);

    const c_array = allocator.alloc(WdbxResult, results.len) catch return null;
    for (results, 0..) |r, i| {
        c_array[i] = .{ .index = r.index, .score = r.score };
    }
    if (@intFromPtr(out_len) != 0) out_len.?.* = c_array.len;
    return c_array.ptr;
}

export fn wdbx_free_results(ptr: [*c]WdbxResult, len: usize) callconv(.c) void {
    if (@intFromPtr(ptr) == 0 or len == 0) return;
    const allocator = global_allocator;
    allocator.free(ptr[0..len]);
}
