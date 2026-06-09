const std = @import("std");
const wdbx_mod = @import("mod.zig");

pub fn applyLine(allocator: std.mem.Allocator, store: *wdbx_mod.Store, line: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, line, .{});
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return error.UnknownLineType,
    };

    const type_str = obj.get("type") orelse return error.MissingField;
    const type_slice = switch (type_str) {
        .string => |s| s,
        else => return error.UnknownLineType,
    };

    if (std.mem.eql(u8, type_slice, "kv")) {
        try applyKv(store, obj);
    } else if (std.mem.eql(u8, type_slice, "vector")) {
        try applyVector(store, obj);
    } else if (std.mem.eql(u8, type_slice, "block")) {
        try applyBlock(store, obj);
    } else if (std.mem.eql(u8, type_slice, "spatial")) {
        try applySpatial(store, obj);
    } else {
        return error.UnknownLineType;
    }
}

fn applyKv(store: *wdbx_mod.Store, obj: std.json.ObjectMap) !void {
    const key = obj.get("key") orelse return error.MissingField;
    const val = obj.get("value") orelse return error.MissingField;
    const key_s = switch (key) {
        .string => |s| s,
        else => return error.MissingField,
    };
    const val_s = switch (val) {
        .string => |s| s,
        else => return error.MissingField,
    };
    try store.store(key_s, val_s);
}

fn applyVector(store: *wdbx_mod.Store, obj: std.json.ObjectMap) !void {
    const id_node = obj.get("id") orelse return error.MissingField;
    const values_node = obj.get("values") orelse return error.MissingField;
    const expected_id = try jsonU32(id_node);
    const values_arr = switch (values_node) {
        .array => |a| a,
        else => return error.MissingField,
    };
    if (values_arr.items.len == 0) return error.DimensionMismatch;
    if (values_arr.items.len > wdbx_mod.HNSW_DIMENSIONS) return error.DimensionMismatch;
    var values: [wdbx_mod.HNSW_DIMENSIONS]f32 = undefined;
    var n: usize = 0;
    for (values_arr.items) |v| {
        values[n] = jsonNumberAsF32(v) orelse return error.MissingField;
        n += 1;
    }
    const assigned = try store.putVector(values[0..n]);
    if (assigned != expected_id) return error.CorruptVectorId;
}

fn applyBlock(store: *wdbx_mod.Store, obj: std.json.ObjectMap) !void {
    const profile_node = obj.get("profile") orelse return error.MissingField;
    const query_id_node = obj.get("query_id") orelse return error.MissingField;
    const response_id_node = obj.get("response_id") orelse return error.MissingField;
    const metadata_node = obj.get("metadata") orelse return error.MissingField;
    const timestamp_node = obj.get("timestamp_ms") orelse return error.MissingField;
    const profile_s = switch (profile_node) {
        .string => |s| s,
        else => return error.MissingField,
    };
    const query_id = try jsonU32(query_id_node);
    const response_id = try jsonU32(response_id_node);
    const metadata_s = switch (metadata_node) {
        .string => |s| s,
        else => return error.MissingField,
    };
    const timestamp_ms = switch (timestamp_node) {
        .integer => |i| @as(i64, @intCast(i)),
        else => return error.MissingField,
    };
    _ = try store.restoreBlock(profile_s, query_id, response_id, metadata_s, timestamp_ms);
}

fn applySpatial(store: *wdbx_mod.Store, obj: std.json.ObjectMap) !void {
    const id_node = obj.get("id") orelse return error.MissingField;
    const x_node = obj.get("x") orelse return error.MissingField;
    const y_node = obj.get("y") orelse return error.MissingField;
    const z_node = obj.get("z") orelse return error.MissingField;
    const payload_node = obj.get("payload") orelse return error.MissingField;
    const id = try jsonU32(id_node);
    const x = jsonNumberAsF32(x_node) orelse return error.MissingField;
    const y = jsonNumberAsF32(y_node) orelse return error.MissingField;
    const z = jsonNumberAsF32(z_node) orelse return error.MissingField;
    const payload_s = switch (payload_node) {
        .string => |s| s,
        else => return error.MissingField,
    };
    try store.putSpatial3D(id, .{ .x = x, .y = y, .z = z }, payload_s);
}

/// Extract a `u32` from an untrusted JSON value. Corrupt snapshots fail cleanly
/// instead of panicking on integer casts.
fn jsonU32(v: std.json.Value) error{ MissingField, FieldOutOfRange }!u32 {
    const i = switch (v) {
        .integer => |x| x,
        else => return error.MissingField,
    };
    return std.math.cast(u32, i) orelse error.FieldOutOfRange;
}

fn jsonNumberAsF32(v: std.json.Value) ?f32 {
    return switch (v) {
        .float => |f| @as(f32, @floatCast(f)),
        .integer => |i| @as(f32, @floatFromInt(i)),
        else => null,
    };
}

test {
    std.testing.refAllDecls(@This());
}
