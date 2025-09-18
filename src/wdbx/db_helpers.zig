const std = @import("std");
const json = std.json;
const engine = @import("database");

pub const Db = engine.Db;
pub const DatabaseError = engine.DatabaseError;
pub const WdbxHeader = engine.WdbxHeader;
pub const Result = engine.Db.Result;
pub const DbStats = engine.Db.DbStats;

/// Shared helpers for manipulating vectors and database-oriented payloads.
pub const helpers = struct {
    pub fn parseVector(allocator: std.mem.Allocator, input: []const u8) ![]f32 {
        var parts = std.mem.splitScalar(u8, input, ',');
        var values = try std.ArrayList(f32).initCapacity(allocator, 0);
        errdefer values.deinit(allocator);

        while (parts.next()) |segment| {
            const trimmed = std.mem.trim(u8, segment, " \t\r\n");
            if (trimmed.len == 0) continue;
            const parsed = try std.fmt.parseFloat(f32, trimmed);
            try values.append(allocator, parsed);
        }

        return try values.toOwnedSlice(allocator);
    }

    pub fn parseJsonVector(allocator: std.mem.Allocator, node: json.Value) ![]f32 {
        return switch (node) {
            .array => |arr| parseJsonArray(allocator, arr.items),
            else => error.InvalidVectorPayload,
        };
    }

    fn parseJsonArray(allocator: std.mem.Allocator, items: []const json.Value) ![]f32 {
        var values = try std.ArrayList(f32).initCapacity(allocator, 0);
        errdefer values.deinit(allocator);

        for (items) |item| {
            const parsed = switch (item) {
                .float => |f| std.math.cast(f32, f) orelse return error.InvalidVectorPayload,
                .integer => |i| @as(f32, @floatFromInt(i)),
                .number_string => |s| try std.fmt.parseFloat(f32, s),
                else => return error.InvalidVectorPayload,
            };
            try values.append(allocator, parsed);
        }

        return try values.toOwnedSlice(allocator);
    }
};
