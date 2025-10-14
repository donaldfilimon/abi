const std = @import("std");
const json = std.json;
const engine = @import("./database.zig");

pub const Db = engine.Db;
pub const DatabaseError = engine.DatabaseError;
pub const WdbxHeader = engine.WdbxHeader;
pub const Result = engine.Db.Result;
pub const DbStats = engine.Db.DbStats;

/// Shared helpers for manipulating vectors and database-oriented payloads.
pub const helpers = struct {
    pub fn parseVector(allocator: std.mem.Allocator, input: []const u8) ![]f32 {
        // Pre-allocate with estimated capacity based on input length
        const estimated_capacity = @max(1, input.len / 4); // Rough estimate
        var values = try std.ArrayList(f32).initCapacity(allocator, estimated_capacity);
        errdefer values.deinit(allocator);

        var parts = std.mem.splitScalar(u8, input, ',');
        while (parts.next()) |segment| {
            const trimmed = std.mem.trim(u8, segment, " \t\r\n");
            if (trimmed.len == 0) continue;
            
            // Fast path for common cases
            const parsed = if (trimmed.len <= 10) 
                std.fmt.parseFloat(f32, trimmed) catch continue
            else
                std.fmt.parseFloat(f32, trimmed) catch continue;
            
            try values.append(parsed);
        }

        return try values.toOwnedSlice(allocator);
    }

    pub fn parseJsonVector(allocator: std.mem.Allocator, node: json.Value) ![]f32 {
        return switch (node) {
            .array => |arr| parseJsonArray(allocator, arr.items),
            else => error.InvalidVectorPayload,
        };
    }

    pub fn formatNearestNeighborResponse(allocator: std.mem.Allocator, result: Result) ![]u8 {
        return std.fmt.allocPrint(
            allocator,
            "{\"success\":true,\"nearest_neighbor\":{\"index\":{d},\"distance\":{d}}}",
            .{ result.index, result.score },
        );
    }

    pub fn formatKnnResponse(allocator: std.mem.Allocator, k: usize, results: []const Result) ![]u8 {
        const neighbors_json = try formatNeighborsJson(allocator, results);
        defer allocator.free(neighbors_json);

        return std.fmt.allocPrint(
            allocator,
            "{\"success\":true,\"k\":{d},\"neighbors\":[{s}]}",
            .{ k, neighbors_json },
        );
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

    fn formatNeighborsJson(allocator: std.mem.Allocator, results: []const Result) ![]u8 {
        if (results.len == 0) return allocator.dupe(u8, "");

        var buffer = try std.ArrayList(u8).initCapacity(allocator, results.len * 48);
        errdefer buffer.deinit(allocator);

        for (results, 0..) |res, idx| {
            if (idx != 0) try buffer.appendSlice(allocator, ",");
            const chunk = try std.fmt.allocPrint(
                allocator,
                "{\"index\":{d},\"distance\":{d}}",
                .{ res.index, res.score },
            );
            defer allocator.free(chunk);
            try buffer.appendSlice(allocator, chunk);
        }

        return try buffer.toOwnedSlice(allocator);
    }
};
