//! Layout primitives for splitting screen regions.

const std = @import("std");
const types = @import("types.zig");
const Rect = types.Rect;
const Direction = types.Direction;
const Constraint = types.Constraint;

/// Split a rect into multiple regions based on constraints.
/// Returns up to 8 rects (bounded array to avoid allocation).
pub fn split(rect: Rect, direction: Direction, constraints: []const Constraint) SplitResult {
    var result = SplitResult{};
    if (constraints.len == 0) return result;

    const total: u16 = switch (direction) {
        .horizontal => rect.height,
        .vertical => rect.width,
    };

    // First pass: calculate fixed sizes
    var remaining = total;
    var percentage_count: u16 = 0;
    for (constraints) |c| {
        switch (c) {
            .fixed => |size| remaining -|= @min(size, remaining),
            .percentage => percentage_count += 1,
            .min => |size| remaining -|= @min(size, remaining),
        }
    }

    // Second pass: assign sizes
    var offset: u16 = 0;
    for (constraints) |c| {
        if (result.len >= max_splits) break;

        const size: u16 = switch (c) {
            .fixed => |s| @min(s, total -| offset),
            .percentage => |pct| blk: {
                const raw = (@as(u32, total) * @as(u32, pct)) / 100;
                break :blk @intCast(@min(raw, total -| offset));
            },
            .min => |s| @min(s, total -| offset),
        };

        result.rects[result.len] = switch (direction) {
            .horizontal => .{ .x = rect.x, .y = rect.y + offset, .width = rect.width, .height = size },
            .vertical => .{ .x = rect.x + offset, .y = rect.y, .width = size, .height = rect.height },
        };
        result.len += 1;
        offset += size;
    }

    return result;
}

pub const max_splits = 8;

pub const SplitResult = struct {
    rects: [max_splits]Rect = [_]Rect{.{}} ** max_splits,
    len: usize = 0,

    pub fn slice(self: *const SplitResult) []const Rect {
        return self.rects[0..self.len];
    }
};

test "split horizontal fixed" {
    const rect = Rect{ .x = 0, .y = 0, .width = 80, .height = 24 };
    const result = split(rect, .horizontal, &.{ .{ .fixed = 3 }, .{ .fixed = 18 }, .{ .fixed = 3 } });
    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqual(@as(u16, 3), result.rects[0].height);
    try std.testing.expectEqual(@as(u16, 18), result.rects[1].height);
    try std.testing.expectEqual(@as(u16, 3), result.rects[2].height);
    try std.testing.expectEqual(@as(u16, 3), result.rects[1].y);
}

test "split vertical percentage" {
    const rect = Rect{ .x = 0, .y = 0, .width = 100, .height = 24 };
    const result = split(rect, .vertical, &.{ .{ .percentage = 30 }, .{ .percentage = 70 } });
    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqual(@as(u16, 30), result.rects[0].width);
    try std.testing.expectEqual(@as(u16, 70), result.rects[1].width);
}

test "split empty constraints" {
    const rect = Rect{ .x = 0, .y = 0, .width = 80, .height = 24 };
    const result = split(rect, .horizontal, &.{});
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test {
    std.testing.refAllDecls(@This());
}
