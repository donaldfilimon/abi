//! Layout primitives for splitting screen regions.

const std = @import("std");
const types = @import("types.zig");
const Rect = types.Rect;
const Direction = types.Direction;
const Constraint = types.Constraint;

/// Split a rect into multiple regions based on constraints.
/// Returns up to 8 rects (bounded array to avoid allocation).
///
/// Two-pass algorithm:
///   Pass 1 — compute base sizes and track the last `.min` constraint index.
///   Pass 2 — build rects from the sizes array.
///   If there is a last `.min` and the base total is less than the available
///   space, the remainder is added to that last `.min` entry so it absorbs
///   any leftover pixels.
pub fn split(rect: Rect, direction: Direction, constraints: []const Constraint) SplitResult {
    var result = SplitResult{};
    if (constraints.len == 0) return result;

    const total: u16 = switch (direction) {
        .horizontal => rect.height,
        .vertical => rect.width,
    };

    const count = @min(constraints.len, max_splits);

    // --- Pass 1: compute base sizes, track last_min_index and base_total ---
    var sizes: [max_splits]u16 = [_]u16{0} ** max_splits;
    var base_total: u16 = 0;
    var last_min_index: ?usize = null;

    for (constraints[0..count], 0..) |c, i| {
        const size: u16 = switch (c) {
            .fixed => |s| @min(s, total -| base_total),
            .percentage => |pct| blk: {
                const raw = (@as(u32, total) * @as(u32, pct)) / 100;
                break :blk @intCast(@min(raw, total -| base_total));
            },
            .min => |s| inner: {
                last_min_index = i;
                break :inner @min(s, total -| base_total);
            },
        };
        sizes[i] = size;
        base_total += size;
    }

    // Give remainder to the last .min constraint if space is available
    if (last_min_index) |lmi| {
        if (base_total < total) {
            const remainder = total - base_total;
            sizes[lmi] += remainder;
        }
    }

    // --- Pass 2: build rects from sizes ---
    var offset: u16 = 0;
    for (0..count) |i| {
        result.rects[result.len] = switch (direction) {
            .horizontal => .{ .x = rect.x, .y = rect.y + offset, .width = rect.width, .height = sizes[i] },
            .vertical => .{ .x = rect.x + offset, .y = rect.y, .width = sizes[i], .height = rect.height },
        };
        result.len += 1;
        offset += sizes[i];
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

test "split mixed constraints with min absorbing remainder" {
    // fixed=5 + percentage=50 + min=10 on a 100-wide rect
    // base sizes: 5, 50, 10 = 65; remainder = 35 goes to last min
    const rect = Rect{ .x = 0, .y = 0, .width = 100, .height = 40 };
    const result = split(rect, .vertical, &.{
        .{ .fixed = 5 },
        .{ .percentage = 50 },
        .{ .min = 10 },
    });
    try std.testing.expectEqual(@as(usize, 3), result.len);
    const rects = result.slice();
    try std.testing.expectEqual(@as(u16, 5), rects[0].width);
    try std.testing.expectEqual(@as(u16, 50), rects[1].width);
    try std.testing.expectEqual(@as(u16, 45), rects[2].width); // 10 + 35 remainder
    // Offsets are contiguous
    try std.testing.expectEqual(@as(u16, 0), rects[0].x);
    try std.testing.expectEqual(@as(u16, 5), rects[1].x);
    try std.testing.expectEqual(@as(u16, 55), rects[2].x);
}

test "split oversized constraints clamp to available space" {
    const rect = Rect{ .x = 0, .y = 0, .width = 20, .height = 10 };
    const result = split(rect, .vertical, &.{
        .{ .fixed = 15 },
        .{ .fixed = 15 },
    });
    try std.testing.expectEqual(@as(usize, 2), result.len);
    try std.testing.expectEqual(@as(u16, 15), result.rects[0].width);
    try std.testing.expectEqual(@as(u16, 5), result.rects[1].width); // clamped
}

test "split max_splits truncation" {
    const rect = Rect{ .x = 0, .y = 0, .width = 100, .height = 10 };
    // 10 constraints but max_splits is 8
    const constraints = [_]Constraint{.{ .fixed = 5 }} ** 10;
    const result = split(rect, .vertical, &constraints);
    try std.testing.expectEqual(@as(usize, 8), result.len);
}

test {
    std.testing.refAllDecls(@This());
}
