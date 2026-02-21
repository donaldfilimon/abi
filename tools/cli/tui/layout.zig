//! Layout engine for TUI components.
//!
//! Provides Rect (rectangular screen areas) and Constraint-based
//! layout distribution for building responsive terminal UIs.

const std = @import("std");

/// A rectangular area on the terminal screen.
pub const Rect = struct {
    x: u16,
    y: u16,
    width: u16,
    height: u16,

    /// Create a Rect from terminal size (full screen).
    pub fn fromTerminalSize(cols: u16, rows: u16) Rect {
        return .{ .x = 0, .y = 0, .width = cols, .height = rows };
    }

    /// Shrink the rect by the given margins.
    /// Uses saturating subtraction to prevent underflow.
    pub fn shrink(
        self: Rect,
        top: u16,
        right: u16,
        bottom: u16,
        left: u16,
    ) Rect {
        const h_margin = @as(u16, @intCast(
            @min(@as(u32, left) + @as(u32, right), self.width),
        ));
        const v_margin = @as(u16, @intCast(
            @min(@as(u32, top) + @as(u32, bottom), self.height),
        ));
        return .{
            .x = self.x +| left,
            .y = self.y +| top,
            .width = self.width -| h_margin,
            .height = self.height -| v_margin,
        };
    }

    /// Split horizontally at row offset `at` (from top).
    /// Returns .top (height=at) and .bottom (remaining).
    pub fn splitHorizontal(
        self: Rect,
        at: u16,
    ) struct { top: Rect, bottom: Rect } {
        const clamped = @min(at, self.height);
        return .{
            .top = .{
                .x = self.x,
                .y = self.y,
                .width = self.width,
                .height = clamped,
            },
            .bottom = .{
                .x = self.x,
                .y = self.y +| clamped,
                .width = self.width,
                .height = self.height -| clamped,
            },
        };
    }

    /// Split vertically at column offset `at` (from left).
    /// Returns .left (width=at) and .right (remaining).
    pub fn splitVertical(
        self: Rect,
        at: u16,
    ) struct { left: Rect, right: Rect } {
        const clamped = @min(at, self.width);
        return .{
            .left = .{
                .x = self.x,
                .y = self.y,
                .width = clamped,
                .height = self.height,
            },
            .right = .{
                .x = self.x +| clamped,
                .y = self.y,
                .width = self.width -| clamped,
                .height = self.height,
            },
        };
    }

    /// Return inner rect (1-cell border removed on all sides).
    pub fn inner(self: Rect) Rect {
        return self.shrink(1, 1, 1, 1);
    }

    /// Check if rect has zero area.
    pub fn isEmpty(self: Rect) bool {
        return self.width == 0 or self.height == 0;
    }

    /// Clamp a value to fit within the rect's width.
    pub fn clampWidth(self: Rect, value: u16) u16 {
        return @min(value, self.width);
    }

    /// Clamp a value to fit within the rect's height.
    pub fn clampHeight(self: Rect, value: u16) u16 {
        return @min(value, self.height);
    }
};

/// Layout constraint for a single element.
pub const Constraint = union(enum) {
    /// Exact fixed size.
    fixed: u16,
    /// At least this size, expand if space available.
    min: u16,
    /// At most this size, shrink if needed.
    max: u16,
    /// Percentage of parent (0-100).
    percentage: u8,
    /// Take all remaining space after other constraints resolved.
    fill: void,
};

/// Distribute `available` space among `constraints`.
/// Results are written into the caller-provided `result` slice,
/// which must be the same length as `constraints`.
///
/// Algorithm:
///   1. Resolve fixed and percentage constraints first.
///   2. Distribute remaining space to fill constraints equally.
///   3. Apply min/max bounds.
///   4. If total exceeds available, proportionally shrink.
pub fn distribute(
    available: u16,
    constraints: []const Constraint,
    result: []u16,
) void {
    const len = constraints.len;
    if (len == 0) return;
    std.debug.assert(result.len == len);

    if (available == 0) {
        @memset(result, 0);
        return;
    }

    // Pass 1: resolve fixed and percentage; track fill count
    // and remaining space.
    var used: u32 = 0;
    var fill_count: u32 = 0;
    for (constraints, 0..) |c, i| {
        switch (c) {
            .fixed => |v| {
                result[i] = v;
                used += v;
            },
            .percentage => |pct| {
                const p = @as(u32, pct);
                const v: u16 = @intCast(
                    (@as(u32, available) * p) / 100,
                );
                result[i] = v;
                used += v;
            },
            .min => |v| {
                // Start with the minimum; may grow later.
                result[i] = v;
                used += v;
            },
            .max => |v| {
                // Start with the max bound; may shrink later.
                result[i] = v;
                used += v;
            },
            .fill => {
                result[i] = 0;
                fill_count += 1;
            },
        }
    }

    // Pass 2: distribute remaining space to fill constraints.
    if (fill_count > 0) {
        const remaining = @as(u32, available) -|
            @min(used, @as(u32, available));
        const per_fill: u16 = @intCast(remaining / fill_count);
        var extra: u16 = @intCast(remaining % fill_count);

        for (constraints, 0..) |c, i| {
            if (c == .fill) {
                result[i] = per_fill;
                if (extra > 0) {
                    result[i] += 1;
                    extra -= 1;
                }
            }
        }
        // Recalculate used after fill distribution.
        used = 0;
        for (result[0..len]) |v| {
            used += v;
        }
    }

    // Pass 3: expand min constraints if space remains.
    if (used < @as(u32, available)) {
        var expandable: u32 = 0;
        for (constraints) |c| {
            if (c == .min) expandable += 1;
        }
        if (expandable > 0) {
            const remaining = @as(u32, available) - used;
            const per_expand: u16 = @intCast(
                remaining / expandable,
            );
            var extra: u16 = @intCast(remaining % expandable);
            for (constraints, 0..) |c, i| {
                if (c == .min) {
                    result[i] += per_expand;
                    if (extra > 0) {
                        result[i] += 1;
                        extra -= 1;
                    }
                }
            }
            // Recalculate used.
            used = 0;
            for (result[0..len]) |v| {
                used += v;
            }
        }
    }

    // Pass 4: if total exceeds available, proportionally shrink.
    if (used > @as(u32, available)) {
        const avail32 = @as(u32, available);
        // Proportional shrink: new_i = result[i] * available / used
        var assigned: u32 = 0;
        for (result[0..len], 0..) |v, i| {
            if (i == len - 1) {
                // Last element gets whatever is left to avoid
                // rounding drift.
                result[i] = @intCast(avail32 - assigned);
            } else {
                const scaled: u16 = @intCast(
                    (@as(u32, v) * avail32) / used,
                );
                result[i] = scaled;
                assigned += scaled;
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

test "Rect.fromTerminalSize" {
    const r = Rect.fromTerminalSize(80, 24);
    try std.testing.expectEqual(@as(u16, 0), r.x);
    try std.testing.expectEqual(@as(u16, 0), r.y);
    try std.testing.expectEqual(@as(u16, 80), r.width);
    try std.testing.expectEqual(@as(u16, 24), r.height);
}

test "Rect.shrink normal" {
    const r = Rect.fromTerminalSize(80, 24).shrink(2, 3, 4, 5);
    try std.testing.expectEqual(@as(u16, 5), r.x);
    try std.testing.expectEqual(@as(u16, 2), r.y);
    try std.testing.expectEqual(@as(u16, 72), r.width); // 80 - 3 - 5
    try std.testing.expectEqual(@as(u16, 18), r.height); // 24 - 2 - 4
}

test "Rect.shrink saturating" {
    const r = Rect.fromTerminalSize(10, 10).shrink(100, 100, 100, 100);
    try std.testing.expectEqual(@as(u16, 0), r.width);
    try std.testing.expectEqual(@as(u16, 0), r.height);
}

test "Rect.splitHorizontal normal" {
    const r = Rect.fromTerminalSize(80, 24);
    const s = r.splitHorizontal(10);
    try std.testing.expectEqual(@as(u16, 10), s.top.height);
    try std.testing.expectEqual(@as(u16, 14), s.bottom.height);
    try std.testing.expectEqual(@as(u16, 10), s.bottom.y);
}

test "Rect.splitHorizontal at=0" {
    const s = Rect.fromTerminalSize(80, 24).splitHorizontal(0);
    try std.testing.expectEqual(@as(u16, 0), s.top.height);
    try std.testing.expectEqual(@as(u16, 24), s.bottom.height);
}

test "Rect.splitHorizontal at>=height" {
    const s = Rect.fromTerminalSize(80, 24).splitHorizontal(100);
    try std.testing.expectEqual(@as(u16, 24), s.top.height);
    try std.testing.expectEqual(@as(u16, 0), s.bottom.height);
}

test "Rect.splitVertical normal" {
    const r = Rect.fromTerminalSize(80, 24);
    const s = r.splitVertical(30);
    try std.testing.expectEqual(@as(u16, 30), s.left.width);
    try std.testing.expectEqual(@as(u16, 50), s.right.width);
    try std.testing.expectEqual(@as(u16, 30), s.right.x);
}

test "Rect.splitVertical at=0" {
    const s = Rect.fromTerminalSize(80, 24).splitVertical(0);
    try std.testing.expectEqual(@as(u16, 0), s.left.width);
    try std.testing.expectEqual(@as(u16, 80), s.right.width);
}

test "Rect.splitVertical at>=width" {
    const s = Rect.fromTerminalSize(80, 24).splitVertical(200);
    try std.testing.expectEqual(@as(u16, 80), s.left.width);
    try std.testing.expectEqual(@as(u16, 0), s.right.width);
}

test "Rect.inner normal" {
    const r = Rect.fromTerminalSize(80, 24).inner();
    try std.testing.expectEqual(@as(u16, 1), r.x);
    try std.testing.expectEqual(@as(u16, 1), r.y);
    try std.testing.expectEqual(@as(u16, 78), r.width);
    try std.testing.expectEqual(@as(u16, 22), r.height);
}

test "Rect.inner tiny 1x1" {
    const r = (Rect{ .x = 0, .y = 0, .width = 1, .height = 1 }).inner();
    try std.testing.expect(r.isEmpty());
}

test "Rect.isEmpty" {
    try std.testing.expect(
        (Rect{ .x = 0, .y = 0, .width = 0, .height = 10 }).isEmpty(),
    );
    try std.testing.expect(
        (Rect{ .x = 0, .y = 0, .width = 10, .height = 0 }).isEmpty(),
    );
    try std.testing.expect(
        !(Rect{ .x = 0, .y = 0, .width = 10, .height = 10 }).isEmpty(),
    );
}

test "distribute all-fixed" {
    var result: [3]u16 = undefined;
    distribute(100, &.{
        .{ .fixed = 20 },
        .{ .fixed = 30 },
        .{ .fixed = 50 },
    }, &result);
    try std.testing.expectEqual(@as(u16, 20), result[0]);
    try std.testing.expectEqual(@as(u16, 30), result[1]);
    try std.testing.expectEqual(@as(u16, 50), result[2]);
}

test "distribute all-fill" {
    var result: [3]u16 = undefined;
    distribute(90, &.{
        .{ .fill = {} },
        .{ .fill = {} },
        .{ .fill = {} },
    }, &result);
    try std.testing.expectEqual(@as(u16, 30), result[0]);
    try std.testing.expectEqual(@as(u16, 30), result[1]);
    try std.testing.expectEqual(@as(u16, 30), result[2]);
}

test "distribute all-fill uneven" {
    var result: [3]u16 = undefined;
    distribute(10, &.{
        .{ .fill = {} },
        .{ .fill = {} },
        .{ .fill = {} },
    }, &result);
    // 10 / 3 = 3 remainder 1 → first gets 4
    try std.testing.expectEqual(@as(u16, 4), result[0]);
    try std.testing.expectEqual(@as(u16, 3), result[1]);
    try std.testing.expectEqual(@as(u16, 3), result[2]);
}

test "distribute percentage" {
    var result: [2]u16 = undefined;
    distribute(200, &.{
        .{ .percentage = 25 },
        .{ .percentage = 75 },
    }, &result);
    try std.testing.expectEqual(@as(u16, 50), result[0]);
    try std.testing.expectEqual(@as(u16, 150), result[1]);
}

test "distribute mixed fixed+fill" {
    var result: [3]u16 = undefined;
    distribute(100, &.{
        .{ .fixed = 20 },
        .{ .fill = {} },
        .{ .fixed = 30 },
    }, &result);
    try std.testing.expectEqual(@as(u16, 20), result[0]);
    try std.testing.expectEqual(@as(u16, 50), result[1]);
    try std.testing.expectEqual(@as(u16, 30), result[2]);
}

test "distribute mixed fixed+min+fill" {
    var result: [3]u16 = undefined;
    distribute(100, &.{
        .{ .fixed = 20 },
        .{ .min = 10 },
        .{ .fill = {} },
    }, &result);
    // fixed=20, min starts at 10, fill starts at 0.
    // remaining after fixed+min = 70. fill gets 70.
    // Then min can expand: used=20+10+70=100, no room.
    try std.testing.expectEqual(@as(u16, 20), result[0]);
    try std.testing.expectEqual(@as(u16, 10), result[1]);
    try std.testing.expectEqual(@as(u16, 70), result[2]);
}

test "distribute zero available" {
    var result: [3]u16 = undefined;
    distribute(0, &.{
        .{ .fixed = 20 },
        .{ .fill = {} },
        .{ .percentage = 50 },
    }, &result);
    try std.testing.expectEqual(@as(u16, 0), result[0]);
    try std.testing.expectEqual(@as(u16, 0), result[1]);
    try std.testing.expectEqual(@as(u16, 0), result[2]);
}

test "distribute single constraint" {
    var result: [1]u16 = undefined;
    distribute(80, &.{.{ .fill = {} }}, &result);
    try std.testing.expectEqual(@as(u16, 80), result[0]);
}

test "distribute overflow shrinks proportionally" {
    var result: [2]u16 = undefined;
    distribute(50, &.{
        .{ .fixed = 60 },
        .{ .fixed = 40 },
    }, &result);
    // Total 100 into 50 → proportional: 30, 20
    try std.testing.expectEqual(@as(u16, 30), result[0]);
    try std.testing.expectEqual(@as(u16, 20), result[1]);
}
