const std = @import("std");
const types = @import("types.zig");

pub const max_splits = 8;

pub const SplitResult = struct {
    rects: [max_splits]types.Rect = @as([max_splits]types.Rect, @splat(.{})),
    len: usize = 0,

    pub fn slice(self: *const SplitResult) []const types.Rect {
        return self.rects[0..self.len];
    }
};

pub fn split(rect: types.Rect, direction: types.Direction, constraints: []const types.Constraint) SplitResult {
    _ = rect;
    _ = direction;
    _ = constraints;
    return .{};
}
