const std = @import("std");
const types = @import("types.zig");

pub const Screen = struct {
    front: []types.Cell = &[_]types.Cell{},
    back: []types.Cell = &[_]types.Cell{},
    width: u16 = 0,
    height: u16 = 0,
    allocator: std.mem.Allocator = undefined,

    pub fn init(allocator: std.mem.Allocator, width: u16, height: u16) !Screen {
        _ = width;
        _ = height;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Screen) void {
        _ = self;
    }
    pub fn clear(self: *Screen) void {
        _ = self;
    }
    pub fn setCell(self: *Screen, x: u16, y: u16, cell: types.Cell) void {
        _ = self;
        _ = x;
        _ = y;
        _ = cell;
    }
    pub fn print(self: *Screen, x: u16, y: u16, text: []const u8, style: types.Style) void {
        _ = self;
        _ = x;
        _ = y;
        _ = text;
        _ = style;
    }
    pub fn flush(self: *Screen, writer: *std.Io.Writer) !void {
        _ = self;
        _ = writer;
    }
    pub fn resize(self: *Screen, width: u16, height: u16) !void {
        _ = self;
        _ = width;
        _ = height;
    }
    pub fn rect(self: *const Screen) types.Rect {
        _ = self;
        return .{};
    }
};
