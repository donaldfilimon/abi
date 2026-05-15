const std = @import("std");

pub const Status = enum { ready, busy, error_state };
pub const Item = struct { label: []const u8, value: []const u8 };
pub const State = struct { title: []const u8, status: Status, items: []const Item };

pub const ScreenState = struct {
    width: u16,
    height: u16,
};

pub fn initScreen() !void {}
pub fn clearScreen() !void {}
pub fn render(state: ScreenState) !void {
    _ = state;
}
pub fn deinitScreen() void {}

pub fn renderDashboard(allocator: std.mem.Allocator, state: State) ![]u8 {
    _ = allocator;
    _ = state;
    return "";
}

pub fn statusText(status: Status) []const u8 {
    _ = status;
    return "";
}
