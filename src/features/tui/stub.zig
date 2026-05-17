const std = @import("std");

pub const Status = enum { ready, busy, warning, disabled };
pub const Item = struct { label: []const u8, value: []const u8 };
pub const State = struct { title: []const u8, status: Status = .disabled, items: []const Item = &.{} };

pub const ScreenState = struct {
    width: u16,
    height: u16,
};

pub fn initScreen() !void {}
pub fn initScreenWriter(writer: anytype) !void {
    _ = writer;
}
pub fn clearScreen() !void {}
pub fn clearScreenWriter(writer: anytype) !void {
    _ = writer;
}
pub fn render(state: ScreenState) !void {
    _ = state;
}
pub fn renderWriter(writer: anytype, state: ScreenState) !void {
    _ = writer;
    _ = state;
}
pub fn deinitScreen() void {}
pub fn deinitScreenWriter(writer: anytype) !void {
    _ = writer;
}

pub fn renderDashboard(allocator: std.mem.Allocator, state: State) ![]u8 {
    _ = state;
    return try allocator.dupe(u8, "TUI feature is disabled");
}

pub fn statusText(status: Status) []const u8 {
    return switch (status) {
        .ready => "ready",
        .busy => "busy",
        .warning => "warning",
        .disabled => "disabled",
    };
}
