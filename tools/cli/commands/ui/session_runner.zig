//! Shared terminal session bootstrap for simple UI dashboards.

const std = @import("std");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");

pub const Options = struct {
    dashboard_name: []const u8,
    terminal_title: []const u8,
};

pub const Session = struct {
    terminal: tui.Terminal,

    pub fn deinit(self: *Session) void {
        self.terminal.exit() catch {};
        self.terminal.deinit();
    }
};

pub fn startSimpleDashboard(
    allocator: std.mem.Allocator,
    options: Options,
) ?Session {
    if (!tui.Terminal.isSupported()) {
        utils.output.printError("{s} requires a terminal.", .{options.dashboard_name});
        return null;
    }

    var terminal = tui.Terminal.init(allocator);
    terminal.enter() catch |err| {
        utils.output.printError("Failed to start {s}: {t}", .{ options.dashboard_name, err });
        terminal.deinit();
        return null;
    };
    terminal.setTitle(options.terminal_title) catch {};

    return .{
        .terminal = terminal,
    };
}

test {
    std.testing.refAllDecls(@This());
}
