const std = @import("std");
const render_mod = @import("../stub_render.zig");

pub fn renderDashboard(screen: *render_mod.Screen) void {
    _ = screen;
}

pub fn run(allocator: std.mem.Allocator) !void {
    _ = allocator;
}
