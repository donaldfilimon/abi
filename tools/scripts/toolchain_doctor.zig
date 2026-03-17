const std = @import("std");
const toolchain = @import("toolchain_support.zig");

pub fn main(init: std.process.Init) !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = init.minimal.environ });
    defer io_backend.deinit();
    const io = io_backend.io();

    const issues = try toolchain.printDoctorReport(allocator, io);
    if (issues > 0) {
        std.process.exit(1);
    }
}
