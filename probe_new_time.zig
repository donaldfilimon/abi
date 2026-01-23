const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    const ts = try std.Io.Clock.now(.real, io);
    std.debug.print("Real time (ms): {d}\n", .{ts.toMilliseconds()});

    const ts_awake = try std.Io.Clock.now(.awake, io);
    std.debug.print("Awake time (ns): {d}\n", .{ts_awake.toNanoseconds()});
}
