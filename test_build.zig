const std = @import("std");
const opts = @import("src/build_options.zig");

pub fn main() void {
    std.debug.print("GPU enabled: {}\n", .{opts.enable_gpu});
}
