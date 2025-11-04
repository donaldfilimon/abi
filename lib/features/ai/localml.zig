const std = @import("std");

comptime {
    std.log.warn("ml/localml.zig has been removed. Import features/ai/localml.zig instead.", .{});
    @compileError("ml/localml.zig is deprecated. Use features/ai/localml.zig.");
}
