const std = @import("std");

comptime {
    std.log.warn("ml/neural.zig has been removed. Import features/ai/neural.zig instead.", .{});
    @compileError("ml/neural.zig is deprecated. Use features/ai/neural.zig.");
}
