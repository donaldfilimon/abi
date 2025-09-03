const std = @import("std");

// Forward to the refactored implementation
pub fn main() !void {
    return @import("main_refactored.zig").main();
}
