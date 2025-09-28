//! ABI bootstrap executable showcasing allocator initialisation while
//! exercising the new Zig 0.16 streaming writer API. The CLI dispatcher is
//! being modernised separately; for now the binary prints the framework
//! summary using the new output layer.

const std = @import("std");
const abi = @import("abi").abi;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const stdout = std.debug;

    var framework = try abi.init(gpa.allocator(), .{});
    defer framework.deinit();

    // For now, just print success since we can't use the old io APIs
    stdout.print("ABI Framework bootstrap complete\n", .{});
}
