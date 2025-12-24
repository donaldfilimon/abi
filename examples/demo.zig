const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 Starting ABI Framework Demo", .{});

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    std.log.info("✅ Framework initialized successfully!", .{});
    std.log.info("🎉 ABI Framework is ready for production use!", .{});

    // Check available features at compile time
    if (comptime @hasDecl(abi, "gpu")) {
        std.log.info("🎮 GPU acceleration available", .{});
    }
    if (comptime @hasDecl(abi, "web")) {
        std.log.info("🌐 Web server available", .{});
    }
    if (comptime @hasDecl(abi, "database")) {
        std.log.info("🗄️ Vector database available", .{});
    }
    if (comptime @hasDecl(abi, "ai")) {
        std.log.info("🤖 AI/ML features available", .{});
    }
}
