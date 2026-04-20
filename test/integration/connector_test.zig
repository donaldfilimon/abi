const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Mock environment
    try std.process.setEnvVar("OPENAI_API_KEY", "sk-1234567890abcdef1234567890abcdef");

    const connector = try abi.connectors.loaders.tryLoadOpenAI(allocator);
    if (connector) |config| {
        std.debug.print("Successfully loaded config: {s}\n", .{config.base_url});
        config.deinit(allocator);
    } else {
        std.debug.print("Failed to load config (missing API key?)\n", .{});
    }
}
