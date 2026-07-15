const std = @import("std");

pub const name = "example-plugin";
pub const description = "Minimal example plugin used by registry generation tests.";
pub const version = "0.1.0";
pub const target_feature = "plugins";

pub fn register() void {}

/// Future execution entry point (currently called via the manager's run hook).
/// Handles `__context__:<name>` inputs for context providers and `__cmd__:<name>`
/// (optional `\n`-payload) for slash-commands declared in the plugin manifest.
pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, input, "__context__:")) {
        const provider = input["__context__:".len..];
        if (std.mem.eql(u8, provider, "example-info")) {
            return try allocator.dupe(u8, "example-plugin v0.1.0 (target: plugins)");
        }
        return try std.fmt.allocPrint(allocator, "unknown context provider: {s}", .{provider});
    }
    if (std.mem.startsWith(u8, input, "__cmd__:")) {
        const rest = input["__cmd__:".len..];
        var iter = std.mem.splitScalar(u8, rest, '\n');
        const cmd = iter.next() orelse "";
        const payload = iter.rest();
        if (std.mem.eql(u8, cmd, "echo") or std.mem.eql(u8, cmd, "say")) {
            if (payload.len == 0) return try allocator.dupe(u8, "example-plugin echo: (empty)");
            return try std.fmt.allocPrint(allocator, "example-plugin echo: {s}", .{payload});
        }
        return try std.fmt.allocPrint(allocator, "example-plugin: unknown command '{s}'", .{cmd});
    }
    return try std.fmt.allocPrint(allocator, "example-plugin received input (len={d})", .{input.len});
}

test {
    std.testing.refAllDecls(@This());
}

test "example-plugin handles __cmd__ echo" {
    const out = try run(std.testing.allocator, "__cmd__:echo\nhello");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("example-plugin echo: hello", out);
}
