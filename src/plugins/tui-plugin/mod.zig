const std = @import("std");

pub const name = "tui-plugin";
pub const description = "Example reference plugin targeting the feat-tui gate.";
pub const version = "0.1.0";
pub const target_feature = "tui";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, input, "__cmd__:")) {
        const rest = input["__cmd__:".len..];
        var iter = std.mem.splitScalar(u8, rest, '\n');
        const cmd = iter.next() orelse "";
        if (std.mem.eql(u8, cmd, "hello") or std.mem.eql(u8, cmd, "hi")) {
            return try allocator.dupe(u8, "tui-plugin: hello");
        }
        if (std.mem.eql(u8, cmd, "ping")) {
            return try allocator.dupe(u8, "tui-plugin: pong");
        }
        return try std.fmt.allocPrint(allocator, "tui-plugin: unknown command '{s}'", .{cmd});
    }
    return try std.fmt.allocPrint(allocator, "tui-plugin event (bytes={d})", .{input.len});
}

test {
    std.testing.refAllDecls(@This());
}

test "tui-plugin handles __cmd__ ping" {
    const out = try run(std.testing.allocator, "__cmd__:ping");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("tui-plugin: pong", out);
}
