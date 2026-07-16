const std = @import("std");

pub const name = "ai-plugin";
pub const description = "Example reference plugin targeting the feat-ai gate.";
pub const version = "0.1.0";
pub const target_feature = "ai";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, input, "__cmd__:")) {
        const rest = input["__cmd__:".len..];
        var iter = std.mem.splitScalar(u8, rest, '\n');
        const cmd = iter.next() orelse "";
        if (std.mem.eql(u8, cmd, "profile") or std.mem.eql(u8, cmd, "ai-status")) {
            return try allocator.dupe(u8, "ai-plugin: profile status (reference plugin — see `abi backends`)");
        }
        if (std.mem.eql(u8, cmd, "models")) {
            return try allocator.dupe(u8, "ai-plugin: model list (reference plugin — see `abi complete --help`)");
        }
        return try std.fmt.allocPrint(allocator, "ai-plugin: unknown command '{s}'", .{cmd});
    }
    return try std.fmt.allocPrint(allocator, "ai-plugin event (bytes={d})", .{input.len});
}

test {
    std.testing.refAllDecls(@This());
}
