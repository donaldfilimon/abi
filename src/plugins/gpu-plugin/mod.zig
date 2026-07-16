const std = @import("std");

pub const name = "gpu-plugin";
pub const description = "Example reference plugin targeting the feat-gpu gate.";
pub const version = "0.1.0";
pub const target_feature = "gpu";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, input, "__cmd__:")) {
        const rest = input["__cmd__:".len..];
        var iter = std.mem.splitScalar(u8, rest, '\n');
        const cmd = iter.next() orelse "";
        if (std.mem.eql(u8, cmd, "gpu-status") or std.mem.eql(u8, cmd, "gpu")) {
            return try allocator.dupe(u8, "gpu-plugin: status (reference plugin — see `abi backends` / dashboard System pane)");
        }
        if (std.mem.eql(u8, cmd, "gpu-info")) {
            return try allocator.dupe(u8, "gpu-plugin: info (reference plugin — Metal link vs accelerated is runtime state)");
        }
        return try std.fmt.allocPrint(allocator, "gpu-plugin: unknown command '{s}'", .{cmd});
    }
    return try std.fmt.allocPrint(allocator, "gpu-plugin event (bytes={d})", .{input.len});
}

test {
    std.testing.refAllDecls(@This());
}
