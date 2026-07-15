const std = @import("std");

pub const name = "nn-plugin";
pub const description = "Example reference plugin targeting the feat-nn gate.";
pub const version = "0.1.0";
pub const target_feature = "nn";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, input, "__cmd__:")) {
        const rest = input["__cmd__:".len..];
        var iter = std.mem.splitScalar(u8, rest, '\n');
        const cmd = iter.next() orelse "";
        if (std.mem.eql(u8, cmd, "nn-train")) {
            return try allocator.dupe(u8, "nn-plugin: train (reference plugin — use `abi nn` / `abi train` for real demos)");
        }
        if (std.mem.eql(u8, cmd, "nn-sample")) {
            return try allocator.dupe(u8, "nn-plugin: sample (reference plugin — use `abi nn` for real demos)");
        }
        return try std.fmt.allocPrint(allocator, "nn-plugin: unknown command '{s}'", .{cmd});
    }
    return try std.fmt.allocPrint(allocator, "nn-plugin event (bytes={d})", .{input.len});
}

test {
    std.testing.refAllDecls(@This());
}
