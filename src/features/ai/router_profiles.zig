const std = @import("std");
const identity = @import("identity.zig");

pub const abbey = struct {
    pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        std.log.info("Abbey processing: {s}", .{input});
        const contract = identity.profileContract(.abbey);
        return try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
            contract.response_prefix,
            input,
            contract.response_suffix,
        });
    }
};

pub const aviva = struct {
    pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        const contract = identity.profileContract(.aviva);
        return try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
            contract.response_prefix,
            input,
            contract.response_suffix,
        });
    }
};

pub const abi_profile = struct {
    pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        const contract = identity.profileContract(.abi);
        return try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
            contract.response_prefix,
            input,
            contract.response_suffix,
        });
    }
};

test "abbey processInput" {
    const allocator = std.testing.allocator;
    const result = try abbey.processInput(allocator, "test");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "Abbey:") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "warmth, creativity, and technical care") != null);
}

test "aviva processInput returns direct expert response" {
    const allocator = std.testing.allocator;
    const result = try aviva.processInput(allocator, "what is consciousness?");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "Aviva direct expert") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "concrete answer") != null);
}

test "abi_profile processInput returns orchestration response" {
    const allocator = std.testing.allocator;
    const result = try abi_profile.processInput(allocator, "deploy to production");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "ABI orchestration review") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "intent, risk, context") != null);
}

test {
    std.testing.refAllDecls(@This());
}
