const std = @import("std");

pub const abbey = @import("abbey/mod.zig");
pub const constitution = @import("constitution/mod.zig");
pub const AuditResult = constitution.AuditResult;
pub const Principle = constitution.Principle;

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const response = try abbey.processInput(allocator, input);
    const audit = constitution.Constitution.validate(response);
    if (!audit.passed) std.log.warn("Constitutional violation!", .{});
    return response;
}
