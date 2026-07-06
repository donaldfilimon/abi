const std = @import("std");

pub const PlanStep = struct {
    text: []const u8 = "",
};

pub fn parsePlan(allocator: std.mem.Allocator, output: []const u8) ![]PlanStep {
    _ = allocator;
    _ = output;
    return &.{};
}

pub fn formatPlanResponse(allocator: std.mem.Allocator, steps: []const PlanStep) ![]u8 {
    _ = steps;
    return allocator.dupe(u8, "plan");
}
