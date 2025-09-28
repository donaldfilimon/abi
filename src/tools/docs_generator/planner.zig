const std = @import("std");

pub const GenerationStep = struct {
    name: []const u8,
    run: *const fn (std.mem.Allocator) anyerror!void,
};

pub const GenerationPlan = struct {
    allocator: std.mem.Allocator,
    steps: []const GenerationStep,

    pub fn init(allocator: std.mem.Allocator, steps: []const GenerationStep) GenerationPlan {
        return GenerationPlan{
            .allocator = allocator,
            .steps = steps,
        };
    }

    pub fn execute(self: GenerationPlan) !void {
        for (self.steps) |step| {
            std.log.info("Running documentation step: {s}", .{step.name});
            try step.run(self.allocator);
        }
    }
};
