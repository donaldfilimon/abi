const std = @import("std");
const workflow_mod = @import("../workflow.zig");
const roles = @import("../roles.zig");
const agents_mod = @import("../../agents/mod.zig");
const support = @import("support.zig");

pub const PreparedStep = struct {
    profile: ?roles.Profile,
    profile_name: []const u8,
    inputs: []const u8,
    inputs_owned: bool,
    prompt: []const u8,
    prompt_owned: bool,
    agent: ?*agents_mod.Agent,

    pub fn prepare(runner: anytype, step: *const workflow_mod.Step) PreparedStep {
        const profile = support.assignProfile(runner, step);
        const profile_name: []const u8 = if (profile) |resolved| resolved.id else "default";
        const inputs = support.gatherInputs(runner, step) catch "";
        const prompt = support.buildPrompt(runner, step.prompt_template, inputs) catch step.prompt_template;

        return .{
            .profile = profile,
            .profile_name = profile_name,
            .inputs = inputs,
            .inputs_owned = inputs.len > 0,
            .prompt = prompt,
            .prompt_owned = !std.mem.eql(u8, prompt, step.prompt_template),
            .agent = support.selectAgent(runner, profile_name),
        };
    }

    pub fn deinit(self: *PreparedStep, allocator: std.mem.Allocator) void {
        if (self.inputs_owned) {
            allocator.free(self.inputs);
        }
        if (self.prompt_owned) {
            allocator.free(self.prompt);
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}
