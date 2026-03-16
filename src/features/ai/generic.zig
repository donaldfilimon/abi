//! Generic profile wrapper for prompt-defined profiles.

const std = @import("std");
const time = @import("../../services/shared/mod.zig").time;
const sync = @import("../../services/shared/mod.zig").sync;
const types = @import("types.zig");
const agent_mod = @import("agents/mod.zig");
const prompt_profiles = @import("prompts/mod.zig");
const overrides = @import("aviva/mod.zig");

pub const GenericProfileError = error{
    UnsupportedProfileType,
};

pub const GenericProfile = struct {
    allocator: std.mem.Allocator,
    profile_type: types.ProfileType,
    name: []const u8,
    agent: *agent_mod.Agent,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, profile_type: types.ProfileType) !*Self {
        const prompt_type = mapProfileType(profile_type) orelse return GenericProfileError.UnsupportedProfileType;
        const profile_def = prompt_profiles.getProfile(prompt_type);

        const cfg = agent_mod.AgentConfig{
            .name = profile_def.name,
            .temperature = profile_def.suggested_temperature,
            .max_tokens = agent_mod.DEFAULT_MAX_TOKENS,
            .system_prompt = profile_def.system_prompt,
        };

        const agent_ptr = try allocator.create(agent_mod.Agent);
        errdefer allocator.destroy(agent_ptr);
        agent_ptr.* = try agent_mod.Agent.init(allocator, cfg);

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .profile_type = profile_type,
            .name = profile_def.name,
            .agent = agent_ptr,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.agent.deinit();
        self.allocator.destroy(self.agent);
        self.allocator.destroy(self);
    }

    pub fn getName(self: *const Self) []const u8 {
        return self.name;
    }

    pub fn getType(self: *const Self) types.ProfileType {
        return self.profile_type;
    }

    /// Note: returns anyerror to match ProfileInterface.VTable.process signature.
    /// Actual errors: TimerFailed, OutOfMemory, and errors from agent.process().
    pub fn process(self: *Self, request: types.ProfileRequest) anyerror!types.ProfileResponse {
        var timer = time.Timer.start() catch {
            return error.TimerFailed;
        };

        const applied = try overrides.apply(self.allocator, self.agent, request);
        defer overrides.restore(self.allocator, self.agent, applied);

        const content = try self.agent.process(request.content, self.allocator);
        const elapsed_ms = timer.read() / std.time.ns_per_ms;

        return types.ProfileResponse{
            .content = content,
            .profile = self.profile_type,
            .confidence = 0.7,
            .generation_time_ms = elapsed_ms,
        };
    }

    pub fn interface(self: *Self) types.ProfileInterface {
        return .{
            .ptr = self,
            .vtable = &.{
                .process = @ptrCast(&process),
                .getName = @ptrCast(&getName),
                .getType = @ptrCast(&getType),
            },
        };
    }
};

pub fn supports(profile_type: types.ProfileType) bool {
    return mapProfileType(profile_type) != null;
}

fn mapProfileType(profile_type: types.ProfileType) ?prompt_profiles.ProfileType {
    return switch (profile_type) {
        .assistant => .assistant,
        .coder => .coder,
        .writer => .writer,
        .analyst => .analyst,
        .companion => .companion,
        .docs => .docs,
        .reviewer => .reviewer,
        .minimal => .minimal,
        .abbey => .abbey,
        .ralph => .ralph,
        .aviva => .aviva,
        .abi => .abi,
        .ava => .ava,
    };
}

test {
    std.testing.refAllDecls(@This());
}
