//! Generic persona wrapper for prompt-defined personas.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const types = @import("types.zig");
const agent_mod = @import("../agents/agent.zig");
const prompt_personas = @import("../prompts/personas.zig");
const overrides = @import("agent_overrides.zig");

pub const GenericPersonaError = error{
    UnsupportedPersonaType,
};

pub const GenericPersona = struct {
    allocator: std.mem.Allocator,
    persona_type: types.PersonaType,
    name: []const u8,
    agent: *agent_mod.Agent,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, persona_type: types.PersonaType) !*Self {
        const prompt_type = mapPersonaType(persona_type) orelse return GenericPersonaError.UnsupportedPersonaType;
        const persona_def = prompt_personas.getPersona(prompt_type);

        const cfg = agent_mod.AgentConfig{
            .name = persona_def.name,
            .temperature = persona_def.suggested_temperature,
            .max_tokens = agent_mod.DEFAULT_MAX_TOKENS,
            .system_prompt = persona_def.system_prompt,
        };

        const agent_ptr = try allocator.create(agent_mod.Agent);
        errdefer allocator.destroy(agent_ptr);
        agent_ptr.* = try agent_mod.Agent.init(allocator, cfg);

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .persona_type = persona_type,
            .name = persona_def.name,
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

    pub fn getType(self: *const Self) types.PersonaType {
        return self.persona_type;
    }

    /// Note: returns anyerror to match PersonaInterface.VTable.process signature.
    /// Actual errors: TimerFailed, OutOfMemory, and errors from agent.process().
    pub fn process(self: *Self, request: types.PersonaRequest) anyerror!types.PersonaResponse {
        var timer = time.Timer.start() catch {
            return error.TimerFailed;
        };

        const applied = try overrides.apply(self.allocator, self.agent, request);
        defer overrides.restore(self.allocator, self.agent, applied);

        const content = try self.agent.process(request.content, self.allocator);
        const elapsed_ms = timer.read() / std.time.ns_per_ms;

        return types.PersonaResponse{
            .content = content,
            .persona = self.persona_type,
            .confidence = 0.7,
            .generation_time_ms = elapsed_ms,
        };
    }

    pub fn interface(self: *Self) types.PersonaInterface {
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

pub fn supports(persona_type: types.PersonaType) bool {
    return mapPersonaType(persona_type) != null;
}

fn mapPersonaType(persona_type: types.PersonaType) ?prompt_personas.PersonaType {
    return switch (persona_type) {
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
    };
}

test {
    std.testing.refAllDecls(@This());
}
