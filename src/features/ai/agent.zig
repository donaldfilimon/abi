const std = @import("std");

pub const AgentError = error{
    InvalidConfiguration,
    OutOfMemory,
};

pub const AgentConfig = struct {
    name: []const u8,
    enable_history: bool = true,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,

    pub fn validate(self: AgentConfig) AgentError!void {
        if (self.name.len == 0) return AgentError.InvalidConfiguration;
        if (self.temperature < 0 or self.temperature > 2.0) return AgentError.InvalidConfiguration;
        if (self.top_p < 0 or self.top_p > 1) return AgentError.InvalidConfiguration;
    }
};

pub const Agent = struct {
    allocator: std.mem.Allocator,
    config: AgentConfig,
    history: std.ArrayListUnmanaged([]const u8) = .{},

    pub fn init(allocator: std.mem.Allocator, config: AgentConfig) AgentError!Agent {
        try config.validate();
        return Agent{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *Agent) void {
        for (self.history.items) |item| {
            self.allocator.free(item);
        }
        self.history.deinit(self.allocator);
    }

    pub fn process(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        if (self.config.enable_history) {
            const copy = try self.allocator.dupe(u8, input);
            try self.history.append(self.allocator, copy);
        }
        return std.fmt.allocPrint(allocator, "Echo: {s}", .{input});
    }

    pub fn name(self: *const Agent) []const u8 {
        return self.config.name;
    }
};
