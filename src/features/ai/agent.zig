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

    pub fn historyCount(self: *const Agent) usize {
        return self.history.items.len;
    }

    pub fn historySlice(self: *const Agent) []const []const u8 {
        return self.history.items;
    }

    pub fn clearHistory(self: *Agent) void {
        for (self.history.items) |item| {
            self.allocator.free(item);
        }
        self.history.shrinkAndFree(self.allocator, 0);
    }

    pub fn process(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        if (self.config.enable_history) {
            const copy = try self.allocator.dupe(u8, input);
            errdefer self.allocator.free(copy);
            try self.history.append(self.allocator, copy);
        }
        return std.fmt.allocPrint(allocator, "Echo: {s}", .{input});
    }

    pub fn name(self: *const Agent) []const u8 {
        return self.config.name;
    }

    pub fn setTemperature(self: *Agent, temperature: f32) AgentError!void {
        if (temperature < 0 or temperature > 2.0) return AgentError.InvalidConfiguration;
        self.config.temperature = temperature;
    }

    pub fn setTopP(self: *Agent, top_p: f32) AgentError!void {
        if (top_p < 0 or top_p > 1) return AgentError.InvalidConfiguration;
        self.config.top_p = top_p;
    }

    pub fn setHistoryEnabled(self: *Agent, enabled: bool) void {
        self.config.enable_history = enabled;
    }
};

test "agent history controls" {
    var agent = try Agent.init(std.testing.allocator, .{ .name = "test-agent" });
    defer agent.deinit();

    const response = try agent.process("hello", std.testing.allocator);
    std.testing.allocator.free(response);
    try std.testing.expectEqual(@as(usize, 1), agent.historyCount());

    agent.clearHistory();
    try std.testing.expectEqual(@as(usize, 0), agent.historyCount());

    try agent.setTemperature(0.8);
    try agent.setTopP(0.5);
    agent.setHistoryEnabled(false);
}
