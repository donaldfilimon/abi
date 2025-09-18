//! Minimal AI Agent module used for basic testing and infrastructure wiring.
//!
//! This version intentionally keeps behaviour small and well bounded while the
//! wider refactor is in flight. We provide just enough surface area for the
//! build to succeed and for higher level components to depend on a stable API.

const std = @import("std");

pub const Allocator = std.mem.Allocator;

/// Errors that an agent operation can produce.
pub const AgentError = error{
    InvalidConfiguration,
    InvalidQuery,
    OutOfMemory,
};

/// Simple set of personas that are safe to use across the codebase.
pub const PersonaType = enum {
    adaptive,
    technical,
    empathetic,
};

/// Capabilities flag set – kept small for now, can expand later without
/// breaking the ABI.
pub const AgentCapabilities = packed struct(u8) {
    text_generation: bool = true,
    reasoning: bool = false,
    _reserved: u6 = 0,
};

/// Configuration supplied when constructing an agent.
pub const AgentConfig = struct {
    name: []const u8,
    persona: PersonaType = .adaptive,
    enable_history: bool = true,
    max_history_items: usize = 64,
    capabilities: AgentCapabilities = .{},

    pub fn validate(self: AgentConfig) AgentError!void {
        if (self.name.len == 0) return AgentError.InvalidConfiguration;
        if (self.max_history_items == 0) return AgentError.InvalidConfiguration;
    }
};

/// Minimal Agent implementation – tracks persona and a simple message history.
pub const Agent = struct {
    allocator: Allocator,
    config: AgentConfig,
    history: std.ArrayListUnmanaged([]const u8),

    pub fn init(allocator: Allocator, config: AgentConfig) AgentError!*Agent {
        try config.validate();

        const self = try allocator.create(Agent);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .history = .{},
        };
        return self;
    }

    pub fn deinit(self: *Agent) void {
        for (self.history.items) |entry| {
            self.allocator.free(entry);
        }
        self.history.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Returns a copy of the response so callers can manage lifetime.
    pub fn process(self: *Agent, input: []const u8, allocator: Allocator) AgentError![]const u8 {
        if (input.len == 0) return AgentError.InvalidQuery;

        if (self.config.enable_history) {
            if (self.history.items.len == self.config.max_history_items and self.history.items.len > 0) {
                const oldest = self.history.items[0];
                if (self.history.items.len > 1) {
                    std.mem.copyForwards([]const u8, self.history.items[0 .. self.history.items.len - 1], self.history.items[1..]);
                }
                self.history.items.len -= 1;
                self.allocator.free(oldest);
            }
            const stored = try self.allocator.dupe(u8, input);
            errdefer self.allocator.free(stored);
            try self.history.append(self.allocator, stored);
        }

        return allocator.dupe(u8, input);
    }

    pub fn clearHistory(self: *Agent) void {
        for (self.history.items) |entry| {
            self.allocator.free(entry);
        }
        self.history.clearRetainingCapacity();
    }

    pub fn historyCount(self: *const Agent) usize {
        return self.history.items.len;
    }

    pub fn getPersona(self: *const Agent) PersonaType {
        return self.config.persona;
    }

    pub fn setPersona(self: *Agent, persona: PersonaType) void {
        self.config.persona = persona;
    }

    pub fn name(self: *const Agent) []const u8 {
        return self.config.name;
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

test "agent initialises and manages persona" {
    const testing = std.testing;

    var agent = try Agent.init(testing.allocator, .{ .name = "test" });
    defer agent.deinit();

    try testing.expectEqual(PersonaType.adaptive, agent.getPersona());
    agent.setPersona(.technical);
    try testing.expectEqual(PersonaType.technical, agent.getPersona());
    try testing.expectEqualStrings("test", agent.name());
}

test "agent records history when processing input" {
    const testing = std.testing;

    var agent = try Agent.init(testing.allocator, .{ .name = "history", .max_history_items = 2 });
    defer agent.deinit();

    const response = try agent.process("hello", testing.allocator);
    defer testing.allocator.free(response);

    try testing.expect(std.mem.eql(u8, response, "hello"));
    try testing.expectEqual(@as(usize, 1), agent.historyCount());

    const response2 = try agent.process("world", testing.allocator);
    defer testing.allocator.free(response2);
    try testing.expectEqual(@as(usize, 2), agent.historyCount());

    // Exceed history cap – oldest element is dropped.
    const response3 = try agent.process("again", testing.allocator);
    defer testing.allocator.free(response3);
    try testing.expectEqual(@as(usize, 2), agent.historyCount());
}

test "agent rejects empty input" {
    const testing = std.testing;

    var agent = try Agent.init(testing.allocator, .{ .name = "guard" });
    defer agent.deinit();

    try testing.expectError(AgentError.InvalidQuery, agent.process("", testing.allocator));
}
