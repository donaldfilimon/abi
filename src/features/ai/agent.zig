//! Minimal AI Agent module used for basic testing and infrastructure wiring.
//!
//! This version intentionally keeps behaviour small and well bounded while the
//! wider refactor is in flight. We provide just enough surface area for the
//! build to succeed and for higher level components to depend on a stable API.

const std = @import("std");

const persona_manifest = @import("../../shared/core/persona_manifest.zig");
const metrics = @import("../../shared/observability/metrics.zig");

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
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,

    pub fn validate(self: AgentConfig) AgentError!void {
        if (self.name.len == 0) return AgentError.InvalidConfiguration;
        if (self.max_history_items == 0) return AgentError.InvalidConfiguration;
    }

    pub fn personaString(self: AgentConfig) []const u8 {
        return personaToString(self.persona);
    }
};

/// Minimal Agent implementation – tracks persona and a simple message history.
pub const Agent = struct {
    allocator: Allocator,
    config: AgentConfig,
    history: std.ArrayListUnmanaged([]const u8),
    persona_settings: PersonaSettings,
    system_prompt: []const u8,
    owns_prompt: bool,
    metrics_registry: ?*metrics.MetricsRegistry = null,

    pub fn init(allocator: Allocator, config: AgentConfig) AgentError!*Agent {
        try config.validate();

        const self = try allocator.create(Agent);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .history = .{},
            .persona_settings = .{},
            .system_prompt = "",
            .owns_prompt = false,
        };
        self.persona_settings.temperature = config.temperature;
        self.persona_settings.top_p = config.top_p;
        return self;
    }

    pub fn deinit(self: *Agent) void {
        self.clearTools();
        if (self.owns_prompt and self.system_prompt.len > 0) {
            self.allocator.free(self.system_prompt);
        }
        for (self.history.items) |entry| {
            self.allocator.free(entry);
        }
        self.history.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Returns a copy of the response so callers can manage lifetime.
    pub fn process(self: *Agent, input: []const u8, allocator: Allocator) AgentError![]const u8 {
        if (input.len == 0) return AgentError.InvalidQuery;

        const start = std.time.nanoTimestamp();
        var success = false;
        defer if (self.metrics_registry) |registry| {
            registry.recordCall(personaToString(self.config.persona), std.time.nanoTimestamp() - start, success) catch {};
        };

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

        const duplicated = try allocator.dupe(u8, input);
        success = true;
        return duplicated;
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

    pub fn setMetricsRegistry(self: *Agent, registry: *metrics.MetricsRegistry) void {
        self.metrics_registry = registry;
    }

    pub fn applyPersonaDefinition(self: *Agent, definition: persona_manifest.PersonaDefinition) AgentError!void {
        if (personaTypeFromId(definition.id)) |persona_type| {
            self.setPersona(persona_type);
        }

        try self.setSystemPrompt(definition.system_prompt);

        self.persona_settings.temperature = definition.model.temperature;
        self.persona_settings.top_p = definition.model.top_p;
        self.persona_settings.safety = definition.safety;

        self.clearTools();
        errdefer self.clearTools();

        try self.persona_settings.tools.ensureTotalCapacity(self.allocator, definition.tools.len);
        for (definition.tools) |tool| {
            const duplicated_tool = try self.allocator.dupe(u8, tool);
            errdefer self.allocator.free(duplicated_tool);
            try self.persona_settings.tools.append(self.allocator, duplicated_tool);
        }
    }

    pub fn setSystemPrompt(self: *Agent, prompt: []const u8) AgentError!void {
        if (self.owns_prompt and self.system_prompt.len > 0) {
            self.allocator.free(self.system_prompt);
        }

        if (prompt.len == 0) {
            self.system_prompt = "";
            self.owns_prompt = false;
            return;
        }

        const duplicated_prompt = self.allocator.dupe(u8, prompt) catch return AgentError.OutOfMemory;
        self.system_prompt = duplicated_prompt;
        self.owns_prompt = true;
    }

    pub fn personaSettings(self: *const Agent) PersonaSettingsView {
        return PersonaSettingsView{
            .temperature = self.persona_settings.temperature,
            .top_p = self.persona_settings.top_p,
            .tools = self.persona_settings.tools.items,
            .safety = self.persona_settings.safety,
            .system_prompt = self.system_prompt,
        };
    }

    fn clearTools(self: *Agent) void {
        for (self.persona_settings.tools.items) |tool| {
            self.allocator.free(tool);
        }
        self.persona_settings.tools.clearRetainingCapacity();
    }

    pub fn name(self: *const Agent) []const u8 {
        return self.config.name;
    }
};

pub const PersonaSettings = struct {
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    tools: std.ArrayListUnmanaged([]const u8) = .{},
    safety: persona_manifest.SafetyConfig = .{},
};

pub const PersonaSettingsView = struct {
    temperature: f32,
    top_p: f32,
    tools: []const []const u8,
    safety: persona_manifest.SafetyConfig,
    system_prompt: []const u8,
};

pub fn personaTypeFromId(id: []const u8) ?PersonaType {
    if (std.mem.eql(u8, id, "adaptive")) return .adaptive;
    if (std.mem.eql(u8, id, "technical")) return .technical;
    if (std.mem.eql(u8, id, "empathetic")) return .empathetic;
    return null;
}

pub fn personaToString(persona: PersonaType) []const u8 {
    return switch (persona) {
        .adaptive => "adaptive",
        .technical => "technical",
        .empathetic => "empathetic",
    };
}

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

test "agent applies persona definition and updates settings" {
    const testing = std.testing;

    var agent = try Agent.init(testing.allocator, .{ .name = "persona" });
    defer agent.deinit();

    const definition = persona_manifest.PersonaDefinition{
        .id = "technical",
        .display_name = "Technical",
        .description = "desc",
        .system_prompt = "system prompt",
        .model = .{ .provider = "mock", .name = "mock", .temperature = 0.55, .top_p = 0.8, .streaming = false },
        .tools = &.{ "code", "trace" },
        .safety = .{ .allow_code_execution = false, .allow_network = true, .allow_file_system = false },
        .rate_limits = .{},
    };

    try agent.applyPersonaDefinition(definition);
    try testing.expectEqual(PersonaType.technical, agent.getPersona());

    const view = agent.personaSettings();
    try testing.expectApproxEqAbs(@as(f32, 0.55), view.temperature, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.8), view.top_p, 0.0001);
    try testing.expectEqual(@as(usize, 2), view.tools.len);
    try testing.expect(std.mem.eql(u8, view.tools[0], "code"));
    try testing.expect(std.mem.eql(u8, view.tools[1], "trace"));
    try testing.expectEqualStrings("system prompt", view.system_prompt);
}
