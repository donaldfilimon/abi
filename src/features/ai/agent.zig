//! Minimal AI Agent module used for basic testing and infrastructure wiring.
//!
//! This implementation now incorporates persona manifests, sampling controls,
//! and telemetry hooks while keeping the public surface stable for ongoing
//! refactors.

const std = @import("std");
const persona_manifest = @import("persona_manifest.zig");
const observability = @import("../../shared/observability/mod.zig");
const metrics = @import("../../shared/observability/metrics.zig");
const PersonaDefinition = persona_manifest.PersonaDefinition;
const PersonaSafety = std.meta.FieldType(PersonaDefinition, .safety);
const PersonaToolList = std.meta.FieldType(PersonaDefinition, .tools);

pub const PersonaSettingsView = struct {
    temperature: f32,
    top_p: f32,
    tools: PersonaToolList,
    safety: ?PersonaSafety,
    system_prompt: []const u8,
};

const PersonaSettingsState = struct {
    temperature: f32,
    top_p: f32,
    tools: std.ArrayListUnmanaged([]const u8) = .{},
    safety: ?PersonaSafety = null,
};

pub const Allocator = std.mem.Allocator;

/// Errors that an agent operation can produce.
pub const AgentError = error{
    InvalidConfiguration,
    InvalidQuery,
    OutOfMemory,
};

pub const PersonaType = persona_manifest.PersonaArchetype;
pub const PersonaManifest = persona_manifest.PersonaManifest;

fn personaTypeFromId(id: []const u8) ?PersonaType {
    return persona_manifest.parseArchetype(id);
}

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
    enable_streaming: bool = true,
    enable_function_calling: bool = true,
    max_history_items: usize = 64,
    capabilities: AgentCapabilities = .{},
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    rate_limit_per_minute: u32 = 60,
    tools: []const []const u8 = &.{},
    safety_filters: []const []const u8 = &.{},

    pub fn validate(self: AgentConfig) AgentError!void {
        if (self.name.len == 0) return AgentError.InvalidConfiguration;
        if (self.max_history_items == 0) return AgentError.InvalidConfiguration;
        if (self.temperature < 0 or self.temperature > 2.0) return AgentError.InvalidConfiguration;
        if (self.top_p < 0 or self.top_p > 1) return AgentError.InvalidConfiguration;
        if (self.rate_limit_per_minute == 0) return AgentError.InvalidConfiguration;
    }

    pub fn personaString(self: AgentConfig) []const u8 {
        return persona_manifest.archetypeToString(self.persona);
    }
};

/// Minimal Agent implementation – tracks persona, sampling controls, telemetry,
/// and a simple message history.
pub const Agent = struct {
    allocator: Allocator,
    config: AgentConfig,
    history: std.ArrayListUnmanaged([]const u8),
    tools: std.ArrayListUnmanaged([]const u8),
    safety_filters: std.ArrayListUnmanaged([]const u8),
    temperature: f32,
    top_p: f32,
    rate_limit_per_minute: u32,
    streaming: bool,
    function_calling: bool,
    telemetry: ?*observability.TelemetrySink = null,
    metrics_registry: ?*metrics.MetricsRegistry = null,
    persona_settings: PersonaSettingsState,
    system_prompt: []const u8,
    owns_prompt: bool,

    pub fn init(allocator: Allocator, config: AgentConfig) AgentError!*Agent {
        try config.validate();

        const self = try allocator.create(Agent);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .history = .{},
            .tools = .{},
            .safety_filters = .{},
            .temperature = config.temperature,
            .top_p = config.top_p,
            .rate_limit_per_minute = config.rate_limit_per_minute,
            .streaming = config.enable_streaming,
            .function_calling = config.enable_function_calling,
            .telemetry = null,
            .metrics_registry = null,
            .persona_settings = .{
                .temperature = config.temperature,
                .top_p = config.top_p,
                .tools = .{},
                .safety = null,
            },
            .system_prompt = "",
            .owns_prompt = false,
        };

        self.config.tools = &.{};
        self.config.safety_filters = &.{};
        try self.replaceStringList(&self.tools, config.tools);
        try self.replaceStringList(&self.safety_filters, config.safety_filters);
        self.config.tools = self.tools.items;
        self.config.safety_filters = self.safety_filters.items;

        return self;
    }

    pub fn deinit(self: *Agent) void {
        clearStringList(&self.history, self.allocator);
        self.history.deinit(self.allocator);
        clearStringList(&self.tools, self.allocator);
        self.tools.deinit(self.allocator);
        clearStringList(&self.safety_filters, self.allocator);
        self.safety_filters.deinit(self.allocator);
        self.clearTools();
        self.persona_settings.tools.deinit(self.allocator);
        if (self.owns_prompt and self.system_prompt.len > 0) {
            self.allocator.free(self.system_prompt);
        }
        self.allocator.destroy(self);
    }

    /// Returns a copy of the response so callers can manage lifetime.
    pub fn process(self: *Agent, input: []const u8, allocator: Allocator) AgentError![]const u8 {
        if (input.len == 0) return AgentError.InvalidQuery;

        const persona_label = persona_manifest.archetypeToString(self.getPersona());
        const telemetry_sink = self.telemetry;
        var start_ts: i128 = 0;
        if (telemetry_sink) |_| {
            start_ts = std.time.nanoTimestamp();
        }
        errdefer if (telemetry_sink) |sink| {
            const latency = computeLatencyNs(start_ts, std.time.nanoTimestamp());
            _ = sink.record(persona_label, latency, .failure, "agent_error") catch {};
        };

        if (self.config.enable_history) {
            if (self.history.items.len == self.config.max_history_items) {
                const oldest = self.history.items[0];
                if (self.history.items.len > 1) {
                    std.mem.copyForwards([]const u8, self.history.items[0 .. self.history.items.len - 1], self.history.items[1..]);
                }
                self.history.items.len -= 1;
                self.allocator.free(oldest);
            }
            const stored = self.allocator.dupe(u8, input) catch return AgentError.OutOfMemory;
            errdefer self.allocator.free(stored);
            self.history.append(self.allocator, stored) catch {
                self.allocator.free(stored);
                return AgentError.OutOfMemory;
            };
        }

        const response = allocator.dupe(u8, input) catch return AgentError.OutOfMemory;

        if (telemetry_sink) |sink| {
            const latency = computeLatencyNs(start_ts, std.time.nanoTimestamp());
            _ = sink.record(persona_label, latency, .success, null) catch {};
        }

        return response;
    }

    pub fn clearHistory(self: *Agent) void {
        clearStringList(&self.history, self.allocator);
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

    pub fn attachTelemetry(self: *Agent, sink: *observability.TelemetrySink) void {
        self.telemetry = sink;
    }

    pub fn applyManifest(self: *Agent, manifest: *const PersonaManifest) AgentError!void {
        self.setPersona(manifest.archetype);
        self.temperature = manifest.temperature;
        self.top_p = manifest.top_p;
        self.rate_limit_per_minute = manifest.rate_limit_per_minute;
        self.streaming = manifest.streaming;
        self.function_calling = manifest.function_calling;
        self.config.temperature = manifest.temperature;
        self.config.top_p = manifest.top_p;
        self.config.rate_limit_per_minute = manifest.rate_limit_per_minute;
        self.config.enable_streaming = manifest.streaming;
        self.config.enable_function_calling = manifest.function_calling;
        try self.replaceStringList(&self.tools, manifest.toolsSlice());
        try self.replaceStringList(&self.safety_filters, manifest.safetyFiltersSlice());
        self.config.tools = self.tools.items;
        self.config.safety_filters = self.safety_filters.items;
    }

    pub fn toolsSlice(self: *const Agent) []const []const u8 {
        return self.tools.items;
    }

    pub fn safetyFiltersSlice(self: *const Agent) []const []const u8 {
        return self.safety_filters.items;
    }

    fn replaceStringList(self: *Agent, list: *std.ArrayListUnmanaged([]const u8), values: []const []const u8) AgentError!void {
        clearStringList(list, self.allocator);
        for (values) |value| {
            const copy = self.allocator.dupe(u8, value) catch return AgentError.OutOfMemory;
            errdefer self.allocator.free(copy);
            list.append(self.allocator, copy) catch {
                self.allocator.free(copy);
                return AgentError.OutOfMemory;
            };
        }
    }
};

fn clearStringList(list: *std.ArrayListUnmanaged([]const u8), allocator: Allocator) void {
    for (list.items) |item| {
        allocator.free(item);
    }
    list.items.len = 0;
}

fn computeLatencyNs(start: i128, end: i128) u64 {
    const delta = end - start;
    if (delta <= 0) return 0;
    return std.math.cast(u64, delta) orelse std.math.maxInt(u64);
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

test "agent applies manifest configuration" {
    const testing = std.testing;
    const json =
        "{\n" ++
        "  \"name\": \"coder\",\n" ++
        "  \"system_prompt\": \"Write precise code\",\n" ++
        "  \"temperature\": 0.4,\n" ++
        "  \"top_p\": 0.7,\n" ++
        "  \"rate_limit_per_minute\": 42,\n" ++
        "  \"streaming\": false,\n" ++
        "  \"function_calling\": true,\n" ++
        "  \"archetype\": \"technical\",\n" ++
        "  \"tools\": [\"editor\"],\n" ++
        "  \"safety_filters\": [\"toxicity\"]\n" ++
        "}\n";
    var manifest = try persona_manifest.loadManifestFromSlice(testing.allocator, json, .json);
    defer manifest.deinit();

    var agent = try Agent.init(testing.allocator, .{ .name = "manifest" });
    defer agent.deinit();

    try agent.applyManifest(&manifest);
    try testing.expectEqual(PersonaType.technical, agent.getPersona());
    try testing.expectEqual(@as(f32, 0.4), agent.temperature);
    try testing.expectEqual(@as(f32, 0.7), agent.top_p);
    try testing.expectEqual(@as(u32, 42), agent.rate_limit_per_minute);
    try testing.expectEqual(@as(bool, false), agent.streaming);
    try testing.expectEqual(@as(usize, 1), agent.toolsSlice().len);
}

test "agent emits telemetry metrics" {
    const testing = std.testing;
    var sink = observability.TelemetrySink.init(testing.allocator);
    defer sink.deinit();

    var agent = try Agent.init(testing.allocator, .{ .name = "telemetry" });
    defer agent.deinit();
    agent.attachTelemetry(&sink);

    const response = try agent.process("ping", testing.allocator);
    defer testing.allocator.free(response);

    var snapshot = try sink.snapshot(testing.allocator);
    defer snapshot.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), snapshot.total_calls);
}
