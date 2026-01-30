//! C-compatible AI agent exports.
//! Provides agent creation and chat functionality for FFI.

const std = @import("std");
const errors = @import("errors.zig");

/// Opaque agent handle for C API.
pub const AgentHandle = opaque {};

/// Agent config matching C header (abi_agent_config_t).
pub const AgentConfig = extern struct {
    name: [*:0]const u8,
    persona: ?[*:0]const u8 = null,
    temperature: f32 = 0.7,
    enable_history: bool = true,
};

/// Internal agent state.
const AgentState = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    persona: ?[]const u8,
    temperature: f32,
    history: std.ArrayListUnmanaged(Message),

    const Message = struct {
        role: []const u8,
        content: []const u8,
    };

    pub fn init(allocator: std.mem.Allocator, config: AgentConfig) !*AgentState {
        const state = try allocator.create(AgentState);
        errdefer allocator.destroy(state);

        const name = std.mem.span(config.name);
        const name_copy = try allocator.dupe(u8, name);
        errdefer allocator.free(name_copy);

        var persona_copy: ?[]const u8 = null;
        if (config.persona) |p| {
            persona_copy = try allocator.dupe(u8, std.mem.span(p));
        }
        errdefer if (persona_copy) |p| allocator.free(p);

        state.* = .{
            .allocator = allocator,
            .name = name_copy,
            .persona = persona_copy,
            .temperature = config.temperature,
            .history = std.ArrayListUnmanaged(Message).empty,
        };

        return state;
    }

    pub fn deinit(self: *AgentState) void {
        for (self.history.items) |msg| {
            self.allocator.free(msg.role);
            self.allocator.free(msg.content);
        }
        self.history.deinit(self.allocator);
        if (self.persona) |p| self.allocator.free(p);
        self.allocator.free(self.name);
        self.allocator.destroy(self);
    }

    pub fn chat(self: *AgentState, message: []const u8) ![]u8 {
        // Store user message in history
        try self.history.append(self.allocator, .{
            .role = try self.allocator.dupe(u8, "user"),
            .content = try self.allocator.dupe(u8, message),
        });

        // Generate a mock response (real implementation would call LLM)
        const response = try std.fmt.allocPrint(
            self.allocator,
            "[{s}] Received: \"{s}\" (history: {d} messages)",
            .{ self.name, message, self.history.items.len },
        );

        // Store assistant response
        try self.history.append(self.allocator, .{
            .role = try self.allocator.dupe(u8, "assistant"),
            .content = try self.allocator.dupe(u8, response),
        });

        return response;
    }

    pub fn clearHistory(self: *AgentState) void {
        for (self.history.items) |msg| {
            self.allocator.free(msg.role);
            self.allocator.free(msg.content);
        }
        self.history.clearRetainingCapacity();
    }
};

// Global allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

/// Create an AI agent.
pub export fn abi_agent_create(
    framework: ?*anyopaque,
    config: *const AgentConfig,
    out_agent: *?*AgentHandle,
) errors.Error {
    _ = framework; // Framework handle not used in standalone implementation

    const allocator = gpa.allocator();
    const state = AgentState.init(allocator, config.*) catch |err| {
        return errors.fromZigError(err);
    };
    out_agent.* = @ptrCast(state);
    return errors.OK;
}

/// Destroy an agent.
pub export fn abi_agent_destroy(handle: ?*AgentHandle) void {
    if (handle) |h| {
        const state: *AgentState = @ptrCast(@alignCast(h));
        state.deinit();
    }
}

/// Send a message and get a response.
pub export fn abi_agent_chat(
    handle: ?*AgentHandle,
    message: [*:0]const u8,
    out_response: *?[*:0]u8,
) errors.Error {
    const state: *AgentState = @ptrCast(@alignCast(handle orelse return errors.NOT_INITIALIZED));
    const msg = std.mem.span(message);

    const response = state.chat(msg) catch |err| {
        return errors.fromZigError(err);
    };

    // Convert to null-terminated for C
    const response_z = state.allocator.allocSentinel(u8, response.len, 0) catch {
        state.allocator.free(response);
        return errors.OUT_OF_MEMORY;
    };
    @memcpy(response_z, response);
    state.allocator.free(response);

    out_response.* = response_z.ptr;
    return errors.OK;
}

/// Clear conversation history.
pub export fn abi_agent_clear_history(handle: ?*AgentHandle) errors.Error {
    const state: *AgentState = @ptrCast(@alignCast(handle orelse return errors.NOT_INITIALIZED));
    state.clearHistory();
    return errors.OK;
}

/// Free a string allocated by agent functions.
pub export fn abi_free_string(str: ?[*:0]u8) void {
    if (str) |s| {
        const allocator = gpa.allocator();
        const len = std.mem.len(s);
        allocator.free(s[0 .. len + 1]);
    }
}

test "agent exports" {
    const config = AgentConfig{
        .name = "test-agent",
        .persona = null,
        .temperature = 0.7,
        .enable_history = true,
    };

    var agent: ?*AgentHandle = null;
    try std.testing.expectEqual(errors.OK, abi_agent_create(null, &config, &agent));
    try std.testing.expect(agent != null);

    // Chat
    var response: ?[*:0]u8 = null;
    try std.testing.expectEqual(errors.OK, abi_agent_chat(agent, "Hello!", &response));
    try std.testing.expect(response != null);

    // Verify response contains our message
    const resp_str = std.mem.span(response.?);
    try std.testing.expect(std.mem.indexOf(u8, resp_str, "Hello!") != null);

    // Free response
    abi_free_string(response);

    // Clear history
    try std.testing.expectEqual(errors.OK, abi_agent_clear_history(agent));

    // Destroy
    abi_agent_destroy(agent);
}
