//! Tool-Augmented AI Agent
//!
//! Wraps the base Agent with a tool execution loop, enabling the agent to
//! autonomously call registered tools (file I/O, shell, search, etc.) during
//! conversations. The agent decides which tools to invoke based on the LLM
//! response, executes them, feeds results back, and loops until the LLM
//! produces a final text-only answer.
//!
//! ## Tool Call Protocol
//!
//! The LLM emits tool calls using XML-style markers:
//! ```
//! <tool_call>{"name": "read_file", "args": {"path": "src/abi.zig"}}</tool_call>
//! ```
//!
//! Multiple tool calls can appear in a single response.

const std = @import("std");
const json = std.json;
const agent_mod = @import("../agents/agent.zig");
const tools_mod = @import("mod.zig");

const Agent = agent_mod.Agent;
const AgentConfig = agent_mod.AgentConfig;
const Tool = tools_mod.Tool;
const ToolResult = tools_mod.ToolResult;
const ToolRegistry = tools_mod.ToolRegistry;
const ToolContext = tools_mod.Context;
const ToolExecutionError = tools_mod.ToolExecutionError;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the tool-augmented agent.
pub const ToolAgentConfig = struct {
    /// Base agent configuration (name, backend, model, etc.).
    agent: AgentConfig = .{ .name = "tool-agent" },
    /// Maximum tool call iterations per user message.
    max_tool_iterations: usize = 10,
    /// Maximum characters to include from tool output in context.
    tool_result_max_chars: usize = 8000,
    /// Whether to require confirmation for destructive tools.
    require_confirmation: bool = true,
    /// Tool names considered destructive (require confirmation).
    destructive_tools: []const []const u8 = &default_destructive_tools,
    /// Enable memory integration for learning across sessions.
    enable_memory: bool = false,
    /// Enable self-reflection after each response.
    enable_reflection: bool = false,
    /// Working directory for tool execution.
    working_directory: []const u8 = ".",
};

const default_destructive_tools = [_][]const u8{
    "shell",
    "write_file",
    "edit",
    "insert_lines",
    "delete_lines",
    "kill_process",
    "spawn_background",
};

// ============================================================================
// Tool Call Types
// ============================================================================

/// A parsed tool call request extracted from LLM output.
pub const ToolCallRequest = struct {
    name: []const u8,
    args_json: []const u8,
};

/// Record of a tool call execution for logging/display.
pub const ToolCallRecord = struct {
    tool_name: []const u8,
    args_summary: []const u8,
    success: bool,
    output_preview: []const u8,
};

/// Callback type for destructive operation confirmation.
/// Returns true if the operation should proceed.
pub const ConfirmationFn = *const fn (tool_name: []const u8, args_json: []const u8) bool;

// ============================================================================
// ToolAugmentedAgent
// ============================================================================

/// An AI agent enhanced with tool-use capabilities.
///
/// Wraps a base `Agent` and adds:
/// - Tool registry with registered tools
/// - Tool execution loop (parse → execute → feed back → repeat)
/// - Confirmation for destructive operations
/// - Tool call logging
/// - Optional memory integration
pub const ToolAugmentedAgent = struct {
    allocator: std.mem.Allocator,
    agent: Agent,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    config: ToolAgentConfig,
    confirmation_callback: ?ConfirmationFn,
    tool_call_log: std.ArrayListUnmanaged(ToolCallRecord),
    /// System prompt supplement describing available tools.
    tool_descriptions: ?[]u8,

    const Self = @This();

    /// Initialize a tool-augmented agent.
    pub fn init(allocator: std.mem.Allocator, config: ToolAgentConfig) !Self {
        var base_agent = try Agent.init(allocator, config.agent);
        errdefer base_agent.deinit();

        var registry = ToolRegistry.init(allocator);
        errdefer registry.deinit();

        const tool_ctx = tools_mod.createContext(allocator, config.working_directory);

        return .{
            .allocator = allocator,
            .agent = base_agent,
            .tool_registry = registry,
            .tool_context = tool_ctx,
            .config = config,
            .confirmation_callback = null,
            .tool_call_log = .{},
            .tool_descriptions = null,
        };
    }

    /// Clean up all resources.
    pub fn deinit(self: *Self) void {
        for (self.tool_call_log.items) |*record| {
            self.allocator.free(record.tool_name);
            self.allocator.free(record.args_summary);
            self.allocator.free(record.output_preview);
        }
        self.tool_call_log.deinit(self.allocator);
        if (self.tool_descriptions) |desc| {
            self.allocator.free(desc);
        }
        self.tool_registry.deinit();
        self.agent.deinit();
    }

    /// Register all code-agent tools (file, search, edit, OS).
    pub fn registerCodeAgentTools(self: *Self) !void {
        try tools_mod.registerCodeAgentTools(&self.tool_registry);
        try self.rebuildToolDescriptions();
    }

    /// Register all agent tools including extended OS capabilities.
    pub fn registerAllAgentTools(self: *Self) !void {
        try tools_mod.registerAllAgentTools(&self.tool_registry);
        try self.rebuildToolDescriptions();
    }

    /// Register a single tool.
    pub fn registerTool(self: *Self, t: *const Tool) !void {
        try self.tool_registry.register(t);
        try self.rebuildToolDescriptions();
    }

    /// Set the confirmation callback for destructive operations.
    pub fn setConfirmationCallback(self: *Self, cb: ConfirmationFn) void {
        self.confirmation_callback = cb;
    }

    /// Process user input with tool augmentation.
    ///
    /// 1. Injects tool descriptions into system context
    /// 2. Sends user message to LLM
    /// 3. Parses response for <tool_call> markers
    /// 4. Executes tools, feeds results back as tool messages
    /// 5. Loops until LLM returns text-only or max iterations reached
    /// 6. Returns final text response
    pub fn processWithTools(self: *Self, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
        // Build augmented input with tool descriptions if available
        const augmented_input = if (self.tool_descriptions) |desc| blk: {
            break :blk try std.fmt.allocPrint(allocator, "{s}\n\n[Available Tools]\n{s}\n\n" ++
                "To use a tool, include in your response:\n" ++
                "<tool_call>{{\"name\": \"tool_name\", \"args\": {{...}}}}</tool_call>\n\n" ++
                "User message: {s}", .{ "", desc, input });
        } else blk: {
            break :blk try allocator.dupe(u8, input);
        };
        defer allocator.free(augmented_input);

        // Get initial response from agent
        var response = try self.agent.process(augmented_input, allocator);

        // Tool execution loop
        var iteration: usize = 0;
        while (iteration < self.config.max_tool_iterations) : (iteration += 1) {
            // Parse tool calls from response
            var tool_calls = parseToolCalls(response, allocator) catch {
                if (iteration + 1 < self.config.max_tool_iterations and shouldRepairStructuredReply(response)) {
                    const repaired = try self.repairStructuredReply(response, allocator);
                    allocator.free(response);
                    response = repaired;
                    continue;
                }
                break;
            };
            defer {
                for (tool_calls.items) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.args_json);
                }
                tool_calls.deinit(allocator);
            }

            if (tool_calls.items.len == 0) {
                if (iteration + 1 < self.config.max_tool_iterations and shouldRepairStructuredReply(response)) {
                    const repaired = try self.repairStructuredReply(response, allocator);
                    allocator.free(response);
                    response = repaired;
                    continue;
                }
                break;
            }

            // Execute each tool call
            var results_buf = std.ArrayListUnmanaged(u8).empty;
            defer results_buf.deinit(allocator);

            for (tool_calls.items) |tc| {
                const tool_output = self.executeTool(tc, allocator) catch |err| blk: {
                    const err_msg = std.fmt.allocPrint(allocator, "Tool '{s}' failed: {t}", .{ tc.name, err }) catch
                        break :blk try allocator.dupe(u8, "Tool execution failed");
                    break :blk err_msg;
                };
                defer allocator.free(tool_output);

                // Truncate if needed
                const max_chars = self.config.tool_result_max_chars;
                const display_output = if (tool_output.len > max_chars)
                    tool_output[0..max_chars]
                else
                    tool_output;

                // Append to results buffer
                const formatted = try std.fmt.allocPrint(allocator, "\n[Tool Result: {s}]\n{s}\n", .{ tc.name, display_output });
                defer allocator.free(formatted);
                try results_buf.appendSlice(allocator, formatted);
            }

            // Free previous response
            allocator.free(response);

            // Feed tool results back to agent
            const results_str = try allocator.dupe(u8, results_buf.items);
            defer allocator.free(results_str);

            response = try self.agent.process(results_str, allocator);
        }

        return response;
    }

    /// Get the tool call log.
    pub fn getToolCallLog(self: *const Self) []const ToolCallRecord {
        return self.tool_call_log.items;
    }

    /// Get the number of registered tools.
    pub fn toolCount(self: *const Self) usize {
        return self.tool_registry.count();
    }

    /// Clear the tool call log.
    pub fn clearLog(self: *Self) void {
        for (self.tool_call_log.items) |*record| {
            self.allocator.free(record.tool_name);
            self.allocator.free(record.args_summary);
            self.allocator.free(record.output_preview);
        }
        self.tool_call_log.clearRetainingCapacity();
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    fn executeTool(self: *Self, tc: ToolCallRequest, allocator: std.mem.Allocator) ![]u8 {
        // Check if tool exists
        const tool_def = self.tool_registry.get(tc.name) orelse {
            return std.fmt.allocPrint(allocator, "Error: Unknown tool '{s}'", .{tc.name});
        };

        // Check confirmation for destructive tools
        if (self.config.require_confirmation and self.isDestructiveTool(tc.name)) {
            if (self.confirmation_callback) |cb| {
                if (!cb(tc.name, tc.args_json)) {
                    return try allocator.dupe(u8, "Operation cancelled by user.");
                }
            }
        }

        // Parse args JSON
        const parsed = json.parseFromSlice(json.Value, allocator, tc.args_json, .{}) catch {
            return std.fmt.allocPrint(allocator, "Error: Invalid JSON args for tool '{s}'", .{tc.name});
        };
        defer parsed.deinit();

        // Execute
        var result = tool_def.execute(&self.tool_context, parsed.value) catch |err| {
            return std.fmt.allocPrint(allocator, "Error executing '{s}': {t}", .{ tc.name, err });
        };
        defer result.deinit();

        // Log the call
        const log_record = ToolCallRecord{
            .tool_name = try allocator.dupe(u8, tc.name),
            .args_summary = try allocator.dupe(u8, if (tc.args_json.len > 100) tc.args_json[0..100] else tc.args_json),
            .success = result.success,
            .output_preview = try allocator.dupe(u8, if (result.output.len > 200) result.output[0..200] else result.output),
        };
        try self.tool_call_log.append(allocator, log_record);

        // Return output
        if (result.success) {
            return try allocator.dupe(u8, result.output);
        } else {
            const err_msg = result.error_message orelse "Unknown error";
            return std.fmt.allocPrint(allocator, "Error: {s}", .{err_msg});
        }
    }

    fn isDestructiveTool(self: *const Self, name: []const u8) bool {
        for (self.config.destructive_tools) |dt| {
            if (std.mem.eql(u8, name, dt)) return true;
        }
        return false;
    }

    fn rebuildToolDescriptions(self: *Self) !void {
        if (self.tool_descriptions) |old| {
            self.allocator.free(old);
            self.tool_descriptions = null;
        }
        self.tool_descriptions = try generateToolDescriptions(&self.tool_registry, self.allocator);
    }

    fn repairStructuredReply(self: *Self, response: []const u8, allocator: std.mem.Allocator) ![]u8 {
        const prompt = try std.fmt.allocPrint(
            allocator,
            "Your previous reply looked like structured payload instead of a final user answer.\n" ++
                "Previous reply:\n{s}\n\n" ++
                "If you need tools, use strictly:\n" ++
                "<tool_call>{{\"name\":\"tool_name\",\"args\":{{...}}}}</tool_call>\n\n" ++
                "Otherwise return a concise natural-language final answer with no JSON.",
            .{response},
        );
        defer allocator.free(prompt);

        return self.agent.process(prompt, allocator);
    }
};

// ============================================================================
// Tool Description Generator
// ============================================================================

/// Generate a formatted description of all registered tools for system prompt injection.
pub fn generateToolDescriptions(registry: *const ToolRegistry, allocator: std.mem.Allocator) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8).empty;
    errdefer buf.deinit(allocator);

    // Iterate all tools in registry
    var iter = registry.tools.iterator();
    while (iter.next()) |entry| {
        const t = entry.value_ptr.*;
        try buf.appendSlice(allocator, "- ");
        try buf.appendSlice(allocator, t.name);
        try buf.appendSlice(allocator, "(");

        for (t.parameters, 0..) |param, i| {
            if (i > 0) try buf.appendSlice(allocator, ", ");
            try buf.appendSlice(allocator, param.name);
            try buf.appendSlice(allocator, ": ");
            const type_str = switch (param.type) {
                .string => "string",
                .integer => "integer",
                .boolean => "boolean",
                .array => "array",
                .object => "object",
                .number => "number",
            };
            try buf.appendSlice(allocator, type_str);
            if (!param.required) try buf.appendSlice(allocator, "?");
        }

        try buf.appendSlice(allocator, "): ");
        try buf.appendSlice(allocator, t.description);
        try buf.appendSlice(allocator, "\n");
    }

    return try buf.toOwnedSlice(allocator);
}

fn shouldRepairStructuredReply(response: []const u8) bool {
    const trimmed = std.mem.trim(u8, response, " \t\r\n");
    if (trimmed.len == 0) return true;

    if (std.mem.startsWith(u8, trimmed, "{") or std.mem.startsWith(u8, trimmed, "[")) return true;
    if (std.mem.startsWith(u8, trimmed, "```")) return true;
    if (std.mem.startsWith(u8, trimmed, "<tool_call>") and std.mem.indexOf(u8, trimmed, "</tool_call>") == null) return true;
    if (std.mem.indexOf(u8, trimmed, "\"args\"") != null and std.mem.indexOf(u8, trimmed, "\"name\"") == null) return true;
    return false;
}

// ============================================================================
// Tool Call Parser
// ============================================================================

/// Parse tool calls from LLM response text.
///
/// Looks for patterns like:
/// ```
/// <tool_call>{"name": "tool_name", "args": {...}}</tool_call>
/// ```
pub fn parseToolCalls(response: []const u8, allocator: std.mem.Allocator) !std.ArrayListUnmanaged(ToolCallRequest) {
    var calls = std.ArrayListUnmanaged(ToolCallRequest).empty;
    errdefer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.args_json);
        }
        calls.deinit(allocator);
    }

    const open_tag = "<tool_call>";
    const close_tag = "</tool_call>";

    var pos: usize = 0;
    while (pos < response.len) {
        const start = std.mem.indexOfPos(u8, response, pos, open_tag) orelse break;
        const content_start = start + open_tag.len;
        const end = std.mem.indexOfPos(u8, response, content_start, close_tag) orelse break;

        const call_json = std.mem.trim(u8, response[content_start..end], " \t\r\n");

        // Parse the JSON to extract name and args
        const parsed = json.parseFromSlice(json.Value, allocator, call_json, .{}) catch {
            pos = end + close_tag.len;
            continue;
        };
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => {
                pos = end + close_tag.len;
                continue;
            },
        };

        const name_val = obj.get("name") orelse {
            pos = end + close_tag.len;
            continue;
        };

        const name_str = switch (name_val) {
            .string => |s| s,
            else => {
                pos = end + close_tag.len;
                continue;
            },
        };

        // Serialize args back to JSON string
        const args_val = obj.get("args");
        var args_json: []u8 = undefined;
        if (args_val) |av| {
            args_json = json.Stringify.valueAlloc(allocator, av, .{}) catch {
                pos = end + close_tag.len;
                continue;
            };
        } else {
            args_json = try allocator.dupe(u8, "{}");
        }
        errdefer allocator.free(args_json);

        const name_copy = try allocator.dupe(u8, name_str);
        errdefer allocator.free(name_copy);

        try calls.append(allocator, .{
            .name = name_copy,
            .args_json = args_json,
        });

        pos = end + close_tag.len;
    }

    if (calls.items.len == 0) {
        if (try parseStandaloneToolCall(response, allocator)) |call| {
            try calls.append(allocator, call);
        }
    }

    return calls;
}

fn parseStandaloneToolCall(response: []const u8, allocator: std.mem.Allocator) !?ToolCallRequest {
    const normalized = normalizeStandaloneJson(response);
    if (normalized.len == 0) return null;
    if (normalized[0] != '{') return null;

    const parsed = json.parseFromSlice(json.Value, allocator, normalized, .{}) catch return null;
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return null,
    };

    const name_val = obj.get("name") orelse return null;
    const name = switch (name_val) {
        .string => |s| s,
        else => return null,
    };

    var args_json: []u8 = undefined;
    if (obj.get("args")) |args_val| {
        args_json = json.Stringify.valueAlloc(allocator, args_val, .{}) catch return null;
    } else {
        args_json = try allocator.dupe(u8, "{}");
    }
    errdefer allocator.free(args_json);

    const name_copy = try allocator.dupe(u8, name);
    errdefer allocator.free(name_copy);

    return .{
        .name = name_copy,
        .args_json = args_json,
    };
}

fn normalizeStandaloneJson(response: []const u8) []const u8 {
    var trimmed = std.mem.trim(u8, response, " \t\r\n");
    if (!std.mem.startsWith(u8, trimmed, "```")) return trimmed;

    const first_nl = std.mem.indexOfScalar(u8, trimmed, '\n') orelse return trimmed;
    trimmed = std.mem.trim(u8, trimmed[first_nl + 1 ..], " \t\r\n");

    if (std.mem.endsWith(u8, trimmed, "```")) {
        trimmed = std.mem.trim(u8, trimmed[0 .. trimmed.len - 3], " \t\r\n");
    }

    return trimmed;
}

// ============================================================================
// Tests
// ============================================================================

test "parseToolCalls - single tool call" {
    const allocator = std.testing.allocator;

    const response =
        \\Let me read that file for you.
        \\<tool_call>{"name": "read_file", "args": {"path": "src/abi.zig"}}</tool_call>
    ;

    var calls = try parseToolCalls(response, allocator);
    defer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.args_json);
        }
        calls.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.items.len);
    try std.testing.expectEqualStrings("read_file", calls.items[0].name);
}

test "parseToolCalls - multiple tool calls" {
    const allocator = std.testing.allocator;

    const response =
        \\I'll check the file and then list the directory.
        \\<tool_call>{"name": "read_file", "args": {"path": "build.zig"}}</tool_call>
        \\<tool_call>{"name": "list_dir", "args": {"path": "src/"}}</tool_call>
    ;

    var calls = try parseToolCalls(response, allocator);
    defer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.args_json);
        }
        calls.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 2), calls.items.len);
    try std.testing.expectEqualStrings("read_file", calls.items[0].name);
    try std.testing.expectEqualStrings("list_dir", calls.items[1].name);
}

test "parseToolCalls - no tool calls" {
    const allocator = std.testing.allocator;

    const response = "Here is a plain text response with no tool calls.";

    var calls = try parseToolCalls(response, allocator);
    defer calls.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), calls.items.len);
}

test "parseToolCalls - malformed JSON skipped" {
    const allocator = std.testing.allocator;

    const response =
        \\<tool_call>not valid json</tool_call>
        \\<tool_call>{"name": "shell", "args": {"command": "ls"}}</tool_call>
    ;

    var calls = try parseToolCalls(response, allocator);
    defer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.args_json);
        }
        calls.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.items.len);
    try std.testing.expectEqualStrings("shell", calls.items[0].name);
}

test "parseToolCalls - standalone JSON tool call" {
    const allocator = std.testing.allocator;

    const response = "{\"name\":\"search_files\",\"args\":{\"path\":\"src\",\"recursive\":true}}";

    var calls = try parseToolCalls(response, allocator);
    defer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.args_json);
        }
        calls.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.items.len);
    try std.testing.expectEqualStrings("search_files", calls.items[0].name);
}

test "parseToolCalls - fenced JSON tool call" {
    const allocator = std.testing.allocator;

    const response =
        \\```json
        \\{"name":"read_file","args":{"path":"src/abi.zig"}}
        \\```
    ;

    var calls = try parseToolCalls(response, allocator);
    defer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.args_json);
        }
        calls.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.items.len);
    try std.testing.expectEqualStrings("read_file", calls.items[0].name);
}

test "generateToolDescriptions - empty registry" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const desc = try generateToolDescriptions(&registry, allocator);
    defer allocator.free(desc);

    try std.testing.expectEqual(@as(usize, 0), desc.len);
}

test "generateToolDescriptions - with tools" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const test_fn = struct {
        fn execute(_: *ToolContext, _: json.Value) ToolExecutionError!ToolResult {
            return ToolResult.init(std.testing.allocator, true, "ok");
        }
    }.execute;

    const t = Tool{
        .name = "test_tool",
        .description = "A test tool",
        .parameters = &[_]tools_mod.Parameter{
            .{ .name = "input", .type = .string, .required = true, .description = "Input text" },
        },
        .execute = test_fn,
    };

    try registry.register(&t);

    const desc = try generateToolDescriptions(&registry, allocator);
    defer allocator.free(desc);

    try std.testing.expect(desc.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, desc, "test_tool") != null);
}

test "ToolAgentConfig - defaults" {
    const config = ToolAgentConfig{};
    try std.testing.expectEqual(@as(usize, 10), config.max_tool_iterations);
    try std.testing.expectEqual(@as(usize, 8000), config.tool_result_max_chars);
    try std.testing.expect(config.require_confirmation);
    try std.testing.expect(!config.enable_memory);
    try std.testing.expect(!config.enable_reflection);
}

test "ToolAugmentedAgent - init and deinit" {
    const allocator = std.testing.allocator;
    var agent = try ToolAugmentedAgent.init(allocator, .{
        .agent = .{ .name = "test-agent" },
    });
    defer agent.deinit();

    try std.testing.expectEqual(@as(usize, 0), agent.toolCount());
    try std.testing.expectEqual(@as(usize, 0), agent.getToolCallLog().len);
}

test "ToolAugmentedAgent - isDestructiveTool" {
    const allocator = std.testing.allocator;
    var agent = try ToolAugmentedAgent.init(allocator, .{
        .agent = .{ .name = "test-agent" },
    });
    defer agent.deinit();

    try std.testing.expect(agent.isDestructiveTool("shell"));
    try std.testing.expect(agent.isDestructiveTool("write_file"));
    try std.testing.expect(!agent.isDestructiveTool("read_file"));
    try std.testing.expect(!agent.isDestructiveTool("system_info"));
}

test {
    std.testing.refAllDecls(@This());
}
