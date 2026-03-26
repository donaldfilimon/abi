//! Agent Tools stub types — extracted from stub.zig.

const std = @import("std");
const agent_stub = @import("../agents/stub.zig");

// --- Tool Core Types ---

pub const ParameterType = enum { string, integer, float, boolean, array };

pub const Parameter = struct {
    name: []const u8 = "",
    param_type: ParameterType = .string,
    required: bool = false,
    description: []const u8 = "",
};

pub const ToolResult = struct {
    success: bool = false,
    output: []const u8 = "",
    error_message: ?[]const u8 = null,
};

pub const Tool = struct {
    name: []const u8 = "",
    description: []const u8 = "",
    parameters: []const Parameter = &.{},
};

pub const ToolExecutionError = error{ FeatureDisabled, ToolNotFound, InvalidParameters, ExecutionFailed };

pub const Context = struct {};

// --- Task Types ---

pub const TaskStatus = enum { pending, running, completed, failed };
pub const Task = struct { id: []const u8 = "", status: TaskStatus = .pending, description: []const u8 = "" };
pub const SubagentConfig = struct { name: []const u8 = "", max_steps: usize = 10 };
pub const Subagent = struct {};
pub const TaskTool = struct {};

// --- ToolAgent Types ---

pub const ToolAgentConfig = struct {
    agent: agent_stub.AgentConfig = .{ .name = "tool-agent" },
    max_tool_iterations: usize = 10,
    tool_result_max_chars: usize = 8000,
    require_confirmation: bool = true,
    destructive_tools: []const []const u8 = &.{},
    enable_memory: bool = false,
    enable_reflection: bool = false,
    working_directory: []const u8 = ".",
};

pub const ToolCallRequest = struct {
    name: []const u8,
    args_json: []const u8,
};

pub const ToolCallRecord = struct {
    tool_name: []const u8 = "",
    args_summary: []const u8 = "",
    success: bool = false,
    output_preview: []const u8 = "",
};

pub const ConfirmationFn = *const fn ([]const u8, []const u8) bool;
