//! Agent Tools stub — active when AI feature is disabled.

const std = @import("std");

// Sub-module stubs
pub const tool = struct {
    pub const Tool = OuterTool;
    pub const ToolResult = OuterToolResult;
    pub const ToolRegistry = OuterToolRegistry;
    pub const Context = OuterContext;
    pub const Parameter = OuterParameter;
    pub const ParameterType = OuterParameterType;
    pub const ToolExecutionError = OuterToolExecutionError;
    pub const createContext = outerCreateContext;

    const OuterParameterType = enum { string, integer, float, boolean, array };

    const OuterParameter = struct {
        name: []const u8 = "",
        param_type: OuterParameterType = .string,
        required: bool = false,
        description: []const u8 = "",
    };

    const OuterToolResult = struct {
        success: bool = false,
        output: []const u8 = "",
        error_message: ?[]const u8 = null,
    };

    const OuterTool = struct {
        name: []const u8 = "",
        description: []const u8 = "",
        parameters: []const OuterParameter = &.{},
    };

    const OuterToolExecutionError = error{ FeatureDisabled, ToolNotFound, InvalidParameters, ExecutionFailed };

    const OuterToolRegistry = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) OuterToolRegistry {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *OuterToolRegistry) void {}
        pub fn get(_: *OuterToolRegistry, _: []const u8) ?*const OuterTool {
            return null;
        }
    };

    const OuterContext = struct {};
    fn outerCreateContext() OuterContext {
        return .{};
    }
};

pub const task = struct {
    pub const TaskTool = OuterTaskTool;
    pub const Subagent = OuterSubagent;
    pub const SubagentConfig = OuterSubagentConfig;
    pub const Task = OuterTask;
    pub const TaskStatus = OuterTaskStatus;

    const OuterTaskStatus = enum { pending, running, completed, failed };
    const OuterTask = struct { id: []const u8 = "", status: OuterTaskStatus = .pending, description: []const u8 = "" };
    const OuterSubagentConfig = struct { name: []const u8 = "", max_steps: usize = 10 };
    const OuterSubagent = struct {};
    const OuterTaskTool = struct {};
};

pub const discord_tools = struct {
    pub fn registerAll(_: *tool.OuterToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const os_tools = struct {
    pub fn registerAll(_: *tool.OuterToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const file_tools = struct {
    pub fn registerAll(_: *tool.OuterToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const search_tools = struct {
    pub fn registerAll(_: *tool.OuterToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const edit_tools = struct {
    pub fn registerAll(_: *tool.OuterToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const process_tools = struct {
    pub fn registerAll(_: *tool.OuterToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const network_tools = struct {
    pub fn registerAll(_: *tool.OuterToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const system_tools = struct {
    pub fn registerAll(_: *tool.OuterToolRegistry) !void {
        return error.FeatureDisabled;
    }
};

// Top-level re-exports
pub const Tool = tool.OuterTool;
pub const ToolResult = tool.OuterToolResult;
pub const ToolRegistry = tool.OuterToolRegistry;
pub const Context = tool.OuterContext;
pub const Parameter = tool.OuterParameter;
pub const ParameterType = tool.OuterParameterType;
pub const createContext = tool.outerCreateContext;
pub const ToolExecutionError = tool.OuterToolExecutionError;
pub fn hasPathTraversal(_: []const u8) bool {
    return false;
}

pub const TaskTool = task.OuterTaskTool;
pub const Subagent = task.OuterSubagent;
pub const SubagentConfig = task.OuterSubagentConfig;
pub const Task = task.OuterTask;
pub const TaskStatus = task.OuterTaskStatus;

pub const DiscordTools = discord_tools;
pub const registerDiscordTools = discord_tools.registerAll;
pub const OsTools = os_tools;
pub const registerOsTools = os_tools.registerAll;
pub const FileTools = file_tools;
pub const registerFileTools = file_tools.registerAll;
pub const SearchTools = search_tools;
pub const registerSearchTools = search_tools.registerAll;
pub const EditTools = edit_tools;
pub const registerEditTools = edit_tools.registerAll;
pub const ProcessTools = process_tools;
pub const registerProcessTools = process_tools.registerAll;
pub const NetworkTools = network_tools;
pub const registerNetworkTools = network_tools.registerAll;
pub const SystemTools = system_tools;
pub const registerSystemTools = system_tools.registerAll;

pub fn registerCodeAgentTools(_: *ToolRegistry) !void {
    return error.FeatureDisabled;
}
pub fn registerAllAgentTools(_: *ToolRegistry) !void {
    return error.FeatureDisabled;
}

// ---------------------------------------------------------------------------
// ToolAgent types (merged from stubs/tool_agent.zig)
// ---------------------------------------------------------------------------

const agent_stub = @import("../agents/stub.zig");

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

pub const ToolAugmentedAgent = struct {
    allocator: std.mem.Allocator = undefined,
    agent: agent_stub.Agent = undefined,
    tool_registry: tool.OuterToolRegistry = undefined,
    config: ToolAgentConfig = .{},
    confirmation_callback: ?ConfirmationFn = null,
    tool_call_log: std.ArrayListUnmanaged(ToolCallRecord) = .{},
    tool_descriptions: ?[]u8 = null,

    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: ToolAgentConfig) error{FeatureDisabled}!Self {
        return error.FeatureDisabled;
    }

    pub fn deinit(_: *Self) void {}

    pub fn registerCodeAgentTools(_: *Self) !void {
        return error.FeatureDisabled;
    }

    pub fn registerAllAgentTools(_: *Self) !void {
        return error.FeatureDisabled;
    }

    pub fn registerTool(_: *Self, _: anytype) !void {
        return error.FeatureDisabled;
    }

    pub fn setConfirmationCallback(_: *Self, _: ConfirmationFn) void {}

    pub fn processWithTools(_: *Self, _: []const u8, _: std.mem.Allocator) error{FeatureDisabled}![]u8 {
        return error.FeatureDisabled;
    }

    pub fn getToolCallLog(_: *const Self) []const ToolCallRecord {
        return &.{};
    }

    pub fn toolCount(_: *const Self) usize {
        return 0;
    }

    pub fn clearLog(_: *Self) void {}
};

pub fn generateToolDescriptions(_: anytype, _: std.mem.Allocator) error{FeatureDisabled}![]u8 {
    return error.FeatureDisabled;
}

pub fn parseToolCalls(_: []const u8, _: std.mem.Allocator) error{FeatureDisabled}!std.ArrayListUnmanaged(ToolCallRequest) {
    return error.FeatureDisabled;
}
