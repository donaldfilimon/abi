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

    const OuterParameterType = enum {
        string,
        integer,
        float,
        boolean,
        array,
    };

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

    const OuterToolExecutionError = error{
        FeatureDisabled,
        ToolNotFound,
        InvalidParameters,
        ExecutionFailed,
    };

    const OuterToolRegistry = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) OuterToolRegistry {
            return .{ .allocator = allocator };
        }

        pub fn deinit(_: *OuterToolRegistry) void {}
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

    const OuterTaskStatus = enum {
        pending,
        running,
        completed,
        failed,
    };

    const OuterTask = struct {
        id: []const u8 = "",
        status: OuterTaskStatus = .pending,
        description: []const u8 = "",
    };

    const OuterSubagentConfig = struct {
        name: []const u8 = "",
        max_steps: usize = 10,
    };

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

// Top-level re-exports
pub const Tool = tool.OuterTool;
pub const ToolResult = tool.OuterToolResult;
pub const ToolRegistry = tool.OuterToolRegistry;
pub const Context = tool.OuterContext;
pub const Parameter = tool.OuterParameter;
pub const ParameterType = tool.OuterParameterType;
pub const createContext = tool.outerCreateContext;
pub const ToolExecutionError = tool.OuterToolExecutionError;
pub fn hasPathTraversal(path: []const u8) bool {
    _ = path;
    return false;
}

pub const TaskTool = task.OuterTaskTool;
pub const Subagent = task.OuterSubagent;
pub const SubagentConfig = task.OuterSubagentConfig;
pub const Task = task.OuterTask;
pub const TaskStatus = task.OuterTaskStatus;

// Discord tool exports
pub const DiscordTools = discord_tools;
pub const registerDiscordTools = discord_tools.registerAll;

// OS tool exports
pub const OsTools = os_tools;
pub const registerOsTools = os_tools.registerAll;

// File tool exports
pub const FileTools = file_tools;
pub const registerFileTools = file_tools.registerAll;

// Search tool exports
pub const SearchTools = search_tools;
pub const registerSearchTools = search_tools.registerAll;

// Edit tool exports
pub const EditTools = edit_tools;
pub const registerEditTools = edit_tools.registerAll;

/// Stub: register all code agent tools.
pub fn registerCodeAgentTools(_: *ToolRegistry) !void {
    return error.FeatureDisabled;
}
