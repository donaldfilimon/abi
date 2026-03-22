//! Agent Tools stub — active when AI feature is disabled.

const std = @import("std");
const types = @import("types.zig");
const agent_stub = @import("../agents/stub.zig");

// Re-export types
pub const ParameterType = types.ParameterType;
pub const Parameter = types.Parameter;
pub const ToolResult = types.ToolResult;
pub const Tool = types.Tool;
pub const ToolExecutionError = types.ToolExecutionError;
pub const Context = types.Context;
pub const TaskStatus = types.TaskStatus;
pub const Task = types.Task;
pub const SubagentConfig = types.SubagentConfig;
pub const Subagent = types.Subagent;
pub const TaskTool = types.TaskTool;
pub const ToolAgentConfig = types.ToolAgentConfig;
pub const ToolCallRequest = types.ToolCallRequest;
pub const ToolCallRecord = types.ToolCallRecord;
pub const ConfirmationFn = types.ConfirmationFn;

// --- Tool Registry ---
pub const ToolRegistry = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) ToolRegistry {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *ToolRegistry) void {}
    pub fn get(_: *ToolRegistry, _: []const u8) ?*const Tool {
        return null;
    }
};

pub fn createContext() Context {
    return .{};
}
pub fn hasPathTraversal(_: []const u8) bool {
    return false;
}

// --- Sub-module stubs ---
pub const tool = struct {
    pub const Tool_ = Tool;
    pub const ToolResult_ = ToolResult;
    pub const ToolRegistry_ = ToolRegistry;
    pub const Context_ = Context;
    pub const Parameter_ = Parameter;
    pub const ParameterType_ = ParameterType;
    pub const ToolExecutionError_ = ToolExecutionError;
    pub const createContext_ = createContext;
    pub const OuterTool = Tool;
    pub const OuterToolResult = ToolResult;
    pub const OuterToolRegistry = ToolRegistry;
    pub const OuterContext = Context;
    pub const OuterParameter = Parameter;
    pub const OuterParameterType = ParameterType;
    pub const OuterToolExecutionError = ToolExecutionError;
    pub const outerCreateContext = createContext;
};

pub const task = struct {
    pub const TaskTool_ = TaskTool;
    pub const Subagent_ = Subagent;
    pub const SubagentConfig_ = SubagentConfig;
    pub const Task_ = Task;
    pub const TaskStatus_ = TaskStatus;
    pub const OuterTaskTool = TaskTool;
    pub const OuterSubagent = Subagent;
    pub const OuterSubagentConfig = SubagentConfig;
    pub const OuterTask = Task;
    pub const OuterTaskStatus = TaskStatus;
};

pub const discord_tools = struct {
    pub fn registerAll(_: *ToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const os_tools = struct {
    pub fn registerAll(_: *ToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const file_tools = struct {
    pub fn registerAll(_: *ToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const search_tools = struct {
    pub fn registerAll(_: *ToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const edit_tools = struct {
    pub fn registerAll(_: *ToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const process_tools = struct {
    pub fn registerAll(_: *ToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const network_tools = struct {
    pub fn registerAll(_: *ToolRegistry) !void {
        return error.FeatureDisabled;
    }
};
pub const system_tools = struct {
    pub fn registerAll(_: *ToolRegistry) !void {
        return error.FeatureDisabled;
    }
};

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

// --- ToolAugmentedAgent ---
pub const ToolAugmentedAgent = struct {
    allocator: std.mem.Allocator = undefined,
    agent: agent_stub.Agent = undefined,
    tool_registry: ToolRegistry = undefined,
    config: ToolAgentConfig = .{},
    confirmation_callback: ?ConfirmationFn = null,
    tool_call_log: std.ArrayListUnmanaged(ToolCallRecord) = .empty,
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

test {
    std.testing.refAllDecls(@This());
}
