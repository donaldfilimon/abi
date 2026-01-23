pub const tool = @import("tool.zig");
pub const task = @import("task.zig");
pub const discord_tools = @import("discord.zig");
pub const os_tools = @import("os_tools.zig");
pub const file_tools = @import("file_tools.zig");
pub const search_tools = @import("search_tools.zig");
pub const edit_tools = @import("edit_tools.zig");

pub const Tool = tool.Tool;
pub const ToolResult = tool.ToolResult;
pub const ToolRegistry = tool.ToolRegistry;
pub const Context = tool.Context;
pub const Parameter = tool.Parameter;
pub const ParameterType = tool.ParameterType;
pub const createContext = tool.createContext;
pub const ToolExecutionError = tool.ToolExecutionError;

pub const TaskTool = task.TaskTool;
pub const Subagent = task.Subagent;
pub const SubagentConfig = task.SubagentConfig;
pub const Task = task.Task;
pub const TaskStatus = task.TaskStatus;

// Discord tool exports
pub const DiscordTools = discord_tools;
pub const registerDiscordTools = discord_tools.registerAll;

// OS tool exports
pub const OsTools = os_tools;
pub const registerOsTools = os_tools.registerAll;

// File tool exports (Claude-Code-like)
pub const FileTools = file_tools;
pub const registerFileTools = file_tools.registerAll;

// Search tool exports (Claude-Code-like)
pub const SearchTools = search_tools;
pub const registerSearchTools = search_tools.registerAll;

// Edit tool exports (Claude-Code-like)
pub const EditTools = edit_tools;
pub const registerEditTools = edit_tools.registerAll;

/// Register all Claude-Code-like tools (file, search, edit, os) with a registry
pub fn registerCodeAgentTools(registry: *ToolRegistry) !void {
    try file_tools.registerAll(registry);
    try search_tools.registerAll(registry);
    try edit_tools.registerAll(registry);
    try os_tools.registerAll(registry);
}
