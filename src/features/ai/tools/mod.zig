//! Agent Tools Module
//!
//! Provides tool infrastructure for AI agents including:
//! - Tool registry for registering and executing tools
//! - Built-in tools (file, search, edit, OS operations)
//! - Discord integration tools
//! - Task management and subagent coordination

const std = @import("std");

pub const tool = @import("tool");
pub const task = @import("task");
pub const discord_tools = @import("discord");
pub const os_tools = @import("os_tools");
pub const file_tools = @import("file_tools");
pub const search_tools = @import("search_tools");
pub const edit_tools = @import("edit_tools");
pub const process_tools = @import("process_tools");
pub const network_tools = @import("network_tools");
pub const system_tools = @import("system_tools");
pub const mcp_tools = @import("mcp_tools");
pub const deep_research = @import("deep_research");

pub const Tool = tool.Tool;
pub const ToolResult = tool.ToolResult;
pub const ToolRegistry = tool.ToolRegistry;
pub const Context = tool.Context;
pub const Parameter = tool.Parameter;
pub const ParameterType = tool.ParameterType;
pub const createContext = tool.createContext;
pub const ToolExecutionError = tool.ToolExecutionError;
pub const hasPathTraversal = tool.hasPathTraversal;

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

// Process tool exports
pub const ProcessTools = process_tools;
pub const registerProcessTools = process_tools.registerAll;

// Network tool exports
pub const NetworkTools = network_tools;
pub const registerNetworkTools = network_tools.registerAll;

// System tool exports
pub const SystemTools = system_tools;
pub const registerSystemTools = system_tools.registerAll;

// MCP tool exports
pub const McpTools = mcp_tools;
pub const registerMcpTools = mcp_tools.registerAll;

// Deep Research tool exports
pub const DeepResearchTools = deep_research;
pub const registerDeepResearchTools = deep_research.registerAll;

/// Register all Claude-Code-like tools (file, search, edit, os) with a registry
pub fn registerCodeAgentTools(registry: *ToolRegistry) !void {
    try file_tools.registerAll(registry);
    try search_tools.registerAll(registry);
    try edit_tools.registerAll(registry);
    try os_tools.registerAll(registry);
}

/// Register all agent tools including extended OS capabilities
pub fn registerAllAgentTools(registry: *ToolRegistry) !void {
    try registerCodeAgentTools(registry);
    try process_tools.registerAll(registry);
    try network_tools.registerAll(registry);
    try system_tools.registerAll(registry);
    try mcp_tools.registerAll(registry);
    try deep_research.registerAll(registry);
}

test {
    std.testing.refAllDecls(@This());
}
