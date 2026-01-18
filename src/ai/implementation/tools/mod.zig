pub const tool = @import("tool.zig");
pub const task = @import("task.zig");
pub const discord_tools = @import("discord.zig");

pub const Tool = tool.Tool;
pub const ToolResult = tool.ToolResult;
pub const ToolRegistry = tool.ToolRegistry;
pub const Context = tool.Context;
pub const Parameter = tool.Parameter;
pub const ParameterType = tool.ParameterType;

pub const TaskTool = task.TaskTool;
pub const Subagent = task.Subagent;
pub const SubagentConfig = task.SubagentConfig;
pub const Task = task.Task;
pub const TaskStatus = task.TaskStatus;

// Discord tool exports
pub const DiscordTools = discord_tools;
pub const registerDiscordTools = discord_tools.registerAll;
