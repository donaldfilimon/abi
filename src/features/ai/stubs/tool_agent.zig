const std = @import("std");

pub const ToolAgentConfig = struct {
    agent: AgentConfig = .{},
    max_tool_iterations: usize = 10,
    tool_result_max_chars: usize = 8000,
    require_confirmation: bool = true,
    destructive_tools: []const []const u8 = &.{},
    enable_memory: bool = false,
    enable_reflection: bool = false,
    working_directory: []const u8 = ".",
};

const AgentConfig = struct {
    name: []const u8 = "",
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
    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: ToolAgentConfig) error{AiDisabled}!Self {
        return error.AiDisabled;
    }

    pub fn deinit(_: *Self) void {}

    pub fn registerCodeAgentTools(_: *Self) !void {
        return error.AiDisabled;
    }

    pub fn registerAllAgentTools(_: *Self) !void {
        return error.AiDisabled;
    }

    pub fn registerTool(_: *Self, _: anytype) !void {
        return error.AiDisabled;
    }

    pub fn setConfirmationCallback(_: *Self, _: ConfirmationFn) void {}

    pub fn processWithTools(_: *Self, _: []const u8, _: std.mem.Allocator) error{AiDisabled}![]u8 {
        return error.AiDisabled;
    }

    pub fn getToolCallLog(_: *const Self) []const ToolCallRecord {
        return &.{};
    }

    pub fn toolCount(_: *const Self) usize {
        return 0;
    }

    pub fn clearLog(_: *Self) void {}
};

pub fn generateToolDescriptions(_: anytype, _: std.mem.Allocator) error{AiDisabled}![]u8 {
    return error.AiDisabled;
}

pub fn parseToolCalls(_: []const u8, _: std.mem.Allocator) error{AiDisabled}!std.ArrayListUnmanaged(ToolCallRequest) {
    return error.AiDisabled;
}
