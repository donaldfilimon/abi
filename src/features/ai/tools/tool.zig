const std = @import("std");
const json = std.json;

pub const ParameterType = enum {
    string,
    integer,
    boolean,
    array,
    object,
    number,
};

pub const Parameter = struct {
    name: []const u8,
    type: ParameterType,
    required: bool = false,
    description: []const u8 = "",
    enum_values: ?[]const []const u8 = null,
};

pub const ToolResult = struct {
    allocator: std.mem.Allocator,
    success: bool,
    output: []const u8,
    error_message: ?[]const u8,
    metadata: ?json.Object,

    pub fn init(allocator: std.mem.Allocator, success_val: bool, output_val: []const u8) ToolResult {
        return ToolResult{
            .allocator = allocator,
            .success = success_val,
            .output = output_val,
            .error_message = null,
            .metadata = null,
        };
    }

    pub fn fromError(allocator: std.mem.Allocator, err: []const u8) ToolResult {
        return ToolResult{
            .allocator = allocator,
            .success = false,
            .output = "",
            .error_message = err,
            .metadata = null,
        };
    }

    pub fn deinit(self: *ToolResult) void {
        if (self.metadata) |*obj| {
            obj.deinit();
        }
    }
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    working_directory: []const u8,
    environment: ?*const std.process.EnvironmentMap,
    cancellation: ?*const std.atomic.Value(bool),
};

pub fn createContext(allocator: std.mem.Allocator, wd: []const u8) Context {
    return Context{
        .allocator = allocator,
        .working_directory = wd,
        .environment = null,
        .cancellation = null,
    };
}

pub const ToolExecutionError = error{
    InvalidArguments,
    ExecutionFailed,
    Timeout,
    Cancelled,
    PermissionDenied,
    FileNotFound,
    ToolNotFound,
    InvalidState,
};

pub const Tool = struct {
    name: []const u8,
    description: []const u8,
    parameters: []const Parameter,
    execute: *const fn (*Context, json.Value) anyerror!ToolResult,
};

pub const ToolRegistry = struct {
    allocator: std.mem.Allocator,
    tools: std.StringHashMap(*const Tool),

    pub fn init(allocator: std.mem.Allocator) ToolRegistry {
        return ToolRegistry{
            .allocator = allocator,
            .tools = std.StringHashMap(*const Tool).init(allocator),
        };
    }

    pub fn deinit(self: *ToolRegistry) void {
        self.tools.deinit();
    }

    pub fn register(self: *ToolRegistry, tool: *const Tool) !void {
        try self.tools.put(tool.name, tool);
    }

    pub fn get(self: *ToolRegistry, name: []const u8) ?*const Tool {
        return self.tools.get(name);
    }
};
