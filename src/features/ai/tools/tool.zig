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
    metadata: ?json.ObjectMap,

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

/// Environment map type for tool context
pub const EnvMap = std.StringHashMapUnmanaged([]const u8);

pub const Context = struct {
    allocator: std.mem.Allocator,
    working_directory: []const u8,
    environment: ?*const EnvMap,
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

/// Check if a path contains traversal patterns that could escape the working directory.
/// Returns true if the path is unsafe (contains `..`, null bytes, or encoded traversal).
pub fn hasPathTraversal(path: []const u8) bool {
    if (path.len == 0) return false;
    // Null bytes can truncate paths at the OS level
    if (std.mem.indexOfScalar(u8, path, 0) != null) return true;
    // URL-encoded traversal: %2e = '.', %2f = '/'
    if (std.mem.indexOf(u8, path, "%2e") != null) return true;
    if (std.mem.indexOf(u8, path, "%2E") != null) return true;
    // Check for ".." path components
    var it = std.mem.splitScalar(u8, path, '/');
    while (it.next()) |component| {
        if (std.mem.eql(u8, component, "..")) return true;
    }
    // Also check backslash-separated (Windows paths)
    var it2 = std.mem.splitScalar(u8, path, '\\');
    while (it2.next()) |component| {
        if (std.mem.eql(u8, component, "..")) return true;
    }
    return false;
}

pub const ToolExecutionError = error{
    OutOfMemory,
    InvalidArguments,
    ExecutionFailed,
    Timeout,
    Cancelled,
    PermissionDenied,
    FileNotFound,
    ToolNotFound,
    InvalidState,
};

/// Function pointer type for tool execution.
pub const ToolExecuteFn = *const fn (*Context, json.Value) ToolExecutionError!ToolResult;

pub const Tool = struct {
    name: []const u8,
    description: []const u8,
    parameters: []const Parameter,
    execute: ToolExecuteFn,
};

pub const ToolRegistry = struct {
    allocator: std.mem.Allocator,
    tools: std.StringHashMapUnmanaged(*const Tool),

    pub fn init(allocator: std.mem.Allocator) ToolRegistry {
        return ToolRegistry{
            .allocator = allocator,
            .tools = .{},
        };
    }

    pub fn deinit(self: *ToolRegistry) void {
        self.tools.deinit(self.allocator);
    }

    pub fn register(self: *ToolRegistry, tool: *const Tool) !void {
        try self.tools.put(self.allocator, tool.name, tool);
    }

    pub fn get(self: *ToolRegistry, name: []const u8) ?*const Tool {
        return self.tools.get(name);
    }

    /// Get the number of registered tools.
    pub fn count(self: *const ToolRegistry) usize {
        return self.tools.count();
    }

    /// Check if a tool with the given name exists.
    pub fn contains(self: *const ToolRegistry, name: []const u8) bool {
        return self.tools.contains(name);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ToolResult.init creates success result" {
    const allocator = std.testing.allocator;
    var result = ToolResult.init(allocator, true, "output data");

    try std.testing.expect(result.success);
    try std.testing.expectEqualStrings("output data", result.output);
    try std.testing.expect(result.error_message == null);
    try std.testing.expect(result.metadata == null);

    result.deinit();
}

test "ToolResult.fromError creates failure result" {
    const allocator = std.testing.allocator;
    var result = ToolResult.fromError(allocator, "Something went wrong");

    try std.testing.expect(!result.success);
    try std.testing.expectEqualStrings("", result.output);
    try std.testing.expectEqualStrings("Something went wrong", result.error_message.?);
    try std.testing.expect(result.metadata == null);

    result.deinit();
}

test "ToolResult.deinit handles null metadata safely" {
    const allocator = std.testing.allocator;
    var result = ToolResult.init(allocator, true, "test");

    // Should not crash when metadata is null
    result.deinit();
}

test "ToolRegistry - register and retrieve tool" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    // Create a simple test tool
    const testExecuteFn = struct {
        fn execute(_: *Context, _: json.Value) ToolExecutionError!ToolResult {
            return ToolResult.init(std.testing.allocator, true, "executed");
        }
    }.execute;

    const tool = Tool{
        .name = "test_tool",
        .description = "A test tool",
        .parameters = &[_]Parameter{},
        .execute = testExecuteFn,
    };

    try registry.register(&tool);

    const found = registry.get("test_tool");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("test_tool", found.?.name);
    try std.testing.expectEqualStrings("A test tool", found.?.description);
}

test "ToolRegistry - get returns null for non-existent tool" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const found = registry.get("non_existent_tool");
    try std.testing.expect(found == null);
}

test "ToolRegistry - register multiple tools" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const testExecuteFn = struct {
        fn execute(_: *Context, _: json.Value) ToolExecutionError!ToolResult {
            return ToolResult.init(std.testing.allocator, true, "executed");
        }
    }.execute;

    const tool1 = Tool{
        .name = "tool_one",
        .description = "First tool",
        .parameters = &[_]Parameter{},
        .execute = testExecuteFn,
    };

    const tool2 = Tool{
        .name = "tool_two",
        .description = "Second tool",
        .parameters = &[_]Parameter{},
        .execute = testExecuteFn,
    };

    const tool3 = Tool{
        .name = "tool_three",
        .description = "Third tool",
        .parameters = &[_]Parameter{},
        .execute = testExecuteFn,
    };

    try registry.register(&tool1);
    try registry.register(&tool2);
    try registry.register(&tool3);

    try std.testing.expectEqual(@as(usize, 3), registry.count());
    try std.testing.expect(registry.contains("tool_one"));
    try std.testing.expect(registry.contains("tool_two"));
    try std.testing.expect(registry.contains("tool_three"));
    try std.testing.expect(!registry.contains("tool_four"));
}

test "ToolRegistry - overwrite existing tool with same name" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    const testExecuteFn = struct {
        fn execute(_: *Context, _: json.Value) ToolExecutionError!ToolResult {
            return ToolResult.init(std.testing.allocator, true, "executed");
        }
    }.execute;

    const tool1 = Tool{
        .name = "my_tool",
        .description = "Original description",
        .parameters = &[_]Parameter{},
        .execute = testExecuteFn,
    };

    const tool2 = Tool{
        .name = "my_tool",
        .description = "Updated description",
        .parameters = &[_]Parameter{},
        .execute = testExecuteFn,
    };

    try registry.register(&tool1);
    try registry.register(&tool2);

    // Count should still be 1 (overwrites, not duplicates)
    try std.testing.expectEqual(@as(usize, 1), registry.count());

    const found = registry.get("my_tool");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("Updated description", found.?.description);
}

test "Parameter - default values" {
    const param = Parameter{
        .name = "test_param",
        .type = .string,
    };

    try std.testing.expectEqualStrings("test_param", param.name);
    try std.testing.expectEqual(ParameterType.string, param.type);
    try std.testing.expect(!param.required);
    try std.testing.expectEqualStrings("", param.description);
    try std.testing.expect(param.enum_values == null);
}

test "Parameter - with all fields" {
    const enum_vals = [_][]const u8{ "option1", "option2", "option3" };
    const param = Parameter{
        .name = "choice_param",
        .type = .string,
        .required = true,
        .description = "Choose an option",
        .enum_values = &enum_vals,
    };

    try std.testing.expectEqualStrings("choice_param", param.name);
    try std.testing.expectEqual(ParameterType.string, param.type);
    try std.testing.expect(param.required);
    try std.testing.expectEqualStrings("Choose an option", param.description);
    try std.testing.expect(param.enum_values != null);
    try std.testing.expectEqual(@as(usize, 3), param.enum_values.?.len);
}

test "ParameterType - all types are distinct" {
    const types = [_]ParameterType{
        .string,
        .integer,
        .boolean,
        .array,
        .object,
        .number,
    };

    // Verify all types are unique
    for (types, 0..) |t1, i| {
        for (types[i + 1 ..]) |t2| {
            try std.testing.expect(t1 != t2);
        }
    }
}

test "createContext - creates valid context" {
    const allocator = std.testing.allocator;
    const ctx = createContext(allocator, "/home/user/project");

    try std.testing.expectEqualStrings("/home/user/project", ctx.working_directory);
    try std.testing.expect(ctx.environment == null);
    try std.testing.expect(ctx.cancellation == null);
}

test "Context - with environment" {
    const allocator = std.testing.allocator;

    var env = EnvMap{};
    defer env.deinit(allocator);
    try env.put(allocator, "PATH", "/usr/bin");
    try env.put(allocator, "HOME", "/home/user");

    const ctx = Context{
        .allocator = allocator,
        .working_directory = "/work",
        .environment = &env,
        .cancellation = null,
    };

    try std.testing.expect(ctx.environment != null);
    try std.testing.expectEqualStrings("/usr/bin", ctx.environment.?.get("PATH").?);
    try std.testing.expectEqualStrings("/home/user", ctx.environment.?.get("HOME").?);
}

test "Context - with cancellation" {
    const allocator = std.testing.allocator;

    var cancelled = std.atomic.Value(bool).init(false);

    const ctx = Context{
        .allocator = allocator,
        .working_directory = "/work",
        .environment = null,
        .cancellation = &cancelled,
    };

    try std.testing.expect(ctx.cancellation != null);
    try std.testing.expect(!ctx.cancellation.?.load(.seq_cst));

    cancelled.store(true, .seq_cst);
    try std.testing.expect(ctx.cancellation.?.load(.seq_cst));
}

test "ToolExecutionError - all error types exist" {
    // Verify all error types are defined and distinct
    const errors = [_]ToolExecutionError{
        error.OutOfMemory,
        error.InvalidArguments,
        error.ExecutionFailed,
        error.Timeout,
        error.Cancelled,
        error.PermissionDenied,
        error.FileNotFound,
        error.ToolNotFound,
        error.InvalidState,
    };

    // Just verify they all exist and are errors
    for (errors) |err| {
        var err_name_buf: [64]u8 = undefined;
        const err_name = std.fmt.bufPrint(&err_name_buf, "{t}", .{err}) catch "UnknownError";
        try std.testing.expect(err_name.len > 0);
    }
}

test "Tool - with parameters" {
    const testExecuteFn = struct {
        fn execute(_: *Context, _: json.Value) ToolExecutionError!ToolResult {
            return ToolResult.init(std.testing.allocator, true, "done");
        }
    }.execute;

    const params = [_]Parameter{
        .{ .name = "input", .type = .string, .required = true, .description = "Input text" },
        .{ .name = "count", .type = .integer, .required = false, .description = "Number of items" },
        .{ .name = "verbose", .type = .boolean, .required = false },
    };

    const tool = Tool{
        .name = "process",
        .description = "Process some data",
        .parameters = &params,
        .execute = testExecuteFn,
    };

    try std.testing.expectEqualStrings("process", tool.name);
    try std.testing.expectEqual(@as(usize, 3), tool.parameters.len);
    try std.testing.expectEqualStrings("input", tool.parameters[0].name);
    try std.testing.expect(tool.parameters[0].required);
    try std.testing.expect(!tool.parameters[1].required);
}

test "ToolRegistry - empty registry" {
    const allocator = std.testing.allocator;
    var registry = ToolRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expectEqual(@as(usize, 0), registry.count());
    try std.testing.expect(!registry.contains("any_tool"));
    try std.testing.expect(registry.get("any_tool") == null);
}
