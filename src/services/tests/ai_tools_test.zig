//! AI Tools Tests — Path Validation, Registration, Context
//!
//! Tests tool infrastructure: path traversal detection, registry, context.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const tools = if (build_options.enable_ai) abi.ai.tools else struct {};

// ============================================================================
// Path Traversal Detection Tests
// ============================================================================

test "tools: hasPathTraversal detects .. components" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    try std.testing.expect(tools.hasPathTraversal("../etc/passwd"));
    try std.testing.expect(tools.hasPathTraversal("foo/../../../etc/shadow"));
    try std.testing.expect(tools.hasPathTraversal("a/b/.."));
}

test "tools: hasPathTraversal allows normal paths" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    try std.testing.expect(!tools.hasPathTraversal("src/main.zig"));
    try std.testing.expect(!tools.hasPathTraversal("./relative/path"));
    try std.testing.expect(!tools.hasPathTraversal("/absolute/path/file.txt"));
    try std.testing.expect(!tools.hasPathTraversal(""));
}

test "tools: hasPathTraversal detects encoded traversal" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    try std.testing.expect(tools.hasPathTraversal("%2e%2e/etc"));
    try std.testing.expect(tools.hasPathTraversal("foo/%2E%2E/bar"));
}

test "tools: hasPathTraversal detects null bytes" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    try std.testing.expect(tools.hasPathTraversal("safe.txt\x00../evil"));
}

test "tools: hasPathTraversal handles backslash paths" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    try std.testing.expect(tools.hasPathTraversal("foo\\..\\bar"));
}

// ============================================================================
// Tool Registry Tests
// ============================================================================

test "tools: ToolRegistry init and deinit" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var registry = tools.ToolRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expectEqual(@as(usize, 0), registry.count());
}

test "tools: register and retrieve tool" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var registry = tools.ToolRegistry.init(allocator);
    defer registry.deinit();

    const dummy_fn = struct {
        fn execute(_: *tools.Context, _: std.json.Value) tools.ToolExecutionError!tools.ToolResult {
            return tools.ToolResult.init(std.testing.allocator, true, "ok");
        }
    }.execute;

    const test_tool = tools.Tool{
        .name = "test_tool",
        .description = "A test tool",
        .parameters = &[_]tools.Parameter{},
        .execute = dummy_fn,
    };

    try registry.register(&test_tool);
    try std.testing.expectEqual(@as(usize, 1), registry.count());
    try std.testing.expect(registry.contains("test_tool"));
    try std.testing.expect(!registry.contains("nonexistent"));
}

// ============================================================================
// ToolResult Tests
// ============================================================================

test "tools: ToolResult success" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var result = tools.ToolResult.init(allocator, true, "output data");
    defer result.deinit();

    try std.testing.expect(result.success);
    try std.testing.expectEqualStrings("output data", result.output);
    try std.testing.expect(result.error_message == null);
}

test "tools: ToolResult error" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var result = tools.ToolResult.fromError(allocator, "something failed");
    defer result.deinit();

    try std.testing.expect(!result.success);
    try std.testing.expectEqualStrings("something failed", result.error_message.?);
}

// ============================================================================
// Context Tests
// ============================================================================

test "tools: createContext with working directory" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const ctx = tools.createContext(allocator, "/home/user/project");
    try std.testing.expectEqualStrings("/home/user/project", ctx.working_directory);
    try std.testing.expect(ctx.environment == null);
    try std.testing.expect(ctx.cancellation == null);
}
