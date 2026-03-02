const std = @import("std");
const tool = @import("tool.zig");
const Tool = tool.Tool;
const ToolResult = tool.ToolResult;
const Context = tool.Context;
const Parameter = tool.Parameter;
const json = std.json;
const os = @import("../../../services/shared/os.zig");

fn executeServeMcp(ctx: *Context, args: json.Value) tool.ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const target = if (obj.get("target")) |v| switch (v) {
        .string => |s| s,
        else => "wdbx",
    } else "wdbx";

    const bg_cmd = std.fmt.allocPrint(ctx.allocator, "nohup abi mcp serve {s} > /tmp/abi-mcp.log 2>&1 &", .{target}) catch
        return error.OutOfMemory;
    defer ctx.allocator.free(bg_cmd);

    var result = os.exec(ctx.allocator, bg_cmd) catch {
        return ToolResult.fromError(ctx.allocator, "Failed to spawn MCP server");
    };
    defer result.deinit();

    const output = std.fmt.allocPrint(ctx.allocator, "MCP server started for: {s}", .{target}) catch
        return error.OutOfMemory;

    return ToolResult.init(ctx.allocator, true, output);
}

pub const serve_mcp_tool = Tool{
    .name = "serve_mcp",
    .description = "Start an MCP (Model Context Protocol) server",
    .parameters = &[_]Parameter{
        .{ .name = "target", .type = .string, .required = false, .description = "Target service (e.g., wdbx, zls)" },
    },
    .execute = &executeServeMcp,
};

fn executeServeAcp(ctx: *Context, args: json.Value) tool.ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const port = if (obj.get("port")) |v| switch (v) {
        .integer => |i| i,
        else => 8080,
    } else 8080;

    const bg_cmd = std.fmt.allocPrint(ctx.allocator, "nohup abi acp serve --port {d} > /tmp/abi-acp.log 2>&1 &", .{port}) catch
        return error.OutOfMemory;
    defer ctx.allocator.free(bg_cmd);

    var result = os.exec(ctx.allocator, bg_cmd) catch {
        return ToolResult.fromError(ctx.allocator, "Failed to spawn ACP server");
    };
    defer result.deinit();

    const output = std.fmt.allocPrint(ctx.allocator, "ACP server started on port: {d}", .{port}) catch
        return error.OutOfMemory;

    return ToolResult.init(ctx.allocator, true, output);
}

pub const serve_acp_tool = Tool{
    .name = "serve_acp",
    .description = "Start an ACP (Agent Communication Protocol) server",
    .parameters = &[_]Parameter{
        .{ .name = "port", .type = .integer, .required = false, .description = "Port to listen on" },
    },
    .execute = &executeServeAcp,
};

pub const all_tools = [_]*const Tool{
    &serve_mcp_tool,
    &serve_acp_tool,
};

pub fn registerAll(registry: *tool.ToolRegistry) !void {
    for (all_tools) |t| {
        try registry.register(t);
    }
}
