const std = @import("std");
const tool = @import("tool");
const Tool = tool.Tool;
const ToolResult = tool.ToolResult;
const Context = tool.Context;
const Parameter = tool.Parameter;
const json = std.json;
const os = @import("shared_services").os;
const sync = @import("shared_services").sync;

// Registry for tracking spawned background server PIDs
var server_pids: std.AutoHashMapUnmanaged(u32, []const u8) = .{};
var pid_mutex: sync.Mutex = .{};

fn executeServeMcp(ctx: *Context, args: json.Value) tool.ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const target = if (obj.get("target")) |v| switch (v) {
        .string => |s| s,
        else => "database",
    } else "database";

    const bg_cmd = std.fmt.allocPrint(ctx.allocator, "nohup abi mcp serve {s} > /tmp/abi-mcp.log 2>&1 & echo $!", .{target}) catch
        return error.OutOfMemory;
    defer ctx.allocator.free(bg_cmd);

    var result = os.exec(ctx.allocator, bg_cmd) catch {
        return ToolResult.fromError(ctx.allocator, "Failed to spawn MCP server");
    };
    defer result.deinit();

    // Parse PID from stdout
    const pid_str = std.mem.trim(u8, result.stdout, " \r\n");
    const pid = std.fmt.parseInt(u32, pid_str, 10) catch 0;

    if (pid > 0) {
        pid_mutex.lock();
        defer pid_mutex.unlock();
        server_pids.put(ctx.allocator, pid, "mcp") catch {};
    }

    const output = std.fmt.allocPrint(ctx.allocator, "MCP server started for {s} (PID: {d})", .{ target, pid }) catch
        return error.OutOfMemory;

    return ToolResult.init(ctx.allocator, true, output);
}

pub const serve_mcp_tool = Tool{
    .name = "serve_mcp",
    .description = "Start an MCP (Model Context Protocol) server",
    .parameters = &[_]Parameter{
        .{ .name = "target", .type = .string, .required = false, .description = "Target service (e.g., database, zls)" },
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

    const bg_cmd = std.fmt.allocPrint(ctx.allocator, "nohup abi acp serve --port {d} > /tmp/abi-acp.log 2>&1 & echo $!", .{port}) catch
        return error.OutOfMemory;
    defer ctx.allocator.free(bg_cmd);

    var result = os.exec(ctx.allocator, bg_cmd) catch {
        return ToolResult.fromError(ctx.allocator, "Failed to spawn ACP server");
    };
    defer result.deinit();

    // Parse PID from stdout
    const pid_str = std.mem.trim(u8, result.stdout, " \r\n");
    const pid = std.fmt.parseInt(u32, pid_str, 10) catch 0;

    if (pid > 0) {
        pid_mutex.lock();
        defer pid_mutex.unlock();
        server_pids.put(ctx.allocator, pid, "acp") catch {};
    }

    const output = std.fmt.allocPrint(ctx.allocator, "ACP server started on port {d} (PID: {d})", .{ port, pid }) catch
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

fn executeKillServer(ctx: *Context, args: json.Value) tool.ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const pid = if (obj.get("pid")) |v| switch (v) {
        .integer => |i| i,
        else => return ToolResult.fromError(ctx.allocator, "Expected integer pid"),
    } else return ToolResult.fromError(ctx.allocator, "Missing pid parameter");

    pid_mutex.lock();
    const removed = server_pids.remove(@intCast(pid));
    pid_mutex.unlock();

    if (!removed) {
        return ToolResult.fromError(ctx.allocator, "Server PID not found in registry");
    }

    const cmd = std.fmt.allocPrint(ctx.allocator, "kill -9 {d}", .{pid}) catch return error.OutOfMemory;
    defer ctx.allocator.free(cmd);

    var result = os.exec(ctx.allocator, cmd) catch {
        return ToolResult.fromError(ctx.allocator, "Failed to execute kill command");
    };
    defer result.deinit();

    const output = std.fmt.allocPrint(ctx.allocator, "Terminated server (PID: {d})", .{pid}) catch return error.OutOfMemory;
    return ToolResult.init(ctx.allocator, true, output);
}

pub const kill_server_tool = Tool{
    .name = "kill_server",
    .description = "Terminate an MCP or ACP server by PID",
    .parameters = &[_]Parameter{
        .{ .name = "pid", .type = .integer, .required = true, .description = "PID of the server to terminate" },
    },
    .execute = &executeKillServer,
};

pub const all_tools = [_]*const Tool{
    &serve_mcp_tool,
    &serve_acp_tool,
    &kill_server_tool,
};

pub fn registerAll(registry: *tool.ToolRegistry) !void {
    for (all_tools) |t| {
        try registry.register(t);
    }
}
