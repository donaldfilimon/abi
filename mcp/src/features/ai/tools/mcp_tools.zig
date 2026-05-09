const std = @import("std");
const tool = @import("tool.zig");
const Tool = tool.Tool;
const ToolResult = tool.ToolResult;
const Context = tool.Context;
const Parameter = tool.Parameter;
const json = std.json;
const os = @import("../../../foundation/mod.zig").os;
const sync = @import("../../../foundation/mod.zig").sync;
const arg = @import("args.zig");

// Registry for tracking spawned background server PIDs
var server_pids: std.AutoHashMapUnmanaged(u32, []const u8) = .empty;
var pid_mutex: sync.Mutex = .{};

fn validMcpMode(mode: []const u8) bool {
    return std.mem.eql(u8, mode, "sse") or std.mem.eql(u8, mode, "stdio");
}

fn mcpModeFromArgs(obj: std.json.ObjectMap) []const u8 {
    if (arg.string(obj, "mode")) |mode| {
        if (validMcpMode(mode)) return mode;
    }
    if (arg.string(obj, "target")) |target| {
        if (validMcpMode(target)) return target;
    }
    return "sse";
}

fn executeServeMcp(ctx: *Context, args: json.Value) tool.ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const mode = mcpModeFromArgs(obj);
    if (std.mem.eql(u8, mode, "stdio")) {
        return ToolResult.fromError(ctx.allocator, "Background MCP serving requires mode=sse; use abi-mcp stdio from an MCP client for stdio transport");
    }

    const host = arg.string(obj, "host") orelse "127.0.0.1";
    if (!arg.safeHost(host)) {
        return ToolResult.fromError(ctx.allocator, "Invalid MCP host");
    }

    const port = arg.u16OrDefault(obj, "port", 8081) orelse {
        return ToolResult.fromError(ctx.allocator, "Invalid MCP port");
    };

    const bg_cmd = std.fmt.allocPrint(ctx.allocator, "ABI_MCP_HOST={s} ABI_MCP_PORT={d} nohup ./mcp/launcher.sh sse > /tmp/abi-mcp-{d}.log 2>&1 & echo $!", .{ host, port, port }) catch
        return error.OutOfMemory;
    defer ctx.allocator.free(bg_cmd);

    var result = os.exec(ctx.allocator, bg_cmd) catch {
        return ToolResult.fromError(ctx.allocator, "Failed to spawn MCP server");
    };
    defer result.deinit();

    if (!result.success()) {
        std.log.warn("Failed to spawn MCP server: {s}", .{result.stderr});
        return ToolResult.fromError(ctx.allocator, "Failed to spawn MCP server");
    }

    // Parse PID from stdout
    const pid_str = std.mem.trim(u8, result.stdout, " \r\n");
    const pid = std.fmt.parseInt(u32, pid_str, 10) catch 0;

    if (pid > 0) {
        pid_mutex.lock();
        defer pid_mutex.unlock();
        server_pids.put(ctx.allocator, pid, "mcp") catch |err| {
            std.log.warn("Failed to track MCP server PID {d}: {t}", .{ pid, err });
        };
    }

    const output = std.fmt.allocPrint(ctx.allocator, "MCP SSE server started on {s}:{d} (PID: {d})", .{ host, port, pid }) catch
        return error.OutOfMemory;

    return ToolResult.init(ctx.allocator, true, output);
}

pub const serve_mcp_tool = Tool{
    .name = "serve_mcp",
    .description = "Start the ABI MCP SSE server",
    .parameters = &[_]Parameter{
        .{ .name = "mode", .type = .string, .required = false, .description = "Transport mode; background serving supports sse", .enum_values = &[_][]const u8{"sse"} },
        .{ .name = "host", .type = .string, .required = false, .description = "Bind host, default 127.0.0.1" },
        .{ .name = "port", .type = .integer, .required = false, .description = "Bind port, default 8081" },
    },
    .execute = &executeServeMcp,
};

fn executeServeAcp(ctx: *Context, args: json.Value) tool.ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const port = arg.u16OrDefault(obj, "port", 8080) orelse {
        return ToolResult.fromError(ctx.allocator, "Invalid ACP port");
    };

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
        server_pids.put(ctx.allocator, pid, "acp") catch |err| {
            std.log.warn("Failed to track ACP server PID {d}: {t}", .{ pid, err });
        };
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

test {
    std.testing.refAllDecls(@This());
}
