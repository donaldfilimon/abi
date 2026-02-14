//! Network Inspection Tools for Agent Actions
//!
//! Provides tools for inspecting network state:
//! - List open ports and listeners
//! - DNS lookups
//! - Active network connections

const std = @import("std");
const json = std.json;
const tool = @import("tool.zig");
const os = @import("../../../services/shared/os.zig");

const Tool = tool.Tool;
const ToolResult = tool.ToolResult;
const ToolRegistry = tool.ToolRegistry;
const Context = tool.Context;
const Parameter = tool.Parameter;
const ToolExecutionError = tool.ToolExecutionError;

// ============================================================================
// List Ports Tool
// ============================================================================

fn executeListPorts(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    _ = args;
    var result = os.exec(ctx.allocator, "ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null || echo 'Neither ss nor netstat available'") catch {
        return ToolResult.fromError(ctx.allocator, "Failed to list ports");
    };
    defer ctx.allocator.free(result.stderr);

    if (result.success()) {
        return ToolResult.init(ctx.allocator, true, result.stdout);
    }
    ctx.allocator.free(result.stdout);
    return ToolResult.fromError(ctx.allocator, "Port listing failed");
}

pub const list_ports_tool = Tool{
    .name = "list_ports",
    .description = "List open TCP ports and listeners",
    .parameters = &[_]Parameter{},
    .execute = &executeListPorts,
};

// ============================================================================
// DNS Lookup Tool
// ============================================================================

fn executeDnsLookup(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const host_val = obj.get("hostname") orelse {
        return ToolResult.fromError(ctx.allocator, "Missing required parameter: hostname");
    };
    const hostname = switch (host_val) {
        .string => |s| s,
        else => return ToolResult.fromError(ctx.allocator, "hostname must be a string"),
    };

    // Validate hostname to prevent command injection
    for (hostname) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '.' and c != '-') {
            return ToolResult.fromError(ctx.allocator, "Invalid hostname characters");
        }
    }

    const cmd = std.fmt.allocPrint(ctx.allocator, "dig +short {s} 2>/dev/null || nslookup {s} 2>/dev/null || host {s} 2>/dev/null", .{ hostname, hostname, hostname }) catch
        return error.OutOfMemory;
    defer ctx.allocator.free(cmd);

    var result = os.exec(ctx.allocator, cmd) catch {
        return ToolResult.fromError(ctx.allocator, "DNS lookup failed");
    };
    defer ctx.allocator.free(result.stderr);

    if (result.success()) {
        return ToolResult.init(ctx.allocator, true, result.stdout);
    }
    ctx.allocator.free(result.stdout);
    return ToolResult.fromError(ctx.allocator, "DNS lookup returned no results");
}

pub const dns_lookup_tool = Tool{
    .name = "dns_lookup",
    .description = "Perform DNS lookup for a hostname",
    .parameters = &[_]Parameter{
        .{ .name = "hostname", .type = .string, .required = true, .description = "Hostname to resolve" },
    },
    .execute = &executeDnsLookup,
};

// ============================================================================
// Network Connections Tool
// ============================================================================

fn executeNetworkConnections(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    _ = args;
    var result = os.exec(ctx.allocator, "ss -tunapo 2>/dev/null | head -50 || netstat -tunapo 2>/dev/null | head -50 || echo 'Neither ss nor netstat available'") catch {
        return ToolResult.fromError(ctx.allocator, "Failed to list connections");
    };
    defer ctx.allocator.free(result.stderr);

    if (result.success()) {
        return ToolResult.init(ctx.allocator, true, result.stdout);
    }
    ctx.allocator.free(result.stdout);
    return ToolResult.fromError(ctx.allocator, "Connection listing failed");
}

pub const network_connections_tool = Tool{
    .name = "network_connections",
    .description = "List active network connections (TCP/UDP)",
    .parameters = &[_]Parameter{},
    .execute = &executeNetworkConnections,
};

// ============================================================================
// Registration
// ============================================================================

pub const all_tools = [_]*const Tool{
    &list_ports_tool,
    &dns_lookup_tool,
    &network_connections_tool,
};

pub fn registerAll(registry: *ToolRegistry) !void {
    for (all_tools) |t| {
        try registry.register(t);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "list_ports_tool creation" {
    try std.testing.expectEqualStrings("list_ports", list_ports_tool.name);
}

test "dns_lookup_tool creation" {
    try std.testing.expectEqualStrings("dns_lookup", dns_lookup_tool.name);
    try std.testing.expectEqual(@as(usize, 1), dns_lookup_tool.parameters.len);
}

test "network_connections_tool creation" {
    try std.testing.expectEqualStrings("network_connections", network_connections_tool.name);
}

test "all_tools count" {
    try std.testing.expectEqual(@as(usize, 3), all_tools.len);
}
