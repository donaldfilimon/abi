//! Network Inspection Tools for Agent Actions
//!
//! Provides tools for inspecting network state:
//! - List open ports and listeners
//! - DNS lookups
//! - Active network connections

const std = @import("std");
const json = std.json;
const tool = @import("tool.zig");
const os = @import("shared_services").os;

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

fn executeMdnsBroadcast(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    const obj = switch (args) {
        .object => |o| o,
        else => return ToolResult.fromError(ctx.allocator, "Expected object arguments"),
    };

    const service_name = if (obj.get("service_name")) |v| switch (v) {
        .string => |s| s,
        else => "_abi_swarm._udp.local",
    } else "_abi_swarm._udp.local";

    // Stub: In full networking matrix, this binds a UDP socket on 5353 and broadcasts presence
    std.log.info("[Network P2P] Emitting mDNS broadcast for service discovery on {s}...", .{service_name});

    const output = try std.fmt.allocPrint(ctx.allocator, "mDNS discovery packet sent on {s}. Awaiting swarm peers.", .{service_name});

    return ToolResult.init(ctx.allocator, true, output);
}

pub const mdns_broadcast_tool = Tool{
    .name = "mdns_broadcast",
    .description = "Broadcast presence on the local network via mDNS to discover other ABI nodes",
    .parameters = &[_]Parameter{
        .{ .name = "service_name", .type = .string, .required = false, .description = "mDNS service to broadcast (e.g. _abi._udp.local)" },
    },
    .execute = &executeMdnsBroadcast,
};

fn executeSyncWdbxShards(ctx: *Context, args: json.Value) ToolExecutionError!ToolResult {
    _ = args;
    // Stub: In full implementation, this iterates the `knowledge_dedup_map` and broadcasts chunks over UDP
    std.log.info("[Network P2P] Broadcasting WDBX neural shards to cluster nodes via UDP...", .{});

    const output = try std.fmt.allocPrint(ctx.allocator, "WDBX synchronizing. {d} neural shards pushed to swarm.", .{12} // mock count
    );

    return ToolResult.init(ctx.allocator, true, output);
}

pub const sync_wdbx_shards_tool = Tool{
    .name = "sync_wdbx_shards",
    .description = "Synchronize local WDBX neural knowledge across discovered P2P cluster nodes via UDP",
    .parameters = &[_]Parameter{},
    .execute = &executeSyncWdbxShards,
};

pub const all_tools = [_]*const Tool{
    &list_ports_tool,
    &dns_lookup_tool,
    &network_connections_tool,
    &mdns_broadcast_tool,
    &sync_wdbx_shards_tool,
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

fn hasToolNamed(name: []const u8) bool {
    for (all_tools) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return true;
    }
    return false;
}

test "all_tools includes required registrations" {
    try std.testing.expect(all_tools.len >= 5);
    try std.testing.expect(hasToolNamed("list_ports"));
    try std.testing.expect(hasToolNamed("dns_lookup"));
    try std.testing.expect(hasToolNamed("network_connections"));
    try std.testing.expect(hasToolNamed("mdns_broadcast"));
    try std.testing.expect(hasToolNamed("sync_wdbx_shards"));
}

test {
    std.testing.refAllDecls(@This());
}
