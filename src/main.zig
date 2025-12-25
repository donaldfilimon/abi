const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len <= 1) {
        printHelp();
        return;
    }

    const command = args[1];
    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or
        std.mem.eql(u8, command, "-h"))
    {
        printHelp();
        return;
    }

    if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "--version") or
        std.mem.eql(u8, command, "-v"))
    {
        std.debug.print("ABI Framework v{s}\n", .{abi.version()});
        return;
    }

    if (std.mem.eql(u8, command, "db")) {
        try runDb(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "agent")) {
        try runAgent(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "gpu")) {
        try runGpu(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "network")) {
        try runNetwork(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "system-info")) {
        try runSystemInfo(allocator, &framework);
        return;
    }

    std.debug.print("Unknown command: {s}\nUse 'help' for usage.\n", .{command});
    std.process.exit(1);
}

fn printHelp() void {
    const help_text =
        "Usage: abi <command> [options]\n\n" ++
        "Commands:\n" ++
        "  db <subcommand>   Database operations (add, query, stats, optimize, backup)\n" ++
        "  agent [--message] Run AI agent (interactive or one-shot)\n" ++
        "  gpu [subcommand]  Show GPU backends or devices\n" ++
        "  network [command] Manage network registry (list, register, status)\n" ++
        "  system-info       Show system and framework status\n" ++
        "  version           Show framework version\n" ++
        "  help              Show this help message\n\n" ++
        "Run 'abi db help' for database specific commands.\n" ++
        "Run 'abi gpu help' for GPU commands.\n" ++
        "Run 'abi network help' for network commands.\n";

    std.debug.print("{s}", .{help_text});
}

fn runSystemInfo(allocator: std.mem.Allocator, framework: *abi.Framework) !void {
    const platform = abi.platform.platform;
    const info = platform.PlatformInfo.detect();

    std.debug.print("System Info\n", .{});
    std.debug.print("  OS: {s}\n", .{@tagName(info.os)});
    std.debug.print("  Arch: {s}\n", .{@tagName(info.arch)});
    std.debug.print("  CPU Threads: {d}\n", .{info.max_threads});
    std.debug.print("  ABI Version: {s}\n", .{abi.version()});
    try printGpuSummary(allocator);
    printNetworkSummary();

    std.debug.print("\nFeature Matrix:\n", .{});
    for (std.enums.values(abi.Feature)) |tag| {
        const status = if (framework.isFeatureEnabled(tag)) "enabled" else "disabled";
        std.debug.print("  {s}: {s}\n", .{ @tagName(tag), status });
    }
}

fn runDb(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    try abi.database.cli.run(allocator, args);
}

fn runAgent(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    const agent_mod = abi.ai.agent;

    var name: []const u8 = "cli-agent";
    var message: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, arg, "--name")) {
            if (i < args.len) {
                name = args[i];
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--message") or std.mem.eql(u8, arg, "-m")) {
            if (i < args.len) {
                message = args[i];
                i += 1;
            }
            continue;
        }
    }

    var agent = try agent_mod.Agent.init(allocator, .{ .name = name });
    defer agent.deinit();

    if (message) |msg| {
        const response = try agent.process(msg, allocator);
        defer allocator.free(response);
        std.debug.print("User: {s}\n", .{msg});
        std.debug.print("Agent: {s}\n", .{response});
        return;
    }

    try runAgentInteractive(allocator, &agent);
}

fn runAgentInteractive(allocator: std.mem.Allocator, agent: *abi.ai.agent.Agent) !void {
    std.debug.print("Interactive mode. Type 'exit' to quit.\n", .{});
    var io_backend = std.Io.Threaded.init(allocator);
    defer io_backend.deinit();

    const io = io_backend.io();
    var stdin_file = std.fs.File.stdin();
    var buffer: [4096]u8 = undefined;
    var reader = stdin_file.reader(io, &buffer);

    while (true) {
        std.debug.print("> ", .{});
        const line_opt = reader.interface.takeDelimiter('\n') catch |err| switch (err) {
            error.ReadFailed => return err,
            error.StreamTooLong => {
                std.debug.print("Input too long. Try a shorter line.\n", .{});
                continue;
            },
        };
        const line = line_opt orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) {
            break;
        }
        const response = try agent.process(trimmed, allocator);
        defer allocator.free(response);
        std.debug.print("Agent: {s}\n", .{response});
    }
}

fn runGpu(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    if (args.len == 0) {
        try printGpuBackends(allocator);
        try printGpuDevices(allocator);
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);
    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help")) {
        printGpuHelp();
        return;
    }

    if (std.mem.eql(u8, command, "backends")) {
        try printGpuBackends(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "devices") or std.mem.eql(u8, command, "list")) {
        try printGpuDevices(allocator);
        return;
    }

    std.debug.print("Unknown gpu command: {s}\n", .{command});
    printGpuHelp();
}

fn runNetwork(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    abi.network.init(allocator) catch |err| switch (err) {
        error.NetworkDisabled => {
            std.debug.print("Network support disabled at build time.\n", .{});
            return;
        },
        else => return err,
    };

    if (args.len == 0) {
        try printNetworkStatus();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);
    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help")) {
        printNetworkHelp();
        return;
    }

    if (std.mem.eql(u8, command, "status")) {
        try printNetworkStatus();
        return;
    }

    if (std.mem.eql(u8, command, "list") or std.mem.eql(u8, command, "nodes")) {
        try printNetworkNodes();
        return;
    }

    if (std.mem.eql(u8, command, "register")) {
        if (args.len < 3) {
            std.debug.print("Usage: abi network register <id> <address>\n", .{});
            return;
        }
        const id = std.mem.sliceTo(args[1], 0);
        const address = std.mem.sliceTo(args[2], 0);
        const registry = try abi.network.defaultRegistry();
        try registry.register(id, address);
        std.debug.print("Registered {s} at {s}\n", .{ id, address });
        return;
    }

    if (std.mem.eql(u8, command, "unregister")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi network unregister <id>\n", .{});
            return;
        }
        const id = std.mem.sliceTo(args[1], 0);
        const registry = try abi.network.defaultRegistry();
        const removed = registry.unregister(id);
        if (removed) {
            std.debug.print("Unregistered {s}\n", .{id});
        } else {
            std.debug.print("Node {s} not found\n", .{id});
        }
        return;
    }

    if (std.mem.eql(u8, command, "touch")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi network touch <id>\n", .{});
            return;
        }
        const id = std.mem.sliceTo(args[1], 0);
        const registry = try abi.network.defaultRegistry();
        if (registry.touch(id)) {
            std.debug.print("Touched {s}\n", .{id});
        } else {
            std.debug.print("Node {s} not found\n", .{id});
        }
        return;
    }

    if (std.mem.eql(u8, command, "set-status")) {
        if (args.len < 3) {
            std.debug.print("Usage: abi network set-status <id> <status>\n", .{});
            return;
        }
        const id = std.mem.sliceTo(args[1], 0);
        const status_text = std.mem.sliceTo(args[2], 0);
        const status = parseNodeStatus(status_text) orelse {
            std.debug.print("Invalid status: {s}\n", .{status_text});
            return;
        };
        const registry = try abi.network.defaultRegistry();
        if (registry.setStatus(id, status)) {
            std.debug.print("Updated {s} to {s}\n", .{ id, @tagName(status) });
        } else {
            std.debug.print("Node {s} not found\n", .{id});
        }
        return;
    }

    std.debug.print("Unknown network command: {s}\n", .{command});
    printNetworkHelp();
}

fn printNetworkHelp() void {
    const text =
        "Usage: abi network <command>\n\n" ++
        "Commands:\n" ++
        "  status                    Show network config and node count\n" ++
        "  list | nodes               List registered nodes\n" ++
        "  register <id> <address>    Register or update a node\n" ++
        "  unregister <id>            Remove a node\n" ++
        "  touch <id>                 Update node heartbeat timestamp\n" ++
        "  set-status <id> <status>   Set status (healthy, degraded, offline)\n";
    std.debug.print("{s}", .{text});
}

fn printNetworkStatus() !void {
    if (!abi.network.isEnabled()) {
        std.debug.print("Network: disabled\n", .{});
        return;
    }
    const config = abi.network.defaultConfig();
    if (config == null) {
        std.debug.print("Network: enabled (not initialized)\n", .{});
        return;
    }
    const registry = try abi.network.defaultRegistry();
    std.debug.print(
        "Network cluster: {s}\nNodes: {d}\n",
        .{ config.?.cluster_id, registry.list().len },
    );
}

fn printNetworkNodes() !void {
    const registry = try abi.network.defaultRegistry();
    const nodes = registry.list();
    if (nodes.len == 0) {
        std.debug.print("No nodes registered.\n", .{});
        return;
    }
    std.debug.print("Nodes:\n", .{});
    for (nodes) |node| {
        std.debug.print(
            "  {s} {s} ({s}) last_seen_ms={d}\n",
            .{ node.id, node.address, @tagName(node.status), node.last_seen_ms },
        );
    }
}

fn parseNodeStatus(text: []const u8) ?abi.network.NodeStatus {
    if (std.ascii.eqlIgnoreCase(text, "healthy")) return .healthy;
    if (std.ascii.eqlIgnoreCase(text, "degraded")) return .degraded;
    if (std.ascii.eqlIgnoreCase(text, "offline")) return .offline;
    return null;
}

fn printGpuHelp() void {
    const help_text =
        "Usage: abi gpu <command>\n\n" ++
        "Commands:\n" ++
        "  backends   List GPU backends and build flags\n" ++
        "  devices    List detected GPU devices\n" ++
        "  list       Alias for devices\n";
    std.debug.print("{s}", .{help_text});
}

fn printGpuBackends(allocator: std.mem.Allocator) !void {
    const infos = try abi.gpu.listBackendInfo(allocator);
    defer allocator.free(infos);

    if (!abi.gpu.moduleEnabled()) {
        std.debug.print("GPU support disabled at build time.\n", .{});
    }

    std.debug.print("GPU Backends:\n", .{});
    for (infos) |info| {
        const status = if (info.enabled) "enabled" else "disabled";
        if (info.enabled) {
            std.debug.print(
                "  {s} ({s}) - {s}\n",
                .{ info.name, status, info.description },
            );
        } else {
            std.debug.print(
                "  {s} ({s}) - {s} [enable {s}]\n",
                .{ info.name, status, info.description, abi.gpu.backendFlag(info.backend) },
            );
        }
    }
}

fn printGpuDevices(allocator: std.mem.Allocator) !void {
    if (!abi.gpu.moduleEnabled()) {
        std.debug.print("GPU Devices: disabled (build without -Denable-gpu)\n", .{});
        return;
    }

    const devices = try abi.gpu.listDevices(allocator);
    defer allocator.free(devices);

    if (devices.len == 0) {
        std.debug.print("GPU Devices: none\n", .{});
        return;
    }

    std.debug.print("GPU Devices:\n", .{});
    for (devices) |device| {
        const emulated_suffix = if (device.is_emulated) " [emulated]" else "";
        if (device.total_memory_bytes) |memory| {
            std.debug.print(
                "  #{d} {s} ({s}) {d} bytes{s}\n",
                .{ device.id, device.name, abi.gpu.backendName(device.backend), memory, emulated_suffix },
            );
        } else {
            std.debug.print(
                "  #{d} {s} ({s}){s}\n",
                .{ device.id, device.name, abi.gpu.backendName(device.backend), emulated_suffix },
            );
        }
    }
}

fn printGpuSummary(allocator: std.mem.Allocator) !void {
    const summary = abi.gpu.summary();
    if (!summary.module_enabled) {
        std.debug.print("  GPU Backends: disabled\n", .{});
        std.debug.print("  GPU Devices: 0\n", .{});
        return;
    }

    const backends = try abi.gpu.availableBackends(allocator);
    defer allocator.free(backends);

    std.debug.print("  GPU Backends: ", .{});
    if (backends.len == 0) {
        std.debug.print("none\n", .{});
    } else {
        for (backends, 0..) |backend, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{s}", .{abi.gpu.backendName(backend)});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("  GPU Devices: {d}\n", .{summary.device_count});
}

fn printNetworkSummary() void {
    if (!abi.network.isEnabled()) {
        std.debug.print("  Network: disabled\n", .{});
        return;
    }
    if (!abi.network.isInitialized()) {
        std.debug.print("  Network: enabled (not initialized)\n", .{});
        return;
    }
    if (abi.network.defaultConfig()) |config| {
        const registry = abi.network.defaultRegistry() catch null;
        const node_count = if (registry) |reg| reg.list().len else 0;
        std.debug.print("  Network: {s} ({d} nodes)\n", .{ config.cluster_id, node_count });
    }
}
