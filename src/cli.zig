//! ABI Framework Command-Line Interface
//!
//! Provides a comprehensive CLI for interacting with all ABI framework features.
//! The CLI supports commands for database operations, GPU management, AI agent
//! interaction, network configuration, and system information.
//!
//! ## Usage
//! ```bash
//! abi <command> [options]
//! ```
//!
//! ## Commands
//! - `db` - Database operations (add, query, stats, optimize, backup)
//! - `agent` - Run AI agent (interactive or one-shot)
//! - `gpu` - GPU commands (backends, devices, summary, default)
//! - `network` - Manage network registry (list, register, status)
//! - `system-info` - Show system and framework status
//! - `version` - Show framework version
//! - `help` - Show help message
//!
//! ## Subcommands
//! Each command has its own help system. Run:
//! - `abi db help` for database commands
//! - `abi gpu help` for GPU commands
//! - `abi network help` for network commands
//!
//! ## Example
//! ```bash
//! # Initialize database and add a vector
//! abi db init
//! abi db add [1.0, 2.0, 3.0]
//!
//! # Run AI agent interactively
//! abi agent --interactive
//!
//! # Check GPU availability
//! abi gpu backends
//! ```
const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    const args = try init.minimal.args.toSlice(allocator);

    if (args.len <= 1) {
        printHelp();
        return;
    }

    const command = args[1];
    if (matchesAny(command, &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    if (matchesAny(command, &[_][]const u8{ "version", "--version", "-v" })) {
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

    if (std.mem.eql(u8, command, "explore")) {
        try runExplore(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "simd")) {
        try runSimdDemo(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "config")) {
        try runConfig(allocator, args[2..]);
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
        "  config [command]  Configuration management (init, show, validate)\n" ++
        "  explore [options] Search and explore codebase\n" ++
        "  gpu [subcommand]  GPU commands (backends, devices, summary, default)\n" ++
        "  network [command] Manage network registry (list, register, status)\n" ++
        "  simd              Run SIMD performance demo\n" ++
        "  system-info       Show system and framework status\n" ++
        "  version           Show framework version\n" ++
        "  help              Show this help message\n\n" ++
        "Run 'abi <command> help' for command-specific help.\n";

    std.debug.print("{s}", .{help_text});
}

fn runSystemInfo(allocator: std.mem.Allocator, framework: *abi.Framework) !void {
    const platform = abi.platform.platform;
    const info = platform.PlatformInfo.detect();

    std.debug.print("System Info\n", .{});
    std.debug.print("  OS: {t}\n", .{info.os});
    std.debug.print("  Arch: {t}\n", .{info.arch});
    std.debug.print("  CPU Threads: {d}\n", .{info.max_threads});
    std.debug.print("  ABI Version: {s}\n", .{abi.version()});
    try printGpuSummary(allocator);
    printNetworkSummary();

    std.debug.print("\nFeature Matrix:\n", .{});
    for (std.enums.values(abi.Feature)) |tag| {
        const status = if (framework.isFeatureEnabled(tag)) "enabled" else "disabled";
        std.debug.print("  {t}: {s}\n", .{ tag, status });
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

        if (matchesAny(arg, &[_][]const u8{ "--message", "-m" })) {
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
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();

    const io = io_backend.io();
    var stdin_file = std.Io.File.stdin();
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
    if (matchesAny(command, &[_][]const u8{ "help", "--help" })) {
        printGpuHelp();
        return;
    }

    if (std.mem.eql(u8, command, "backends")) {
        try printGpuBackends(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "summary")) {
        try printGpuSummaryCommand(allocator);
        return;
    }

    if (matchesAny(command, &[_][]const u8{ "devices", "list" })) {
        try printGpuDevices(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "default")) {
        try printGpuDefaultDevice(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "status")) {
        try printGpuStatus(allocator);
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
    if (matchesAny(command, &[_][]const u8{ "help", "--help" })) {
        printNetworkHelp();
        return;
    }

    if (std.mem.eql(u8, command, "status")) {
        try printNetworkStatus();
        return;
    }

    if (matchesAny(command, &[_][]const u8{ "list", "nodes" })) {
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
            std.debug.print("Updated {s} to {t}\n", .{ id, status });
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
            "  {s} {s} ({t}) last_seen_ms={d}\n",
            .{ node.id, node.address, node.status, node.last_seen_ms },
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
        "  summary    Show GPU module summary\n" ++
        "  devices    List detected GPU devices\n" ++
        "  list       Alias for devices\n" ++
        "  default    Show default GPU device\n" ++
        "  status     Show CUDA native/fallback status\n";
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
        if (!info.enabled) {
            std.debug.print(
                "  {s} (disabled) - {s} [enable {s}]\n",
                .{ info.name, info.description, info.build_flag },
            );
            continue;
        }

        if (!info.available) {
            std.debug.print(
                "  {s} (enabled) - {s} [unavailable: {s}]\n",
                .{ info.name, info.description, info.availability },
            );
            continue;
        }

        if (info.device_count > 0) {
            std.debug.print(
                "  {s} (enabled) - {s} [devices: {d}]\n",
                .{ info.name, info.description, info.device_count },
            );
        } else {
            std.debug.print(
                "  {s} (enabled) - {s}\n",
                .{ info.name, info.description },
            );
        }
    }
}

fn printGpuSummaryCommand(allocator: std.mem.Allocator) !void {
    const summary = abi.gpu.summary();
    std.debug.print("GPU Summary\n", .{});
    if (!summary.module_enabled) {
        std.debug.print("  Status: disabled\n", .{});
        return;
    }

    const backends = try abi.gpu.availableBackends(allocator);
    defer allocator.free(backends);

    std.debug.print("  Backends: ", .{});
    if (backends.len == 0) {
        std.debug.print("none\n", .{});
    } else {
        for (backends, 0..) |backend, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{s}", .{abi.gpu.backendName(backend)});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print(
        "  Devices: {d} (emulated {d})\n",
        .{ summary.device_count, summary.emulated_devices },
    );
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
                .{
                    device.id,
                    device.name,
                    abi.gpu.backendName(device.backend),
                    memory,
                    emulated_suffix,
                },
            );
        } else {
            std.debug.print(
                "  #{d} {s} ({s}){s}\n",
                .{ device.id, device.name, abi.gpu.backendName(device.backend), emulated_suffix },
            );
        }
        const caps = device.capability;
        std.debug.print(
            "      caps: unified={s} fp16={s} int8={s} async={s}\n",
            .{
                boolLabel(caps.unified_memory),
                boolLabel(caps.supports_fp16),
                boolLabel(caps.supports_int8),
                boolLabel(caps.supports_async_transfers),
            },
        );
        if (caps.max_threads_per_block != null or caps.max_shared_memory_bytes != null) {
            std.debug.print("      limits: threads/block=", .{});
            printOptionalU32(caps.max_threads_per_block);
            std.debug.print(" shared=", .{});
            printOptionalU32(caps.max_shared_memory_bytes);
            std.debug.print("\n", .{});
        }
    }
}

fn printGpuDefaultDevice(allocator: std.mem.Allocator) !void {
    if (!abi.gpu.moduleEnabled()) {
        std.debug.print("GPU default device: disabled (build without -Denable-gpu)\n", .{});
        return;
    }

    const device = try abi.gpu.defaultDevice(allocator);
    if (device == null) {
        std.debug.print("GPU default device: none\n", .{});
        return;
    }

    const selected = device.?;
    const emulated_suffix = if (selected.is_emulated) " [emulated]" else "";
    if (selected.total_memory_bytes) |memory| {
        std.debug.print(
            "GPU default device: #{d} {s} ({s}) {d} bytes{s}\n",
            .{
                selected.id,
                selected.name,
                abi.gpu.backendName(selected.backend),
                memory,
                emulated_suffix,
            },
        );
    } else {
        std.debug.print(
            "GPU default device: #{d} {s} ({s}){s}\n",
            .{ selected.id, selected.name, abi.gpu.backendName(selected.backend), emulated_suffix },
        );
    }
}

fn printGpuStatus(allocator: std.mem.Allocator) !void {
    if (!abi.gpu.moduleEnabled()) {
        std.debug.print("GPU status: disabled (build without -Denable-gpu)\n", .{});
        return;
    }

    try abi.gpu.ensureInitialized(allocator);

    const backends = try abi.gpu.availableBackends(allocator);
    defer allocator.free(backends);

    std.debug.print("GPU Status:\n", .{});

    if (backends.len == 0) {
        std.debug.print("  No backends available\n", .{});
        return;
    }

    for (backends) |backend| {
        const backend_name = abi.gpu.backendName(backend);
        const backend_enabled = abi.gpu.isEnabled(backend);

        if (!backend_enabled) {
            std.debug.print("  {s}: disabled (build)\n", .{backend_name});
            continue;
        }

        const devices = try abi.gpu.listDevices(allocator);
        defer allocator.free(devices);

        const backend_devices_count = blk: {
            var count: usize = 0;
            for (devices) |device| {
                if (device.backend == backend) count += 1;
            }
            break :blk count;
        };

        if (backend_devices_count == 0) {
            std.debug.print("  {s}: enabled (no devices)\n", .{backend_name});
        } else {
            std.debug.print("  {s}: enabled ({d} device(s))\n", .{ backend_name, backend_devices_count });

            for (devices) |device| {
                if (device.backend == backend) {
                    const mode = if (device.is_emulated) "fallback/simulation" else "native GPU";
                    std.debug.print("    #{d} {s} ({s})\n", .{ device.id, device.name, mode });
                }
            }
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

    std.debug.print(
        "  GPU Devices: {d} (emulated {d})\n",
        .{ summary.device_count, summary.emulated_devices },
    );
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

fn boolLabel(value: bool) []const u8 {
    return if (value) "yes" else "no";
}

fn matchesAny(text: []const u8, options: []const []const u8) bool {
    for (options) |option| {
        if (std.mem.eql(u8, text, option)) return true;
    }
    return false;
}

fn printOptionalU32(value: ?u32) void {
    if (value) |v| {
        std.debug.print("{d}", .{v});
    } else {
        std.debug.print("n/a", .{});
    }
}

fn runSimdDemo(allocator: std.mem.Allocator) !void {
    const has_simd = abi.hasSimdSupport();
    std.debug.print("SIMD Support: {s}\n", .{if (has_simd) "available" else "unavailable"});

    if (!has_simd) {
        std.debug.print("SIMD not available on this platform.\n", .{});
        return;
    }

    const size = 1000;
    const total_size = size * 3;
    var data = try allocator.alloc(f32, total_size);
    defer allocator.free(data);

    var a = data[0..size];
    var b = data[size .. size * 2];
    var result = data[size * 2 .. total_size];

    var i: usize = 0;
    while (i < size) : (i += 1) {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(i * 2);
    }

    var timer = try std.time.Timer.start();
    const start = timer.lap();

    abi.simd.vectorAdd(a, b, result);

    const end = timer.read();
    const add_time = end - start;

    i = 0;
    while (i < size) : (i += 1) {
        const expected = a[i] + b[i];
        if (@abs(result[i] - expected) > 1e-6) {
            std.debug.print("SIMD verification failed at index {d}\n", .{i});
            return;
        }
    }

    const dot_start = timer.lap();
    const dot_result = abi.simd.vectorDot(a, b);
    const dot_end = timer.read();
    const dot_time = dot_end - dot_start;

    const norm_start = timer.lap();
    const norm_result = abi.simd.vectorL2Norm(a);
    const norm_end = timer.read();
    const norm_time = norm_end - norm_start;

    const cos_start = timer.lap();
    const cos_result = abi.simd.cosineSimilarity(a, b);
    const cos_end = timer.read();
    const cos_time = cos_end - cos_start;

    std.debug.print("SIMD Operations Performance ({} elements):\n", .{size});
    std.debug.print("  Vector Addition: {d} ns ({d:.2} ops/sec)\n", .{
        add_time,
        @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(add_time)) / std.time.ns_per_s),
    });
    std.debug.print("  Dot Product: {d} ns ({d:.2} ops/sec)\n", .{
        dot_time,
        @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(dot_time)) / std.time.ns_per_s),
    });
    std.debug.print("  L2 Norm: {d} ns ({d:.2} ops/sec)\n", .{
        norm_time,
        @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(norm_time)) / std.time.ns_per_s),
    });
    std.debug.print("  Cosine Similarity: {d} ns ({d:.2} ops/sec)\n", .{
        cos_time,
        @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(cos_time)) / std.time.ns_per_s),
    });

    std.debug.print("Results:\n", .{});
    std.debug.print("  Dot Product: {d:.6}\n", .{dot_result});
    std.debug.print("  L2 Norm: {d:.6}\n", .{norm_result});
    std.debug.print("  Cosine Similarity: {d:.6}\n", .{cos_result});
}

fn runExplore(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    if (args.len == 0 or matchesAny(args[0], &[_][]const u8{ "help", "--help", "-h" })) {
        printExploreHelp();
        return;
    }

    var query: ?[]const u8 = null;
    var root_path: []const u8 = ".";
    var level: abi.ai.explore.ExploreLevel = .medium;
    var output_format: abi.ai.explore.OutputFormat = .human;
    var include_patterns = std.ArrayListUnmanaged([]const u8){};
    var exclude_patterns = std.ArrayListUnmanaged([]const u8){};
    var case_sensitive = false;
    var use_regex = false;
    var max_files: usize = undefined;
    var max_depth: usize = undefined;
    var timeout_ms: u64 = undefined;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (matchesAny(arg, &[_][]const u8{ "--help", "-h" })) {
            printExploreHelp();
            return;
        }

        if (matchesAny(arg, &[_][]const u8{ "--level", "-l" })) {
            if (i < args.len) {
                const level_str = std.mem.sliceTo(args[i], 0);
                level = switch (std.ascii.eqlIgnoreCase(level_str, "quick")) {
                    true => .quick,
                    else => switch (std.ascii.eqlIgnoreCase(level_str, "medium")) {
                        true => .medium,
                        else => switch (std.ascii.eqlIgnoreCase(level_str, "thorough")) {
                            true => .thorough,
                            else => switch (std.ascii.eqlIgnoreCase(level_str, "deep")) {
                                true => .deep,
                                else => {
                                    std.debug.print("Unknown level: {s}. Use: quick, medium, thorough, deep\n", .{level_str});
                                    return;
                                },
                            },
                        },
                    },
                };
                i += 1;
            }
            continue;
        }

        if (matchesAny(arg, &[_][]const u8{ "--format", "-f" })) {
            if (i < args.len) {
                const format_str = std.mem.sliceTo(args[i], 0);
                output_format = switch (std.ascii.eqlIgnoreCase(format_str, "json")) {
                    true => .json,
                    else => switch (std.ascii.eqlIgnoreCase(format_str, "compact")) {
                        true => .compact,
                        else => switch (std.ascii.eqlIgnoreCase(format_str, "yaml")) {
                            true => .yaml,
                            else => .human,
                        },
                    },
                };
                i += 1;
            }
            continue;
        }

        if (matchesAny(arg, &[_][]const u8{ "--include", "-i" })) {
            if (i < args.len) {
                try include_patterns.append(allocator, std.mem.sliceTo(args[i], 0));
                i += 1;
            }
            continue;
        }

        if (matchesAny(arg, &[_][]const u8{ "--exclude", "-e" })) {
            if (i < args.len) {
                try exclude_patterns.append(allocator, std.mem.sliceTo(args[i], 0));
                i += 1;
            }
            continue;
        }

        if (matchesAny(arg, &[_][]const u8{ "--case-sensitive", "-c" })) {
            case_sensitive = true;
            continue;
        }

        if (matchesAny(arg, &[_][]const u8{ "--regex", "-r" })) {
            use_regex = true;
            continue;
        }

        if (matchesAny(arg, &[_][]const u8{"--max-files"})) {
            if (i < args.len) {
                max_files = try std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10);
                i += 1;
            }
            continue;
        }

        if (matchesAny(arg, &[_][]const u8{"--max-depth"})) {
            if (i < args.len) {
                max_depth = try std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10);
                i += 1;
            }
            continue;
        }

        if (matchesAny(arg, &[_][]const u8{"--timeout"})) {
            if (i < args.len) {
                timeout_ms = try std.fmt.parseInt(u64, std.mem.sliceTo(args[i], 0), 10);
                i += 1;
            }
            continue;
        }

        if (std.mem.startsWith(u8, arg, "--path")) {
            if (i < args.len) {
                root_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (query == null) {
            query = std.mem.sliceTo(arg, 0);
        } else {
            std.debug.print("Unknown argument: {s}\n", .{arg});
            printExploreHelp();
            return;
        }
    }

    const search_query = query orelse {
        std.debug.print("Error: No search query provided.\n", .{});
        printExploreHelp();
        return;
    };

    var config = abi.ai.explore.ExploreConfig.defaultForLevel(level);
    config.output_format = output_format;
    config.case_sensitive = case_sensitive;
    config.use_regex = use_regex;

    if (include_patterns.items.len > 0) {
        config.include_patterns = include_patterns.items;
    }
    if (exclude_patterns.items.len > 0) {
        config.exclude_patterns = exclude_patterns.items;
    }
    if (max_files > 0) config.max_files = max_files;
    if (max_depth > 0) config.max_depth = max_depth;
    if (timeout_ms > 0) config.timeout_ms = timeout_ms;

    var agent = abi.ai.explore.ExploreAgent.init(allocator, config);
    defer agent.deinit();

    const start_time = try std.time.Instant.now();
    const result = try agent.explore(root_path, search_query);
    defer result.deinit();

    const end_time = try std.time.Instant.now();
    const duration_ms = @divTrunc(end_time.since(start_time), std.time.ns_per_ms);

    switch (output_format) {
        .human => {
            try result.formatHuman(std.debug);
        },
        .json => {
            try result.formatJSON(std.debug);
        },
        .compact => {
            std.debug.print("Query: \"{s}\" | Found: {d} matches in {d}ms\n", .{
                search_query, result.matches_found, duration_ms,
            });
        },
        .yaml => {
            std.debug.print("query: \"{s}\"\n", .{search_query});
            std.debug.print("level: {t}\n", .{level});
            std.debug.print("matches_found: {d}\n", .{result.matches_found});
            std.debug.print("duration_ms: {d}\n", .{duration_ms});
        },
    }

    include_patterns.deinit(allocator);
    exclude_patterns.deinit(allocator);
}

fn printExploreHelp() void {
    const help_text =
        "Usage: abi explore [options] <query>\n\n" ++
        "Search and explore the codebase for patterns.\n\n" ++
        "Arguments:\n" ++
        "  <query>              Search pattern or natural language query\n\n" ++
        "Options:\n" ++
        "  -l, --level <level>  Exploration depth: quick, medium, thorough, deep (default: medium)\n" ++
        "  -f, --format <fmt>   Output format: human, json, compact, yaml (default: human)\n" ++
        "  -i, --include <pat>  Include files matching pattern (can be used multiple times)\n" ++
        "  -e, --exclude <pat>  Exclude files matching pattern (can be used multiple times)\n" ++
        "  -c, --case-sensitive Match case sensitively\n" ++
        "  -r, --regex          Treat query as regex pattern\n" ++
        "  --path <path>        Root directory to search (default: .)\n" ++
        "  --max-files <n>      Maximum files to scan\n" ++
        "  --max-depth <n>      Maximum directory depth\n" ++
        "  --timeout <ms>       Timeout in milliseconds\n" ++
        "  -h, --help           Show this help message\n\n" ++
        "Examples:\n" ++
        "  abi explore \"HTTP handler\"\n" ++
        "  abi explore -l thorough \"TODO\"\n" ++
        "  abi explore -f json \"function_name\"\n" ++
        "  abi explore -i \"*.zig\" \"pub fn\"\n" ++
        "  abi explore --regex \"fn\\s+\\w+\"";
    std.debug.print("{s}\n", .{help_text});
}

fn runConfig(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    if (args.len == 0 or matchesAny(args[0], &[_][]const u8{ "help", "--help", "-h" })) {
        printConfigHelp();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);

    if (std.mem.eql(u8, command, "init")) {
        try runConfigInit(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "show")) {
        try runConfigShow(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "validate")) {
        try runConfigValidate(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "env")) {
        try runConfigEnv(allocator);
        return;
    }

    std.debug.print("Unknown config command: {s}\n", .{command});
    printConfigHelp();
}

fn runConfigInit(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    var output_path: []const u8 = "abi.json";

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (matchesAny(arg, &[_][]const u8{ "--output", "-o" })) {
            if (i < args.len) {
                output_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    // Create default configuration
    const default_config = getDefaultConfigJson();

    // Write to file
    const file = std.fs.cwd().createFile(output_path, .{ .truncate = true }) catch |err| {
        std.debug.print("Error creating config file '{s}': {t}\n", .{ output_path, err });
        return;
    };
    defer file.close();

    file.writeAll(default_config) catch |err| {
        std.debug.print("Error writing config file: {t}\n", .{err});
        return;
    };

    std.debug.print("Created configuration file: {s}\n", .{output_path});
    std.debug.print("\nEdit this file to customize your ABI framework settings.\n", .{});
    std.debug.print("Run 'abi config validate {s}' to check your configuration.\n", .{output_path});
    _ = allocator;
}

fn runConfigShow(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    var format: enum { human, json } = .human;
    var config_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (matchesAny(arg, &[_][]const u8{ "--format", "-f" })) {
            if (i < args.len) {
                const fmt = std.mem.sliceTo(args[i], 0);
                if (std.mem.eql(u8, fmt, "json")) {
                    format = .json;
                }
                i += 1;
            }
            continue;
        }

        if (config_path == null) {
            config_path = std.mem.sliceTo(arg, 0);
        }
    }

    if (config_path) |path| {
        // Load and show from file
        var loader = abi.config.ConfigLoader.init(allocator);
        const config = loader.loadFromFile(path) catch |err| {
            std.debug.print("Error loading config file '{s}': {t}\n", .{ path, err });
            return;
        };
        defer @constCast(&config).deinit();

        switch (format) {
            .human => printConfigHuman(&config),
            .json => std.debug.print("{s}\n", .{getDefaultConfigJson()}),
        }
    } else {
        // Show default configuration
        switch (format) {
            .human => printDefaultConfigHuman(),
            .json => std.debug.print("{s}\n", .{getDefaultConfigJson()}),
        }
    }
}

fn runConfigValidate(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    if (args.len == 0) {
        std.debug.print("Usage: abi config validate <config-file>\n", .{});
        return;
    }

    const path = std.mem.sliceTo(args[0], 0);

    // Try to load the configuration file
    var loader = abi.config.ConfigLoader.init(allocator);
    var config = loader.loadFromFile(path) catch |err| {
        std.debug.print("Error: Failed to load '{s}'\n", .{path});
        std.debug.print("  Reason: {t}\n", .{err});
        std.process.exit(1);
    };
    defer config.deinit();

    // Validate the configuration
    config.validate() catch |err| {
        std.debug.print("Error: Configuration validation failed\n", .{});
        std.debug.print("  Reason: {t}\n", .{err});
        std.process.exit(1);
    };

    std.debug.print("Configuration file '{s}' is valid.\n", .{path});
    std.debug.print("\nConfiguration summary:\n", .{});
    printConfigHuman(&config);
}

fn runConfigEnv(allocator: std.mem.Allocator) !void {
    std.debug.print("Environment Variables for ABI Framework\n", .{});
    std.debug.print("========================================\n\n", .{});

    std.debug.print("Framework Settings:\n", .{});
    std.debug.print("  ABI_ENABLE_AI          Enable AI features (true/false)\n", .{});
    std.debug.print("  ABI_ENABLE_GPU         Enable GPU features (true/false)\n", .{});
    std.debug.print("  ABI_ENABLE_WEB         Enable web features (true/false)\n", .{});
    std.debug.print("  ABI_ENABLE_DATABASE    Enable database features (true/false)\n", .{});
    std.debug.print("  ABI_ENABLE_NETWORK     Enable network features (true/false)\n", .{});
    std.debug.print("  ABI_WORKER_THREADS     Number of worker threads (0=auto)\n", .{});
    std.debug.print("  ABI_LOG_LEVEL          Log level (debug/info/warn/err)\n", .{});

    std.debug.print("\nAI Connectors:\n", .{});
    std.debug.print("  ABI_OPENAI_API_KEY     OpenAI API key\n", .{});
    std.debug.print("  OPENAI_API_KEY         OpenAI API key (fallback)\n", .{});
    std.debug.print("  ABI_HF_API_TOKEN       HuggingFace API token\n", .{});
    std.debug.print("  HF_API_TOKEN           HuggingFace API token (fallback)\n", .{});
    std.debug.print("  ABI_OLLAMA_HOST        Ollama host URL\n", .{});
    std.debug.print("  OLLAMA_HOST            Ollama host URL (fallback)\n", .{});

    std.debug.print("\nDatabase:\n", .{});
    std.debug.print("  ABI_DATABASE_NAME      Database file name\n", .{});

    std.debug.print("\nNetwork:\n", .{});
    std.debug.print("  ABI_CLUSTER_ID         Cluster identifier\n", .{});
    std.debug.print("  ABI_NODE_ADDRESS       Node address (host:port)\n", .{});

    std.debug.print("\nWeb:\n", .{});
    std.debug.print("  ABI_WEB_PORT           Web server port\n", .{});
    std.debug.print("  ABI_WEB_CORS           Enable CORS (true/false)\n", .{});

    std.debug.print("\nGPU:\n", .{});
    std.debug.print("  ABI_GPU_BACKEND        Preferred GPU backend\n", .{});

    _ = allocator;
}

fn printConfigHelp() void {
    const help_text =
        "Usage: abi config <command> [options]\n\n" ++
        "Manage ABI framework configuration files.\n\n" ++
        "Commands:\n" ++
        "  init [options]       Generate a default configuration file\n" ++
        "  show [file]          Display configuration (default or from file)\n" ++
        "  validate <file>      Validate a configuration file\n" ++
        "  env                  List environment variables\n" ++
        "  help                 Show this help message\n\n" ++
        "Init options:\n" ++
        "  -o, --output <path>  Output file path (default: abi.json)\n\n" ++
        "Show options:\n" ++
        "  -f, --format <fmt>   Output format: human, json (default: human)\n\n" ++
        "Examples:\n" ++
        "  abi config init                    Create default abi.json\n" ++
        "  abi config init -o myconfig.json   Create custom config file\n" ++
        "  abi config show                    Show default configuration\n" ++
        "  abi config show abi.json           Show file configuration\n" ++
        "  abi config show -f json            Show as JSON\n" ++
        "  abi config validate abi.json       Validate config file\n" ++
        "  abi config env                     List environment variables\n";
    std.debug.print("{s}", .{help_text});
}

fn printConfigHuman(config: *const abi.config.Config) void {
    std.debug.print("  Source: {t}\n", .{config.source});
    std.debug.print("\n  [Framework]\n", .{});
    std.debug.print("    enable_ai: {s}\n", .{boolLabel(config.framework.enable_ai)});
    std.debug.print("    enable_gpu: {s}\n", .{boolLabel(config.framework.enable_gpu)});
    std.debug.print("    enable_web: {s}\n", .{boolLabel(config.framework.enable_web)});
    std.debug.print("    enable_database: {s}\n", .{boolLabel(config.framework.enable_database)});
    std.debug.print("    enable_network: {s}\n", .{boolLabel(config.framework.enable_network)});
    std.debug.print("    worker_threads: {d}\n", .{config.framework.worker_threads});
    std.debug.print("    log_level: {t}\n", .{config.framework.log_level});

    std.debug.print("\n  [Database]\n", .{});
    std.debug.print("    name: {s}\n", .{config.database.name});
    std.debug.print("    persistence_enabled: {s}\n", .{boolLabel(config.database.persistence_enabled)});
    std.debug.print("    vector_search_enabled: {s}\n", .{boolLabel(config.database.vector_search_enabled)});
    std.debug.print("    default_search_limit: {d}\n", .{config.database.default_search_limit});

    std.debug.print("\n  [AI]\n", .{});
    std.debug.print("    temperature: {d:.2}\n", .{config.ai.temperature});
    std.debug.print("    max_tokens: {d}\n", .{config.ai.max_tokens});
    std.debug.print("    streaming_enabled: {s}\n", .{boolLabel(config.ai.streaming_enabled)});

    std.debug.print("\n  [Network]\n", .{});
    std.debug.print("    distributed_enabled: {s}\n", .{boolLabel(config.network.distributed_enabled)});
    std.debug.print("    cluster_id: {s}\n", .{config.network.cluster_id});
    std.debug.print("    node_address: {s}\n", .{config.network.node_address});

    std.debug.print("\n  [Web]\n", .{});
    std.debug.print("    server_enabled: {s}\n", .{boolLabel(config.web.server_enabled)});
    std.debug.print("    port: {d}\n", .{config.web.port});
    std.debug.print("    cors_enabled: {s}\n", .{boolLabel(config.web.cors_enabled)});
}

fn printDefaultConfigHuman() void {
    std.debug.print("Default Configuration:\n", .{});
    std.debug.print("  Source: default\n", .{});
    std.debug.print("\n  [Framework]\n", .{});
    std.debug.print("    enable_ai: yes\n", .{});
    std.debug.print("    enable_gpu: yes\n", .{});
    std.debug.print("    enable_web: yes\n", .{});
    std.debug.print("    enable_database: yes\n", .{});
    std.debug.print("    enable_network: no\n", .{});
    std.debug.print("    worker_threads: 0 (auto-detect)\n", .{});
    std.debug.print("    log_level: info\n", .{});

    std.debug.print("\n  [Database]\n", .{});
    std.debug.print("    name: abi.db\n", .{});
    std.debug.print("    persistence_enabled: yes\n", .{});
    std.debug.print("    vector_search_enabled: yes\n", .{});
    std.debug.print("    default_search_limit: 10\n", .{});

    std.debug.print("\n  [AI]\n", .{});
    std.debug.print("    temperature: 0.70\n", .{});
    std.debug.print("    max_tokens: 2048\n", .{});
    std.debug.print("    streaming_enabled: yes\n", .{});

    std.debug.print("\n  [Network]\n", .{});
    std.debug.print("    distributed_enabled: no\n", .{});
    std.debug.print("    cluster_id: default\n", .{});
    std.debug.print("    node_address: 0.0.0.0:9000\n", .{});

    std.debug.print("\n  [Web]\n", .{});
    std.debug.print("    server_enabled: no\n", .{});
    std.debug.print("    port: 8080\n", .{});
    std.debug.print("    cors_enabled: yes\n", .{});
}

fn getDefaultConfigJson() []const u8 {
    return
        \\{
        \\  "framework": {
        \\    "enable_ai": true,
        \\    "enable_gpu": true,
        \\    "enable_web": true,
        \\    "enable_database": true,
        \\    "enable_network": false,
        \\    "enable_profiling": false,
        \\    "worker_threads": 0,
        \\    "log_level": "info"
        \\  },
        \\  "database": {
        \\    "name": "abi.db",
        \\    "max_records": 0,
        \\    "persistence_enabled": true,
        \\    "persistence_path": "abi_data",
        \\    "vector_search_enabled": true,
        \\    "default_search_limit": 10,
        \\    "max_vector_dimension": 4096
        \\  },
        \\  "gpu": {
        \\    "enable_cuda": false,
        \\    "enable_vulkan": false,
        \\    "enable_metal": false,
        \\    "enable_webgpu": false,
        \\    "enable_opengl": false,
        \\    "enable_opengles": false,
        \\    "enable_webgl2": false,
        \\    "preferred_backend": "",
        \\    "memory_pool_mb": 0
        \\  },
        \\  "ai": {
        \\    "default_model": "",
        \\    "max_tokens": 2048,
        \\    "temperature": 0.7,
        \\    "top_p": 0.9,
        \\    "streaming_enabled": true,
        \\    "timeout_ms": 60000,
        \\    "history_enabled": true,
        \\    "max_history": 100
        \\  },
        \\  "network": {
        \\    "distributed_enabled": false,
        \\    "cluster_id": "default",
        \\    "node_address": "0.0.0.0:9000",
        \\    "heartbeat_interval_ms": 5000,
        \\    "node_timeout_ms": 30000,
        \\    "max_nodes": 16,
        \\    "peer_discovery": false
        \\  },
        \\  "web": {
        \\    "server_enabled": false,
        \\    "port": 8080,
        \\    "max_connections": 256,
        \\    "request_timeout_ms": 30000,
        \\    "cors_enabled": true,
        \\    "cors_origins": "*"
        \\  }
        \\}
        \\
    ;
}

test "matchesAny helper function" {
    try std.testing.expect(matchesAny("help", &[_][]const u8{ "help", "--help", "-h" }));
    try std.testing.expect(matchesAny("--help", &[_][]const u8{ "help", "--help", "-h" }));
    try std.testing.expect(matchesAny("-h", &.{ "help", "--help", "-h" }));
    try std.testing.expect(!matchesAny("invalid", &[_][]const u8{ "help", "--help", "-h" }));
    try std.testing.expect(matchesAny("test", &[_][]const u8{"test"}));
    try std.testing.expect(!matchesAny("test", &[_][]const u8{"other"}));
}

test "parseNodeStatus helper function" {
    try std.testing.expect(parseNodeStatus("healthy") == .healthy);
    try std.testing.expect(parseNodeStatus("Healthy") == .healthy);
    try std.testing.expect(parseNodeStatus("HEALTHY") == .healthy);
    try std.testing.expect(parseNodeStatus("degraded") == .degraded);
    try std.testing.expect(parseNodeStatus("Degraded") == .degraded);
    try std.testing.expect(parseNodeStatus("offline") == .offline);
    try std.testing.expect(parseNodeStatus("Offline") == .offline);
    try std.testing.expect(parseNodeStatus("invalid") == null);
    try std.testing.expect(parseNodeStatus("") == null);
}

test "boolLabel helper function" {
    try std.testing.expectEqualStrings("yes", boolLabel(true));
    try std.testing.expectEqualStrings("no", boolLabel(false));
}

test "getDefaultConfigJson returns valid JSON" {
    const json = getDefaultConfigJson();
    try std.testing.expect(json.len > 0);
    try std.testing.expect(std.mem.startsWith(u8, json, "{"));
    try std.testing.expect(std.mem.indexOf(u8, json, "\"framework\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"database\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"ai\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"network\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"web\"") != null);
}
