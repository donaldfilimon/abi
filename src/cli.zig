//! ABI CLI implementation shared by tools/cli and the legacy fallback.
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
    if (matchesAny(command, &.{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    if (matchesAny(command, &.{ "version", "--version", "-v" })) {
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

    if (std.mem.eql(u8, command, "simd")) {
        try runSimdDemo(allocator);
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
        "  gpu [subcommand]  GPU commands (backends, devices, summary, default)\n" ++
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

        if (matchesAny(arg, &.{ "--message", "-m" })) {
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
    if (matchesAny(command, &.{ "help", "--help" })) {
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

    if (matchesAny(command, &.{ "devices", "list" })) {
        try printGpuDevices(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "default")) {
        try printGpuDefaultDevice(allocator);
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
    if (matchesAny(command, &.{ "help", "--help" })) {
        printNetworkHelp();
        return;
    }

    if (std.mem.eql(u8, command, "status")) {
        try printNetworkStatus();
        return;
    }

    if (matchesAny(command, &.{ "list", "nodes" })) {
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
        "  default    Show the default GPU device\n";
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

    // Create test data
    const size = 1000;
    var a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    var b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    var result = try allocator.alloc(f32, size);
    defer allocator.free(result);

    // Initialize test vectors
    var i: usize = 0;
    while (i < size) : (i += 1) {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(i * 2);
    }

    // Time SIMD operations
    var timer = try std.time.Timer.start();
    const start = timer.lap();

    abi.simd.vectorAdd(a, b, result);

    const end = timer.read();
    const add_time = end - start;

    // Verify results
    i = 0;
    while (i < size) : (i += 1) {
        const expected = a[i] + b[i];
        if (@abs(result[i] - expected) > 1e-6) {
            std.debug.print("SIMD verification failed at index {d}\n", .{i});
            return;
        }
    }

    // Test dot product
    const dot_start = timer.lap();
    const dot_result = abi.simd.vectorDot(a, b);
    const dot_end = timer.read();
    const dot_time = dot_end - dot_start;

    // Test L2 norm
    const norm_start = timer.lap();
    const norm_result = abi.simd.vectorL2Norm(a);
    const norm_end = timer.read();
    const norm_time = norm_end - norm_start;

    // Test cosine similarity
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
