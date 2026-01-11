//! GPU CLI command.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

/// Run the GPU command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        try printBackends(allocator);
        try printDevices(allocator);
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(command, &[_][]const u8{ "help", "--help" })) {
        printHelp();
        return;
    }

    if (std.mem.eql(u8, command, "backends")) {
        try printBackends(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "summary")) {
        try printSummaryCommand(allocator);
        return;
    }

    if (utils.args.matchesAny(command, &[_][]const u8{ "devices", "list" })) {
        try printDevices(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "default")) {
        try printDefaultDevice(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "status")) {
        try printStatus(allocator);
        return;
    }

    std.debug.print("Unknown gpu command: {s}\n", .{command});
    printHelp();
}

/// Print a short GPU summary for system-info.
pub fn printSummary(allocator: std.mem.Allocator) !void {
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

fn printHelp() void {
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

fn printBackends(allocator: std.mem.Allocator) !void {
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

fn printSummaryCommand(allocator: std.mem.Allocator) !void {
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

fn printDevices(allocator: std.mem.Allocator) !void {
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
                utils.output.boolLabel(caps.unified_memory),
                utils.output.boolLabel(caps.supports_fp16),
                utils.output.boolLabel(caps.supports_int8),
                utils.output.boolLabel(caps.supports_async_transfers),
            },
        );
        if (caps.max_threads_per_block != null or caps.max_shared_memory_bytes != null) {
            std.debug.print("      limits: threads/block=", .{});
            utils.output.printOptionalU32(caps.max_threads_per_block);
            std.debug.print(" shared=", .{});
            utils.output.printOptionalU32(caps.max_shared_memory_bytes);
            std.debug.print("\n", .{});
        }
    }
}

fn printDefaultDevice(allocator: std.mem.Allocator) !void {
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

fn printStatus(allocator: std.mem.Allocator) !void {
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
            for (devices) |dev| {
                if (dev.backend == backend) count += 1;
            }
            break :blk count;
        };

        if (backend_devices_count == 0) {
            std.debug.print("  {s}: enabled (no devices)\n", .{backend_name});
        } else {
            std.debug.print("  {s}: enabled ({d} device(s))\n", .{ backend_name, backend_devices_count });

            for (devices) |dev| {
                if (dev.backend == backend) {
                    const mode = if (dev.is_emulated) "fallback/simulation" else "native GPU";
                    std.debug.print("    #{d} {s} ({s})\n", .{ dev.id, dev.name, mode });
                }
            }
        }
    }
}
