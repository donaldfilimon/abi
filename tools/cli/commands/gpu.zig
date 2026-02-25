//! GPU CLI command.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const gpu_detect = abi.gpu.backends.detect;
const gpu_listing = abi.gpu.backends.listing;
const gpu_meta = abi.gpu.backends.meta;

// Wrapper functions for comptime children dispatch
fn wrapBackends(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try printBackends(allocator);
}
fn wrapSummaryCmd(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try printSummaryCommand(allocator);
}
fn wrapDevices(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try printDevices(allocator);
}
fn wrapDefault(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try printDefaultDevice(allocator);
}
fn wrapStatus(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try printStatus(allocator);
}

pub const meta: command_mod.Meta = .{
    .name = "gpu",
    .description = "GPU commands (backends, devices, summary, default)",
    .kind = .group,
    .subcommands = &.{ "backends", "devices", "list", "summary", "default", "status" },
    .children = &.{
        .{ .name = "backends", .description = "List GPU backends and build flags", .handler = wrapBackends },
        .{ .name = "devices", .description = "List detected GPU devices", .handler = wrapDevices },
        .{ .name = "list", .description = "List detected GPU devices", .handler = wrapDevices },
        .{ .name = "summary", .description = "Show GPU module summary", .handler = wrapSummaryCmd },
        .{ .name = "default", .description = "Show default GPU device", .handler = wrapDefault },
        .{ .name = "status", .description = "Show native/fallback status", .handler = wrapStatus },
    },
};

const gpu_subcommands = [_][]const u8{
    "backends", "devices", "list", "summary", "default", "status", "help",
};

/// Run the GPU command with the provided arguments.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        // Default action: show backends + devices
        try printBackends(allocator);
        try printDevices(allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp(allocator);
        return;
    }
    // Unknown subcommand
    utils.output.printError("Unknown gpu command: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &gpu_subcommands)) |suggestion| {
        utils.output.printInfo("Did you mean: {s}", .{suggestion});
    }
}

/// Print a short GPU summary for system-info.
pub fn printSummary(allocator: std.mem.Allocator) !void {
    const summary = gpu_listing.summary();
    if (!summary.module_enabled) {
        std.debug.print("  GPU Backends: disabled\n", .{});
        std.debug.print("  GPU Devices: 0\n", .{});
        return;
    }

    const backends = try gpu_detect.availableBackends(allocator);
    defer allocator.free(backends);

    std.debug.print("  GPU Backends: ", .{});
    if (backends.len == 0) {
        std.debug.print("none\n", .{});
    } else {
        for (backends, 0..) |backend, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{s}", .{gpu_meta.backendName(backend)});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print(
        "  GPU Devices: {d} (emulated {d})\n",
        .{ summary.device_count, summary.emulated_devices },
    );
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi gpu", "<command>")
        .description("GPU management and information commands.")
        .section("Commands")
        .subcommand(.{ .name = "backends", .description = "List GPU backends and build flags" })
        .subcommand(.{ .name = "summary", .description = "Show GPU module summary" })
        .subcommand(.{ .name = "devices", .description = "List detected GPU devices" })
        .subcommand(.{ .name = "default", .description = "Show default GPU device" })
        .subcommand(.{ .name = "status", .description = "Show native/fallback status" })
        .option(utils.help.common_options.help)
        .newline()
        .section("Examples")
        .example("abi gpu backends", "List backends")
        .example("abi gpu devices", "List devices")
        .example("abi gpu summary", "Show summary");

    builder.print();
}

fn printBackends(allocator: std.mem.Allocator) !void {
    const infos = try gpu_listing.listBackendInfo(allocator);
    defer allocator.free(infos);

    if (!gpu_detect.moduleEnabled()) {
        utils.output.printWarning("GPU support disabled at build time.", .{});
    }

    utils.output.printHeader("GPU Backends");
    for (infos) |info| {
        if (!info.enabled) {
            if (info.build_flag.len > 0) {
                std.debug.print(
                    "  {s} (disabled) - {s} [enable {s}]\n",
                    .{ info.name, info.description, info.build_flag },
                );
            } else {
                std.debug.print(
                    "  {s} (disabled) - {s}\n",
                    .{ info.name, info.description },
                );
            }
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
            const suffix = if (std.mem.eql(u8, info.name, "simulated")) " (fallback)" else "";
            std.debug.print(
                "  {s} (enabled) - {s}{s}\n",
                .{ info.name, info.description, suffix },
            );
        }
    }
    if (gpu_detect.moduleEnabled()) {
        std.debug.print("\n  Simulated is always available as a fallback when no hardware backend is detected.\n", .{});
    }
}

fn printSummaryCommand(allocator: std.mem.Allocator) !void {
    const summary = gpu_listing.summary();
    utils.output.printHeader("GPU Summary");
    if (!summary.module_enabled) {
        utils.output.printWarning("Status: disabled", .{});
        return;
    }

    const backends = try gpu_detect.availableBackends(allocator);
    defer allocator.free(backends);

    std.debug.print("  Backends: ", .{});
    if (backends.len == 0) {
        std.debug.print("none\n", .{});
    } else {
        for (backends, 0..) |backend, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{s}", .{gpu_meta.backendName(backend)});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print(
        "  Devices: {d} (emulated {d})\n",
        .{ summary.device_count, summary.emulated_devices },
    );
}

fn printDevices(allocator: std.mem.Allocator) !void {
    if (!gpu_detect.moduleEnabled()) {
        utils.output.printWarning("GPU Devices: disabled (build without -Denable-gpu)", .{});
        return;
    }

    const devices = try gpu_listing.listDevices(allocator);
    defer allocator.free(devices);

    if (devices.len == 0) {
        utils.output.printInfo("GPU Devices: none", .{});
        return;
    }

    utils.output.printHeader("GPU Devices");
    for (devices) |device| {
        const emulated_suffix = if (device.is_emulated) " [emulated]" else "";
        if (device.total_memory_bytes) |memory| {
            std.debug.print(
                "  #{d} {s} ({s}) {d} bytes{s}\n",
                .{
                    device.id,
                    device.name,
                    gpu_meta.backendName(device.backend),
                    memory,
                    emulated_suffix,
                },
            );
        } else {
            std.debug.print(
                "  #{d} {s} ({s}){s}\n",
                .{ device.id, device.name, gpu_meta.backendName(device.backend), emulated_suffix },
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
    if (!gpu_detect.moduleEnabled()) {
        std.debug.print("GPU default device: disabled (build without -Denable-gpu)\n", .{});
        return;
    }

    const device = try gpu_listing.defaultDevice(allocator);
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
                gpu_meta.backendName(selected.backend),
                memory,
                emulated_suffix,
            },
        );
    } else {
        std.debug.print(
            "GPU default device: #{d} {s} ({s}){s}\n",
            .{ selected.id, selected.name, gpu_meta.backendName(selected.backend), emulated_suffix },
        );
    }
}

fn printStatus(allocator: std.mem.Allocator) !void {
    if (!gpu_detect.moduleEnabled()) {
        utils.output.printWarning("GPU status: disabled (build without -Denable-gpu)", .{});
        return;
    }

    try abi.gpu.ensureInitialized(allocator);

    const backends = try gpu_detect.availableBackends(allocator);
    defer allocator.free(backends);

    utils.output.printHeader("GPU Status");

    if (backends.len == 0) {
        std.debug.print("  No backends available\n", .{});
        return;
    }

    for (backends) |backend| {
        const backend_name = gpu_meta.backendName(backend);
        const backend_enabled = abi.gpu.isEnabled(backend);

        if (!backend_enabled) {
            std.debug.print("  {s}: disabled (build)\n", .{backend_name});
            continue;
        }

        const devices = try gpu_listing.listDevices(allocator);
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
