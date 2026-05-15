const std = @import("std");
const abi = @import("root.zig");

const Command = struct {
    name: []const u8,
    usage: []const u8,
    summary: []const u8,
};

const commands = [_]Command{
    .{ .name = "help", .usage = "abi help [command]", .summary = "Show top-level or command-specific help" },
    .{ .name = "train", .usage = "abi train <input>", .summary = "Run the AI pipeline compatibility command" },
    .{ .name = "agent", .usage = "abi agent <plan|train|tui|os> ...", .summary = "Run safe agent planning, WDBX-backed training scaffolds, TUI, or OS dry-runs" },
    .{ .name = "backends", .usage = "abi backends", .summary = "Show GPU, accelerator, shader, and MLIR backend status" },
    .{ .name = "plugin", .usage = "abi plugin list", .summary = "Inspect installed plugins" },
    .{ .name = "tui", .usage = "abi tui", .summary = "Render a minimal terminal dashboard" },
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    // Simple flag check
    if (args.len >= 2) {
        if (std.mem.eql(u8, args[1], "--tui")) {
            _ = try renderTui(allocator);
            return;
        }
    }

    const exit_code = runCli(allocator, args) catch |err| switch (err) {
        error.CommandDenied => blk: {
            std.debug.print("error: command denied by safety policy\n", .{});
            break :blk 3;
        },
        else => blk: {
            std.debug.print("error: {s}\n", .{@errorName(err)});
            break :blk 1;
        },
    };

    if (exit_code != 0) std.process.exit(exit_code);
}

fn runCli(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 2) {
        printUsage();
        return 0;
    }

    const cmd = args[1];

    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        if (args.len >= 3) return printCommandHelp(args[2]);
        printUsage();
        return 0;
    } else if (std.mem.eql(u8, cmd, "train")) {
        if (args.len != 3) return usageError("usage: abi train <input>");
        return handleTrain(allocator, args[2]);
    } else if (std.mem.eql(u8, cmd, "agent")) {
        return handleAgent(allocator, args);
    } else if (std.mem.eql(u8, cmd, "backends")) {
        if (args.len != 2) return usageError("usage: abi backends");
        return handleBackends();
    } else if (std.mem.eql(u8, cmd, "plugin")) {
        return handlePlugin(allocator, args);
    } else if (std.mem.eql(u8, cmd, "tui")) {
        if (args.len != 2) return usageError("usage: abi tui");
        return renderTui(allocator);
    } else {
        std.debug.print("error: unknown command '{s}'\n\n", .{cmd});
        printUsage();
        return 2;
    }
}

fn handleBackends() !u8 {
    const gpu_status = abi.features.gpu.detectBackend();
    const training = abi.features.accelerator.selectBackend(.training);
    const shader = try abi.features.shaders.compile(std.heap.page_allocator, .{
        .name = "status",
        .source = "fn main() void {}",
    });
    defer shader.deinit(std.heap.page_allocator);
    const lowered = try abi.features.mlir.lower(std.heap.page_allocator, .{
        .name = "status",
        .operations = &.{"matmul"},
    });
    defer lowered.deinit(std.heap.page_allocator);

    std.debug.print("GPU: {s} available={any} accelerated={any}\n", .{
        abi.features.gpu.backendName(gpu_status.backend),
        gpu_status.available,
        gpu_status.accelerated,
    });
    std.debug.print("Accelerator: {s} ({s})\n", .{ abi.features.accelerator.backendName(training.backend), training.message });
    std.debug.print("Shader: {s} backend={s}\n", .{ abi.features.shaders.languageName(shader.language), shader.backend });
    std.debug.print("MLIR: {s} backend={s}\n", .{ abi.features.mlir.dialectName(lowered.dialect), lowered.target_backend });
    return 0;
}

fn handleTrain(allocator: std.mem.Allocator, input: []const u8) !u8 {
    const response = try abi.features.ai.run(allocator, input);
    defer allocator.free(response);
    std.debug.print("{s}\n", .{response});
    return 0;
}

fn handleAgent(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usageError("usage: abi agent <plan|train|tui|os> ...");

    const sub_cmd = args[2];
    if (std.mem.eql(u8, sub_cmd, "plan")) {
        if (args.len != 4) return usageError("usage: abi agent plan <input>");
        const result = try abi.features.ai.runAgent(allocator, .{ .name = "cli-agent", .instructions = "Plan only; do not execute.", .dry_run = true }, args[3]);
        defer result.deinit(allocator);
        // Note: simplified agent interface call
        std.debug.print("{s}\n", .{result.output});
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "train")) {
        if (args.len != 4) return usageError("usage: abi agent train <abbey|aviva|abi|all>");
        var store = abi.features.wdbx.Store.init(allocator);
        defer store.deinit();

        const dataset = abi.features.ai.DatasetSpec{ .path = "datasets/placeholder.jsonl" };
        const artifact_dir = "zig-cache/agent-artifacts";
        const result = if (std.mem.eql(u8, args[3], "all"))
            try abi.features.ai.trainKnownProfiles(allocator, &store, dataset, artifact_dir)
        else
            try abi.features.ai.trainWithStore(allocator, &store, .{
                .profile = args[3],
                .dataset = dataset,
                .artifact_dir = artifact_dir,
            });
        defer result.deinit(allocator);

        std.debug.print("{s}: {s} ({d} wdbx record(s), backend={s})\n", .{ result.profile, result.message, store.count(), result.acceleration_backend });
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "tui")) {
        if (args.len != 3) return usageError("usage: abi agent tui");
        return renderTui(allocator);
    } else if (std.mem.eql(u8, sub_cmd, "os") and args.len >= 5) {
        return handleAgentOs(allocator, args);
    } else {
        return usageError("usage: abi agent <plan|train|tui|os dry-run|os execute> ...");
    }
}

fn handleAgentOs(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    _ = allocator;
    const os_cmd = args[3];
    const start: usize = if (std.mem.eql(u8, os_cmd, "execute") and args.len >= 6 and std.mem.eql(u8, args[4], "--confirm")) 5 else 4;
    if (start >= args.len) return usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");

    const request = abi.features.os_control.CommandRequest{
        .intent = .read_only,
        .argv = args[start..],
    };

    if (std.mem.eql(u8, os_cmd, "dry-run")) {
        const rendered = try abi.features.os_control.renderDryRun(std.heap.page_allocator, request);
        defer std.heap.page_allocator.free(rendered);
        std.debug.print("{s}\n", .{rendered});
        return 0;
    } else if (std.mem.eql(u8, os_cmd, "execute")) {
        return error.CommandDenied;
    } else {
        return usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");
    }
}

fn handlePlugin(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 3 or !std.mem.eql(u8, args[2], "list")) return usageError("usage: abi plugin list");

    var registry = abi.registry.Registry.init(allocator);
    defer registry.deinit();
    try registry.loadPlugins();

    std.debug.print("Installed Plugins:\n", .{});
    var it = registry.modules.iterator();
    while (it.next()) |entry| {
        std.debug.print("  - {s}: {s}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
    }
    return 0;
}

fn renderTui(allocator: std.mem.Allocator) !u8 {
    _ = allocator;
    try abi.features.tui.initScreen();
    defer abi.features.tui.deinitScreen();
    try abi.features.tui.render(.{ .width = 80, .height = 24 });
    return 0;
}

fn printUsage() void {
    std.debug.print("Usage: abi <command> [args...] [--tui]\n\nCommands:\n", .{});
    for (commands) |command| {
        std.debug.print("  {s:<8} {s}\n", .{ command.name, command.summary });
    }
    std.debug.print("\nRun `abi help <command>` for details.\n", .{});
}

fn printCommandHelp(name: []const u8) u8 {
    for (commands) |command| {
        if (std.mem.eql(u8, command.name, name)) {
            std.debug.print("{s}\n\n{s}\n", .{ command.usage, command.summary });
            return 0;
        }
    }
    std.debug.print("error: unknown command '{s}'\n\n", .{name});
    printUsage();
    return 2;
}

fn usageError(message: []const u8) u8 {
    std.debug.print("error: {s}\n", .{message});
    return 2;
}
