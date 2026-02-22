//! ralph super — one-shot: init if needed, run, optional gate

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const cfg = @import("config.zig");
const init_mod = @import("init.zig");
const run_mod = @import("run_loop.zig");
const gate_mod = @import("gate.zig");

pub fn runSuper(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var task_override: ?[]const u8 = null;
    var iter_override: ?usize = null;
    var auto_skill = false;
    var store_skill: ?[]const u8 = null;
    var do_gate = false;
    var config_path: []const u8 = cfg.CONFIG_FILE;
    var gate_in: ?[]const u8 = null;
    var gate_out: []const u8 = "reports/ralph_upgrade_summary.md";

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (utils.args.matchesAny(arg, &[_][]const u8{ "--task", "-t" })) {
            i += 1;
            if (i < args.len) task_override = std.mem.sliceTo(args[i], 0);
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--iterations", "-i" })) {
            i += 1;
            if (i < args.len) iter_override = std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10) catch null;
        } else if (std.mem.eql(u8, arg, "--auto-skill")) {
            auto_skill = true;
        } else if (std.mem.eql(u8, arg, "--store-skill")) {
            i += 1;
            if (i < args.len) store_skill = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--gate")) {
            do_gate = true;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--config", "-c" })) {
            i += 1;
            if (i < args.len) config_path = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--gate-in")) {
            i += 1;
            if (i < args.len) gate_in = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--gate-out")) {
            i += 1;
            if (i < args.len) gate_out = std.mem.sliceTo(args[i], 0);
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            std.debug.print(
                \\Usage: abi ralph super [options]
                \\
                \\Super Ralph: init workspace if missing, run the loop, optionally run quality gate.
                \\Power-user one-shot for autonomous multi-step tasks with optional verification.
                \\
                \\Options:
                \\  -t, --task <text>      Task (default: PROMPT.md)
                \\  -i, --iterations <n>   Max loop iterations
                \\  -c, --config <path>    Config file (default: ralph.yml)
                \\      --auto-skill       Extract and store a skill after the run
                \\      --store-skill <s>  Manually store a skill string after run
                \\      --gate             Run quality gate after the run (--in/--out apply)
                \\      --gate-in <path>   Gate input JSON/report (default: latest .ralph run report)
                \\      --gate-out <path>  Gate output Markdown
                \\  -h, --help             Show this help
                \\
            , .{});
            return;
        }
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();
    const cmd_ctx = context_mod.CommandContext{
        .allocator = allocator,
        .io = io,
    };

    const has_workspace = cfg.fileExists(io, cfg.STATE_FILE) or cfg.fileExists(io, config_path);
    if (!has_workspace) {
        std.debug.print("No Ralph workspace found. Running init...\n", .{});
        try init_mod.runInit(&cmd_ctx, &[_][:0]const u8{});
    }

    // Build run args and invoke run.
    // Allocated sentinel strings must live until after runRun returns,
    // so we collect them for deferred cleanup instead of freeing inside each if-block.
    var run_args = std.ArrayListUnmanaged([:0]const u8).empty;
    defer run_args.deinit(allocator);
    var to_free = std.ArrayListUnmanaged([:0]const u8).empty;
    defer {
        for (to_free.items) |s| allocator.free(s);
        to_free.deinit(allocator);
    }

    if (task_override) |t| {
        const tz = try allocator.dupeZ(u8, t);
        try to_free.append(allocator, tz);
        try run_args.appendSlice(allocator, &[_][:0]const u8{ "--task", tz });
    }
    if (iter_override) |n| {
        const s = try std.fmt.allocPrintSentinel(allocator, "{d}", .{n}, 0);
        try to_free.append(allocator, s);
        try run_args.append(allocator, "--iterations");
        try run_args.append(allocator, s);
    }
    if (auto_skill) try run_args.append(allocator, "--auto-skill");
    if (store_skill) |s| {
        const store_z = try allocator.dupeZ(u8, s);
        try to_free.append(allocator, store_z);
        try run_args.appendSlice(allocator, &[_][:0]const u8{ "--store-skill", store_z });
    }
    if (!std.mem.eql(u8, config_path, cfg.CONFIG_FILE)) {
        const config_z = try allocator.dupeZ(u8, config_path);
        try to_free.append(allocator, config_z);
        try run_args.appendSlice(allocator, &[_][:0]const u8{ "--config", config_z });
    }
    try run_mod.runRun(&cmd_ctx, run_args.items);

    if (do_gate) {
        std.debug.print("\nRunning quality gate...\n", .{});
        const gate_out_z = try allocator.dupeZ(u8, gate_out);
        var gate_args = std.ArrayListUnmanaged([:0]const u8).empty;
        var gate_free = std.ArrayListUnmanaged([:0]const u8).empty;
        defer {
            for (gate_free.items) |s| allocator.free(s);
            gate_free.deinit(allocator);
            gate_args.deinit(allocator);
        }
        try gate_free.append(allocator, gate_out_z);

        if (gate_in) |in_path| {
            const gate_in_z = try allocator.dupeZ(u8, in_path);
            try gate_free.append(allocator, gate_in_z);
            try gate_args.appendSlice(allocator, &[_][:0]const u8{ "--in", gate_in_z });
        }
        try gate_args.appendSlice(allocator, &[_][:0]const u8{ "--out", gate_out_z });
        try gate_mod.runGate(&cmd_ctx, gate_args.items);
    }
}
