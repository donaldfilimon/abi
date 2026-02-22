//! ralph improve — autonomous self-improvement loop with guardrails.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const cfg = @import("config.zig");
const runtime = @import("runtime.zig");

pub fn runImprove(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var config_path: []const u8 = cfg.CONFIG_FILE;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (utils.args.matchesAny(arg, &[_][]const u8{ "--config", "-c" })) {
            i += 1;
            if (i < args.len) config_path = std.mem.sliceTo(args[i], 0);
        }
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var loaded_cfg = cfg.RalphConfig{};
    const cfg_contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        config_path,
        allocator,
        .limited(64 * 1024),
    ) catch null;
    defer if (cfg_contents) |text| allocator.free(text);
    if (cfg_contents) |text| {
        cfg.parseRalphYamlInto(text, &loaded_cfg);
    }

    var task: []const u8 =
        "Review ABI source and apply deterministic improvements with Zig 0.16 compatibility. " ++
        "Always run zig build verify-all after each change.";
    var analysis_only = false;
    var max_iterations = loaded_cfg.max_iterations;
    var max_fix_attempts = loaded_cfg.max_fix_attempts;
    var strict_backend = loaded_cfg.llm_strict_backend;
    var model = loaded_cfg.llm_model;
    var plugin = loaded_cfg.llm_plugin;
    var worktree: []const u8 = ".";
    var require_clean_tree = loaded_cfg.require_clean_tree;
    var backend: ?abi.ai.llm.providers.ProviderId = abi.ai.llm.providers.ProviderId.fromString(loaded_cfg.llm_backend);

    var fallback_buf = std.ArrayListUnmanaged(abi.ai.llm.providers.ProviderId).empty;
    defer fallback_buf.deinit(allocator);
    if (loaded_cfg.llm_fallback.len > 0) {
        try appendProvidersCsv(allocator, &fallback_buf, loaded_cfg.llm_fallback);
    }

    i = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            printHelp();
            return;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--task", "-t" })) {
            i += 1;
            if (i < args.len) task = std.mem.sliceTo(args[i], 0);
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--iterations", "-i" })) {
            i += 1;
            if (i < args.len) max_iterations = std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10) catch max_iterations;
        } else if (std.mem.eql(u8, arg, "--max-fix-attempts")) {
            i += 1;
            if (i < args.len) max_fix_attempts = std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10) catch max_fix_attempts;
        } else if (std.mem.eql(u8, arg, "--analysis-only")) {
            analysis_only = true;
        } else if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i < args.len) {
                const raw = std.mem.sliceTo(args[i], 0);
                backend = abi.ai.llm.providers.ProviderId.fromString(raw) orelse {
                    std.debug.print("Unknown provider backend: {s}\n", .{raw});
                    return;
                };
            }
        } else if (std.mem.eql(u8, arg, "--fallback")) {
            i += 1;
            if (i < args.len) {
                fallback_buf.clearRetainingCapacity();
                try appendProvidersCsv(allocator, &fallback_buf, std.mem.sliceTo(args[i], 0));
            }
        } else if (std.mem.eql(u8, arg, "--strict-backend")) {
            strict_backend = true;
        } else if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i < args.len) model = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--plugin")) {
            i += 1;
            if (i < args.len) plugin = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--worktree")) {
            i += 1;
            if (i < args.len) worktree = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--require-clean-tree")) {
            require_clean_tree = true;
        } else if (std.mem.eql(u8, arg, "--allow-dirty-tree")) {
            require_clean_tree = false;
        }
    }

    const fallback_slice = try fallback_buf.toOwnedSlice(allocator);
    defer allocator.free(fallback_slice);

    std.debug.print(
        "Starting Ralph improve (mode={s}, iterations={d}, backend={s})\n",
        .{
            if (analysis_only) "analysis-only" else "autonomous-apply",
            max_iterations,
            if (backend) |b| b.label() else "auto(local-first)",
        },
    );

    const summary = try runtime.runImprove(allocator, io, .{
        .task = task,
        .analysis_only = analysis_only,
        .max_iterations = max_iterations,
        .max_fix_attempts = max_fix_attempts,
        .backend = backend,
        .fallback = fallback_slice,
        .strict_backend = strict_backend,
        .model = model,
        .plugin = plugin,
        .worktree = worktree,
        .require_clean_tree = require_clean_tree,
    });
    defer allocator.free(summary.run_id);

    std.debug.print(
        "Ralph improve complete. run_id={s} iterations={d} passing={d} gate_passed={s} exit={d}\n",
        .{
            summary.run_id,
            summary.iterations,
            summary.passing_iterations,
            if (summary.last_gate_passed) "true" else "false",
            summary.last_gate_exit,
        },
    );
}

fn appendProvidersCsv(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(abi.ai.llm.providers.ProviderId),
    csv: []const u8,
) !void {
    var split = std.mem.splitScalar(u8, csv, ',');
    while (split.next()) |raw_part| {
        const part = std.mem.trim(u8, raw_part, " \t\r\n");
        if (part.len == 0) continue;
        const provider = abi.ai.llm.providers.ProviderId.fromString(part) orelse continue;
        var already = false;
        for (out.items) |existing| {
            if (existing == provider) {
                already = true;
                break;
            }
        }
        if (already) continue;
        try out.append(allocator, provider);
    }
}

fn printHelp() void {
    std.debug.print(
        \\Usage: abi ralph improve [options]
        \\
        \\Autonomous Ralph improve loop with per-iteration verify-all gate.
        \\
        \\Options:
        \\  --analysis-only            Analyze only, do not enforce apply/verify commit flow
        \\  -t, --task <text>          Improvement task
        \\  -i, --iterations <n>       Max iterations (default from ralph.yml)
        \\      --max-fix-attempts <n> Max auto-fix attempts after a gate failure
        \\      --backend <id>         Provider backend id (llama_cpp, mlx, ollama, lm_studio, vllm, anthropic, openai, plugin_http, plugin_native)
        \\      --fallback <csv>       Fallback provider chain
        \\      --strict-backend       Disable fallback and fail fast on selected backend
        \\      --model <id|path>      Model id or local path
        \\      --plugin <id>          Plugin id for plugin providers
        \\      --worktree <path>      Working tree path (default: .)
        \\      --require-clean-tree   Require clean git tree for autonomous apply (default)
        \\      --allow-dirty-tree     Allow dirty tree execution
        \\  -c, --config <path>        Ralph config file (default: ralph.yml)
        \\  -h, --help                 Show this help
        \\
    , .{});
}
