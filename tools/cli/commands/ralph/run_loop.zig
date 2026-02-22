//! ralph run — Execute iterative loop via Abbey engine

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const cfg = @import("config.zig");
const skills_store = @import("skills_store.zig");

pub fn runRun(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var task_override: ?[]const u8 = null;
    var iter_override: ?usize = null;
    var auto_skill = false;
    var store_skill: ?[]const u8 = null;
    var config_path: []const u8 = cfg.CONFIG_FILE;

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
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--config", "-c" })) {
            i += 1;
            if (i < args.len) config_path = std.mem.sliceTo(args[i], 0);
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            std.debug.print(
                \\Usage: abi ralph run [options]
                \\
                \\Execute the Ralph iterative loop via the Abbey engine.
                \\
                \\Options:
                \\  -t, --task <text>      Override task (default: reads PROMPT.md)
                \\  -i, --iterations <n>   Override max_iterations from ralph.yml
                \\  -c, --config <path>    Config file (default: ralph.yml)
                \\      --auto-skill       Extract and store a skill after the run
                \\      --store-skill <s>  Manually store a skill string after run
                \\  -h, --help             Show this help
                \\
            , .{});
            return;
        }
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Parse ralph.yml
    var ralph_cfg = cfg.RalphConfig{};
    const yml_contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        config_path,
        allocator,
        .limited(64 * 1024),
    ) catch |err| {
        std.debug.print("Cannot read {s}: {t}\n", .{ config_path, err });
        std.debug.print("Run 'abi ralph init' to create a workspace.\n", .{});
        return;
    };
    defer allocator.free(yml_contents);
    cfg.parseRalphYamlInto(yml_contents, &ralph_cfg);

    const max_iterations = iter_override orelse ralph_cfg.max_iterations;

    // Determine goal: --task flag or PROMPT.md contents
    var prompt_owned: ?[]u8 = null;
    defer if (prompt_owned) |p| allocator.free(p);

    const goal: []const u8 = if (task_override) |t|
        t
    else blk: {
        const content = std.Io.Dir.cwd().readFileAlloc(
            io,
            ralph_cfg.prompt_file,
            allocator,
            .limited(256 * 1024),
        ) catch |err| {
            std.debug.print("Cannot read {s}: {t}\n", .{ ralph_cfg.prompt_file, err });
            std.debug.print("Use --task or edit {s}.\n", .{ralph_cfg.prompt_file});
            return;
        };
        prompt_owned = content;
        break :blk content;
    };

    std.debug.print("Starting Ralph loop (backend: {s}, max_iterations: {d})\n", .{
        ralph_cfg.backend, max_iterations,
    });
    std.debug.print("Goal: {s}\n\n", .{goal});

    // Create Abbey engine and run loop
    var engine = abi.ai.abbey.createEngine(allocator) catch |err| {
        std.debug.print("Failed to create Abbey engine: {t}\n", .{err});
        return;
    };
    defer engine.deinit();

    const result = engine.runRalphLoop(goal, max_iterations) catch |err| {
        std.debug.print("Ralph loop failed: {t}\n", .{err});
        return;
    };
    defer allocator.free(result);

    engine.recordRalphRun(goal, max_iterations, result.len, 1.0) catch {};

    // Handle skill storage
    var skills_added: u64 = 0;
    if (store_skill) |s| {
        _ = engine.storeSkill(s) catch {};
        skills_store.appendSkill(allocator, io, s, null, 1.0) catch |err| {
            std.debug.print("Warning: could not persist skill: {t}\n", .{err});
        };
        skills_added += 1;
        std.debug.print("Skill stored.\n", .{});
    }
    if (auto_skill) {
        const stored = engine.extractAndStoreSkill(goal, result) catch false;
        if (stored) {
            if (firstSentence(result)) |lesson| {
                skills_store.appendSkill(allocator, io, lesson, null, 0.8) catch {};
            }
            skills_added += 1;
            std.debug.print("Auto-skill extracted and stored.\n", .{});
        }
    }

    // Update state.json
    var state = cfg.readState(allocator, io);
    state.runs += 1;
    state.skills += skills_added;
    state.last_run_ts = blk: {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);
        break :blk @intCast(ts.sec);
    };
    cfg.writeState(allocator, io, state);

    std.debug.print("\n=== Ralph Run Complete ===\n{s}\n", .{result});
}

fn firstSentence(text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return null;
    for (trimmed, 0..) |ch, idx| {
        if (ch == '.' or ch == '\n') return std.mem.trim(u8, trimmed[0 .. idx + 1], " \t\r\n");
    }
    return trimmed;
}
