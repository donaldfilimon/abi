//! Ralph orchestrator command — native Zig 0.16 replacement for shell/Python scripts.
//!
//! Subcommands:
//!   ralph init     — Create workspace: ralph.yml, .ralph/, PROMPT.md
//!   ralph run      — Execute iterative loop via Abbey engine
//!   ralph status   — Show loop state, skills stored, last run stats
//!   ralph gate     — Native quality gate (replaces check_ralph_gate.sh + score_ralph_results.py)
//!   ralph improve  — Self-improvement: analyze source, identify issues, extract lessons
//!   ralph skills   — List/add/clear stored skills

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

// ============================================================================
// Subcommand dispatch (mirrors config.zig pattern)
// ============================================================================

fn ralphInit(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runInit(allocator, parser.remaining());
}
fn ralphRun(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runRun(allocator, parser.remaining());
}
fn ralphStatus(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runStatus(allocator, parser.remaining());
}
fn ralphGate(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runGate(allocator, parser.remaining());
}
fn ralphImprove(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runImprove(allocator, parser.remaining());
}
fn ralphSkills(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runSkills(allocator, parser.remaining());
}
fn ralphSuper(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runSuper(allocator, parser.remaining());
}
fn ralphMulti(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runMulti(allocator, parser.remaining());
}
fn ralphUnknown(cmd: []const u8) void {
    std.debug.print("Unknown ralph subcommand: {s}\n", .{cmd});
}
fn printHelpAlloc(_: std.mem.Allocator) void {
    printHelp();
}

const ralph_commands = [_]utils.subcommand.Command{
    .{ .names = &.{"init"}, .run = ralphInit },
    .{ .names = &.{"run"}, .run = ralphRun },
    .{ .names = &.{"status"}, .run = ralphStatus },
    .{ .names = &.{"gate"}, .run = ralphGate },
    .{ .names = &.{"improve"}, .run = ralphImprove },
    .{ .names = &.{ "skills", "skill" }, .run = ralphSkills },
    .{ .names = &.{ "super", "super-ralph" }, .run = ralphSuper },
    .{ .names = &.{ "multi", "swarm" }, .run = ralphMulti },
};

/// Entry point called by CLI dispatcher.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try utils.subcommand.runSubcommand(
        allocator,
        &parser,
        &ralph_commands,
        null,
        printHelpAlloc,
        ralphUnknown,
    );
}

// ============================================================================
// Constants and workspace helpers
// ============================================================================

const WORKSPACE_DIR = ".ralph";
const AGENT_DIR = ".ralph/agent";
const LOGS_DIR = ".ralph/diagnostics/logs";
const LOCK_FILE = ".ralph/loop.lock";
const STATE_FILE = ".ralph/state.json";
const CONFIG_FILE = "ralph.yml";
const PROMPT_FILE = "PROMPT.md";

/// Minimal fields parsed from ralph.yml line-by-line.
const RalphConfig = struct {
    backend: []const u8 = "claude",
    prompt_file: []const u8 = PROMPT_FILE,
    completion_promise: []const u8 = "LOOP_COMPLETE",
    max_iterations: usize = 100,
};

/// Parse ralph.yml into config. Returned strings are slices into `contents`.
fn parseRalphYamlInto(contents: []const u8, out: *RalphConfig) void {
    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;

        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const key = std.mem.trim(u8, line[0..colon], " \t");
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t\"");

        if (std.mem.eql(u8, key, "backend")) out.backend = value;
        if (std.mem.eql(u8, key, "prompt_file")) out.prompt_file = value;
        if (std.mem.eql(u8, key, "completion_promise")) out.completion_promise = value;
        if (std.mem.eql(u8, key, "max_iterations")) {
            out.max_iterations = std.fmt.parseInt(usize, value, 10) catch out.max_iterations;
        }
    }
}

fn ensureDir(io: std.Io, path: []const u8) void {
    std.Io.Dir.cwd().createDirPath(io, path) catch {};
}

fn fileExists(io: std.Io, path: []const u8) bool {
    _ = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    return true;
}

fn writeFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    content: []const u8,
) !void {
    _ = allocator;
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, content);
}

/// Read and parse state.json, or return defaults on any failure.
const LoopState = struct {
    runs: u64 = 0,
    skills: u64 = 0,
    last_run_ts: i64 = 0,
};

fn readState(allocator: std.mem.Allocator, io: std.Io) LoopState {
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        STATE_FILE,
        allocator,
        .limited(64 * 1024),
    ) catch return .{};
    defer allocator.free(contents);

    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        contents,
        .{},
    ) catch return .{};
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return .{},
    };

    return .{
        .runs = if (obj.get("runs")) |v| switch (v) {
            .integer => |i| @intCast(@max(i, 0)),
            else => 0,
        } else 0,
        .skills = if (obj.get("skills")) |v| switch (v) {
            .integer => |i| @intCast(@max(i, 0)),
            else => 0,
        } else 0,
        .last_run_ts = if (obj.get("last_run_ts")) |v| switch (v) {
            .integer => |i| i,
            else => 0,
        } else 0,
    };
}

fn writeState(allocator: std.mem.Allocator, io: std.Io, state: LoopState) void {
    const json = std.fmt.allocPrint(
        allocator,
        "{{\"runs\":{d},\"skills\":{d},\"last_run_ts\":{d}}}",
        .{ state.runs, state.skills, state.last_run_ts },
    ) catch return;
    defer allocator.free(json);
    writeFile(allocator, io, STATE_FILE, json) catch {};
}

fn removeLockFile(io: std.Io) void {
    std.Io.Dir.cwd().deleteFile(io, LOCK_FILE) catch {};
}

/// Case-insensitive substring search.
fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var match = true;
        for (0..needle.len) |j| {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(needle[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

// ============================================================================
// ralph super — one-shot: init if needed, run, optional gate
// ============================================================================

fn runSuper(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var task_override: ?[]const u8 = null;
    var iter_override: ?usize = null;
    var auto_skill = false;
    var do_gate = false;
    var config_path: []const u8 = CONFIG_FILE;
    var gate_in: []const u8 = "reports/ralph_upgrade_results_openai.json";
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
                \\      --gate             Run quality gate after the run (--in/--out apply)
                \\      --gate-in <path>   Gate input JSON (default: reports/ralph_upgrade_results_openai.json)
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

    const has_workspace = fileExists(io, STATE_FILE) or fileExists(io, config_path);
    if (!has_workspace) {
        std.debug.print("No Ralph workspace found. Running init...\n", .{});
        try runInit(allocator, &[_][:0]const u8{});
    }

    // Build run args and invoke run
    var run_args = std.ArrayListUnmanaged([:0]const u8).empty;
    defer run_args.deinit(allocator);
    if (task_override) |t| {
        const tz = try allocator.dupeZ(u8, t);
        defer allocator.free(tz);
        try run_args.appendSlice(allocator, &[_][:0]const u8{ "--task", tz });
    }
    if (iter_override) |n| {
        const s = try std.fmt.allocPrintSentinel(allocator, "{d}", .{n}, 0);
        defer allocator.free(s);
        try run_args.append(allocator, "--iterations");
        try run_args.append(allocator, s);
    }
    if (auto_skill) try run_args.append(allocator, "--auto-skill");
    if (!std.mem.eql(u8, config_path, CONFIG_FILE)) {
        const config_z = try allocator.dupeZ(u8, config_path);
        defer allocator.free(config_z);
        try run_args.appendSlice(allocator, &[_][:0]const u8{ "--config", config_z });
    }
    try runRun(allocator, run_args.items);

    if (do_gate) {
        std.debug.print("\nRunning quality gate...\n", .{});
        const gate_in_z = try allocator.dupeZ(u8, gate_in);
        const gate_out_z = try allocator.dupeZ(u8, gate_out);
        defer allocator.free(gate_in_z);
        defer allocator.free(gate_out_z);
        const gate_args = [_][:0]const u8{ "--in", gate_in_z, "--out", gate_out_z };
        try runGate(allocator, &gate_args);
    }
}

// ============================================================================
// ralph multi — Zig-native multithreaded multi-agent (ThreadPool + RalphBus)
// ============================================================================

fn runMulti(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var tasks_list = std.ArrayListUnmanaged([]const u8).empty;
    defer tasks_list.deinit(allocator);
    var max_iterations: usize = 20;
    var workers: u32 = 0;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (utils.args.matchesAny(arg, &[_][]const u8{ "--task", "-t" })) {
            i += 1;
            if (i < args.len) try tasks_list.append(allocator, std.mem.sliceTo(args[i], 0));
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--iterations", "-i" })) {
            i += 1;
            if (i < args.len) max_iterations = std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10) catch max_iterations;
        } else if (std.mem.eql(u8, arg, "--workers")) {
            i += 1;
            if (i < args.len) workers = std.fmt.parseInt(u32, std.mem.sliceTo(args[i], 0), 10) catch 0;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            std.debug.print(
                \\Usage: abi ralph multi [options]
                \\
                \\Zig-native multithreaded multi-agent: N Ralph loops in parallel via ThreadPool + RalphBus.
                \\
                \\Options:
                \\  -t, --task <text>     Add a task (repeat for multiple agents)
                \\  -i, --iterations <n>  Max iterations per agent (default: 20)
                \\      --workers <n>     Thread pool size (default: CPU count)
                \\  -h, --help            Show this help
                \\
            , .{});
            return;
        }
    }

    if (tasks_list.items.len == 0) {
        std.debug.print("No tasks. Use -t/--task at least once (e.g. abi ralph multi -t \"goal1\" -t \"goal2\").\n", .{});
        return;
    }

    const goals = tasks_list.items;
    var bus = try abi.ai.abbey.ralph_multi.RalphBus.init(allocator, 64);
    defer bus.deinit();

    var pool = try abi.runtime.ThreadPool.init(allocator, .{
        .thread_count = if (workers > 0) workers else 0,
    });
    defer pool.deinit();

    const results = try allocator.alloc(?[]const u8, goals.len);
    defer {
        for (results) |r| if (r) |s| allocator.free(s);
        allocator.free(results);
    }
    for (results) |*r| r.* = null;

    var ctx: abi.ai.abbey.ralph_swarm.ParallelRalphContext = .{
        .allocator = allocator,
        .bus = &bus,
        .goals = goals,
        .results = results,
        .max_iterations = max_iterations,
        .post_result_to_bus = true,
    };

    std.debug.print("Ralph multi: {d} agents, {d} threads, {d} iterations each.\n", .{
        goals.len,
        pool.thread_count,
        max_iterations,
    });

    for (goals, 0..) |_, idx| {
        const uidx: u32 = @intCast(idx);
        if (!pool.schedule(abi.ai.abbey.ralph_swarm.parallelRalphWorker, .{ &ctx, uidx })) {
            std.debug.print("Schedule failed for agent {d}\n", .{idx});
            return;
        }
    }
    pool.waitIdle();

    std.debug.print("\n=== Results ===\n", .{});
    for (results, 0..) |r, idx| {
        std.debug.print("--- Agent {d} ---\n", .{idx});
        if (r) |s| std.debug.print("{s}\n", .{s}) else std.debug.print("(failed)\n", .{});
    }

    var msg_count: usize = 0;
    while (bus.tryRecv()) |msg| {
        msg_count += 1;
        std.debug.print("[bus] from={d} to={d} kind={s}: {s}\n", .{
            msg.from_id,
            msg.to_id,
            @tagName(msg.kind),
            msg.getContent(),
        });
    }
    if (msg_count > 0) std.debug.print("Bus messages: {d}\n", .{msg_count});
}

// ============================================================================
// ralph init
// ============================================================================

const RALPH_YML_TEMPLATE =
    \\# Ralph Orchestrator Configuration
    \\# Generated by: abi ralph init
    \\
    \\cli:
    \\  backend: "{s}"
    \\
    \\event_loop:
    \\  prompt_file: "PROMPT.md"
    \\  completion_promise: "LOOP_COMPLETE"
    \\  max_iterations: 100
    \\
;

const PROMPT_MD_TEMPLATE =
    \\# Ralph Task
    \\
    \\<!-- Replace this with your task description. -->
    \\
    \\## Goal
    \\
    \\Describe what you want Ralph to accomplish here.
    \\
    \\## Acceptance Criteria
    \\
    \\- [ ] Criterion 1
    \\- [ ] Criterion 2
    \\
;

fn runInit(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var backend: []const u8 = "claude";
    var force = false;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (std.mem.eql(u8, arg, "--backend") or std.mem.eql(u8, arg, "-b")) {
            i += 1;
            if (i < args.len) backend = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--force") or std.mem.eql(u8, arg, "-f")) {
            force = true;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            std.debug.print(
                \\Usage: abi ralph init [options]
                \\
                \\Create a Ralph workspace in the current directory.
                \\
                \\Options:
                \\  -b, --backend <name>  LLM backend (default: claude)
                \\  -f, --force           Overwrite existing workspace
                \\  -h, --help            Show this help
                \\
            , .{});
            return;
        }
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    if (fileExists(io, WORKSPACE_DIR ++ "/state.json") and !force) {
        std.debug.print(
            "Workspace already exists (.ralph/). Use --force to reinitialize.\n",
            .{},
        );
        return;
    }

    // Create directory tree
    ensureDir(io, WORKSPACE_DIR);
    ensureDir(io, AGENT_DIR);
    ensureDir(io, LOGS_DIR);

    // Write ralph.yml
    const yml = try std.fmt.allocPrint(allocator, RALPH_YML_TEMPLATE, .{backend});
    defer allocator.free(yml);
    try writeFile(allocator, io, CONFIG_FILE, yml);

    // Write PROMPT.md (only if missing or force)
    if (!fileExists(io, PROMPT_FILE) or force) {
        try writeFile(allocator, io, PROMPT_FILE, PROMPT_MD_TEMPLATE);
    }

    // Write initial state
    writeState(allocator, io, .{});

    std.debug.print(
        \\Ralph workspace initialized.
        \\
        \\  ralph.yml   — configuration (backend: {s})
        \\  PROMPT.md   — edit this with your task
        \\  .ralph/     — runtime state directory
        \\
        \\Next steps:
        \\  1. Edit PROMPT.md with your task description
        \\  2. Run: abi ralph run
        \\
    , .{backend});
}

// ============================================================================
// ralph run
// ============================================================================

fn runRun(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var task_override: ?[]const u8 = null;
    var iter_override: ?usize = null;
    var auto_skill = false;
    var store_skill: ?[]const u8 = null;
    var config_path: []const u8 = CONFIG_FILE;

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
    var ralph_cfg = RalphConfig{};
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
    parseRalphYamlInto(yml_contents, &ralph_cfg);

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
        _ = engine.storeSkill(s) catch |err| {
            std.debug.print("Warning: could not store skill: {t}\n", .{err});
        };
        skills_added += 1;
        std.debug.print("Skill stored.\n", .{});
    }
    if (auto_skill) {
        const stored = engine.extractAndStoreSkill(goal, result) catch false;
        if (stored) {
            skills_added += 1;
            std.debug.print("Auto-skill extracted and stored.\n", .{});
        }
    }

    // Update state.json
    var state = readState(allocator, io);
    state.runs += 1;
    state.skills += skills_added;
    state.last_run_ts = abi.shared.utils.unixMs();
    writeState(allocator, io, state);

    std.debug.print("\n=== Ralph Run Complete ===\n{s}\n", .{result});
}

// ============================================================================
// ralph status
// ============================================================================

fn runStatus(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    for (args) |arg| {
        if (utils.args.matchesAny(std.mem.sliceTo(arg, 0), &[_][]const u8{ "--help", "-h", "help" })) {
            std.debug.print(
                \\Usage: abi ralph status
                \\
                \\Show Ralph workspace state, skill count, and active lock info.
                \\
            , .{});
            return;
        }
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    if (!fileExists(io, STATE_FILE)) {
        std.debug.print("No Ralph workspace found. Run 'abi ralph init' first.\n", .{});
        return;
    }

    const state = readState(allocator, io);
    const has_lock = fileExists(io, LOCK_FILE);

    // Get engine stats for live skill count
    var engine = abi.ai.abbey.createEngine(allocator) catch null;
    defer if (engine) |*e| e.deinit();

    std.debug.print("\nRalph Status\n", .{});
    std.debug.print("────────────────────────────────\n", .{});

    if (has_lock) {
        std.debug.print("Loop state:    RUNNING (lock present)\n", .{});
    } else {
        std.debug.print("Loop state:    idle\n", .{});
    }

    std.debug.print("Total runs:    {d}\n", .{state.runs});
    std.debug.print("Skills stored: {d}\n", .{state.skills});
    std.debug.print("Last run:      {d}\n", .{state.last_run_ts});

    if (engine) |*e| {
        const stats = e.getStats();
        std.debug.print("Memory items:  {d}\n", .{stats.memory_stats.semantic.knowledge_count});
        std.debug.print("LLM backend:   {s}\n", .{stats.llm_backend});
    }
    std.debug.print("\n", .{});
}

// ============================================================================
// ralph gate — native replacement for check_ralph_gate.sh + score_ralph_results.py
// ============================================================================

const ScoringRule = struct {
    name: []const u8,
    keywords: []const []const u8,
    min_len: usize,
};

const scoring_rules = [_]ScoringRule{
    .{
        .name = "vnext_policy",
        .keywords = &.{ "compat", "legacy", "vnext", "release" },
        .min_len = 80,
    },
    .{
        .name = "toolchain_determinism",
        .keywords = &.{ "zvm", ".zigversion", "PATH", "zig version" },
        .min_len = 80,
    },
    .{
        .name = "mod_stub_drift",
        .keywords = &.{ "mod", "stub", "parity", "compile" },
        .min_len = 80,
    },
    .{
        .name = "split_parity_validation",
        .keywords = &.{ "parity", "test", "behavior", "module" },
        .min_len = 80,
    },
    .{
        .name = "migration_mapping",
        .keywords = &.{ "Framework", "App", "Config", "Capability" },
        .min_len = 70,
    },
};

const ScoreResult = struct {
    score: f64 = 0.0,
    reasons: [8][]const u8 = undefined,
    reason_count: usize = 0,

    fn appendReason(self: *ScoreResult, r: []const u8) void {
        if (self.reason_count < 8) {
            self.reasons[self.reason_count] = r;
            self.reason_count += 1;
        }
    }
};

fn keywordScore(text: []const u8, keywords: []const []const u8) f64 {
    if (keywords.len == 0) return 0.0;
    var hits: f64 = 0.0;
    for (keywords) |kw| {
        if (containsIgnoreCase(text, kw)) hits += 1.0;
    }
    return hits / @as(f64, @floatFromInt(keywords.len));
}

fn scoreItem(
    output: []const u8,
    notes: []const u8,
    rule: ScoringRule,
    require_live: bool,
) ScoreResult {
    var res = ScoreResult{};

    const is_live = std.mem.indexOf(u8, notes, "provider=openai") != null and
        std.mem.indexOf(u8, notes, "placeholder") == null and
        std.mem.indexOf(u8, notes, "dry_run") == null;

    if (require_live and !is_live) {
        res.appendReason("not_live_openai_output");
    }

    const trimmed_len = std.mem.trim(u8, output, " \t\r\n").len;
    if (trimmed_len < rule.min_len) {
        res.appendReason("output_too_short");
    }

    var score = keywordScore(output, rule.keywords);
    if (trimmed_len >= rule.min_len) {
        score = (score + 1.0) / 2.0;
    }
    if (require_live and !is_live) {
        score *= 0.2;
    }
    res.score = score;
    return res;
}

fn runGate(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var in_path: []const u8 = "reports/ralph_upgrade_results_openai.json";
    var out_path: []const u8 = "reports/ralph_upgrade_summary.md";
    var min_average: f64 = 0.75;
    var require_live = false;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (utils.args.matchesAny(arg, &[_][]const u8{ "--in", "-i" })) {
            i += 1;
            if (i < args.len) in_path = std.mem.sliceTo(args[i], 0);
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--out", "-o" })) {
            i += 1;
            if (i < args.len) out_path = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--min-average")) {
            i += 1;
            if (i < args.len) min_average = std.fmt.parseFloat(f64, std.mem.sliceTo(args[i], 0)) catch min_average;
        } else if (std.mem.eql(u8, arg, "--require-live")) {
            require_live = true;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            std.debug.print(
                \\Usage: abi ralph gate [options]
                \\
                \\Native quality gate — replaces check_ralph_gate.sh + score_ralph_results.py.
                \\Scores JSON results against 5 keyword rules, writes Markdown summary.
                \\Exits 0 on pass, 2 on fail.
                \\
                \\Options:
                \\  -i, --in <path>         Input JSON (default: reports/ralph_upgrade_results_openai.json)
                \\  -o, --out <path>        Output Markdown (default: reports/ralph_upgrade_summary.md)
                \\      --min-average <f>   Minimum average score (default: 0.75)
                \\      --require-live      Require live OpenAI provider outputs
                \\  -h, --help              Show this help
                \\
            , .{});
            return;
        }
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Read input JSON
    const json_contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        in_path,
        allocator,
        .limited(16 * 1024 * 1024),
    ) catch {
        std.debug.print("ERROR: missing live Ralph results: {s}\n", .{in_path});
        std.debug.print("Run:\n  abi ralph run --task \"...\" --auto-skill\n", .{});
        std.process.exit(1);
    };
    defer allocator.free(json_contents);

    // Parse JSON as array
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json_contents,
        .{},
    ) catch {
        std.debug.print("Invalid JSON in {s}\n", .{in_path});
        std.process.exit(1);
    };
    defer parsed.deinit();

    const items = switch (parsed.value) {
        .array => |a| a.items,
        else => {
            std.debug.print("Results JSON must be a top-level array.\n", .{});
            std.process.exit(1);
        },
    };

    if (items.len == 0) {
        std.debug.print("Results file is empty.\n", .{});
        std.process.exit(1);
    }

    // Score each item against its matching rule
    const count = @min(items.len, scoring_rules.len);
    var total: f64 = 0.0;

    var rows = std.ArrayListUnmanaged(u8).empty;
    defer rows.deinit(allocator);

    for (0..count) |idx| {
        const item = switch (items[idx]) {
            .object => |o| o,
            else => continue,
        };
        const rule = scoring_rules[idx];

        const output_str = if (item.get("output")) |v| switch (v) {
            .string => |s| s,
            else => "",
        } else "";
        const notes_str = if (item.get("notes")) |v| switch (v) {
            .string => |s| s,
            else => "",
        } else "";

        const scored = scoreItem(output_str, notes_str, rule, require_live);
        total += scored.score;

        // Append markdown row
        try rows.appendSlice(allocator, "- `");
        try rows.appendSlice(allocator, rule.name);
        var score_buf: [32]u8 = undefined;
        const score_str = std.fmt.bufPrint(&score_buf, "`: {d:.3} (", .{scored.score}) catch "`: ? (";
        try rows.appendSlice(allocator, score_str);
        if (scored.reason_count == 0) {
            try rows.appendSlice(allocator, "ok");
        } else {
            for (scored.reasons[0..scored.reason_count], 0..) |r, ri| {
                if (ri > 0) try rows.appendSlice(allocator, ", ");
                try rows.appendSlice(allocator, r);
            }
        }
        try rows.appendSlice(allocator, ")\n");
    }

    const avg = total / @as(f64, @floatFromInt(@max(count, 1)));
    const passed = avg >= min_average;
    const status_str = if (passed) "PASS" else "FAIL";
    const live_text = if (require_live) "required" else "optional";

    const summary = try std.fmt.allocPrint(allocator,
        \\# Ralph Upgrade Score
        \\
        \\- Input: `{s}`
        \\- Live OpenAI outputs: {s}
        \\- Average score: {d:.3}
        \\- Threshold: {d:.3}
        \\- Result: **{s}**
        \\
        \\## Item scores
        \\{s}
    , .{ in_path, live_text, avg, min_average, status_str, rows.items });
    defer allocator.free(summary);

    // Ensure output parent directory exists, then write
    if (std.fs.path.dirname(out_path)) |dir| ensureDir(io, dir);
    writeFile(allocator, io, out_path, summary) catch |err| {
        std.debug.print("Warning: could not write {s}: {t}\n", .{ out_path, err });
    };

    std.debug.print("{s}\n", .{summary});

    if (!passed) {
        std.debug.print(
            "FAIL: Ralph gate did not pass (avg={d:.3} < threshold={d:.3})\n",
            .{ avg, min_average },
        );
        std.process.exit(2);
    }
    std.debug.print("OK: Ralph gate passed ({s}).\n", .{out_path});
}

// ============================================================================
// ralph improve — self-improvement loop
// ============================================================================

fn runImprove(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var task: []const u8 =
        "Review the ABI framework source in src/ for Zig 0.16 migration issues. " ++
        "Check mod.zig/stub.zig parity. Verify build flags in build/options.zig. " ++
        "Run zig build test mentally and identify likely failures. " ++
        "Produce a prioritized list of fixes with exact file paths.";
    var max_iterations: usize = 5;
    var auto_skill = true;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (utils.args.matchesAny(arg, &[_][]const u8{ "--task", "-t" })) {
            i += 1;
            if (i < args.len) task = std.mem.sliceTo(args[i], 0);
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--iterations", "-i" })) {
            i += 1;
            if (i < args.len) max_iterations = std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10) catch max_iterations;
        } else if (std.mem.eql(u8, arg, "--no-auto-skill")) {
            auto_skill = false;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            std.debug.print(
                \\Usage: abi ralph improve [options]
                \\
                \\Self-improvement loop: analyze source, identify issues, extract lesson.
                \\Auto-skill is enabled by default.
                \\
                \\Options:
                \\  -t, --task <text>      Custom improvement task
                \\  -i, --iterations <n>   Max iterations (default: 5)
                \\      --no-auto-skill    Disable automatic skill extraction
                \\  -h, --help             Show this help
                \\
            , .{});
            return;
        }
    }

    std.debug.print(
        "Starting Ralph self-improvement loop ({d} iterations)...\n",
        .{max_iterations},
    );
    std.debug.print("Task: {s}\n\n", .{task});

    var engine = abi.ai.abbey.createEngine(allocator) catch |err| {
        std.debug.print("Failed to create Abbey engine: {t}\n", .{err});
        return;
    };
    defer engine.deinit();

    const result = engine.runRalphLoop(task, max_iterations) catch |err| {
        std.debug.print("Improve loop failed: {t}\n", .{err});
        return;
    };
    defer allocator.free(result);

    engine.recordRalphRun(task, max_iterations, result.len, 1.0) catch {};

    if (auto_skill) {
        const stored = engine.extractAndStoreSkill(task, result) catch false;
        if (stored) {
            std.debug.print("Lesson extracted and stored in Abbey memory.\n\n", .{});
        }
    }

    std.debug.print("\n=== Improvement Analysis ===\n{s}\n", .{result});
}

// ============================================================================
// ralph skills — list/add/clear stored skills
// ============================================================================

fn runSkills(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) return skillsList(allocator);

    const subcmd = std.mem.sliceTo(args[0], 0);
    if (std.mem.eql(u8, subcmd, "list") or std.mem.eql(u8, subcmd, "ls")) {
        return skillsList(allocator);
    } else if (std.mem.eql(u8, subcmd, "add")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi ralph skills add <skill text>\n", .{});
            return;
        }
        return skillsAdd(allocator, std.mem.sliceTo(args[1], 0));
    } else if (std.mem.eql(u8, subcmd, "clear")) {
        return skillsClear(allocator);
    } else if (utils.args.matchesAny(subcmd, &[_][]const u8{ "--help", "-h", "help" })) {
        std.debug.print(
            \\Usage: abi ralph skills <subcommand>
            \\
            \\Manage skills stored in Abbey memory.
            \\Skills are injected into the system prompt for future Ralph runs.
            \\
            \\Subcommands:
            \\  list           Show skill count and stats (default)
            \\  add <text>     Store a new skill
            \\  clear          Reset all memory (removes all skills)
            \\
        , .{});
    } else {
        std.debug.print("Unknown skills subcommand: {s}\n", .{subcmd});
    }
}

fn skillsList(allocator: std.mem.Allocator) void {
    var engine = abi.ai.abbey.createEngine(allocator) catch |err| {
        std.debug.print("Failed to create Abbey engine: {t}\n", .{err});
        return;
    };
    defer engine.deinit();

    const stats = engine.getStats();
    std.debug.print("\nRalph Skills (Abbey memory)\n", .{});
    std.debug.print("────────────────────────────\n", .{});
    std.debug.print("Knowledge items: {d}\n", .{stats.memory_stats.semantic.knowledge_count});
    std.debug.print("LLM backend:     {s}\n\n", .{stats.llm_backend});
    std.debug.print("Skills are injected into the system prompt on the next 'ralph run'.\n", .{});
    std.debug.print("Add skills:      abi ralph skills add \"<lesson>\"\n", .{});
    std.debug.print("Auto-extract:    abi ralph run --auto-skill\n\n", .{});
}

fn skillsAdd(allocator: std.mem.Allocator, text: []const u8) void {
    var engine = abi.ai.abbey.createEngine(allocator) catch |err| {
        std.debug.print("Failed to create Abbey engine: {t}\n", .{err});
        return;
    };
    defer engine.deinit();

    const id = engine.storeSkill(text) catch |err| {
        std.debug.print("Failed to store skill: {t}\n", .{err});
        return;
    };
    std.debug.print("Skill stored (id={d}): {s}\n", .{ id, text });
}

fn skillsClear(allocator: std.mem.Allocator) void {
    var engine = abi.ai.abbey.createEngine(allocator) catch |err| {
        std.debug.print("Failed to create Abbey engine: {t}\n", .{err});
        return;
    };
    defer engine.deinit();

    engine.reset();
    std.debug.print("All Abbey memory (skills and history) cleared.\n", .{});
}

// ============================================================================
// Top-level help
// ============================================================================

fn printHelp() void {
    std.debug.print(
        \\Usage: abi ralph <subcommand> [options]
        \\
        \\Ralph orchestrator — iterative AI agent loop with skill memory.
        \\
        \\Subcommands:
        \\  init       Create workspace: ralph.yml, .ralph/, PROMPT.md
        \\  run        Execute the Ralph iterative loop
        \\  super      Init if needed, run, optional gate (power one-shot)
        \\  multi      Zig-native multithreaded multi-agent (ThreadPool + RalphBus)
        \\  status     Show loop state, skills stored, last run stats
        \\  gate       Native quality gate (replaces check_ralph_gate.sh)
        \\  improve    Self-improvement: analyze source, extract lessons
        \\  skills     List/add/clear stored skills
        \\
        \\Quick start:
        \\  abi ralph init              # Create workspace
        \\  abi ralph run               # Run task from PROMPT.md
        \\  abi ralph super --task "..." # One-shot: init + run (+ optional --gate, --auto-skill)
        \\  abi ralph multi -t "g1" -t "g2"  # Parallel agents (fast, Zig threads + lock-free bus)
        \\  abi ralph run --task "..."  # Inline task
        \\  abi ralph run --auto-skill  # Run + extract lesson
        \\  abi ralph gate              # Check quality gate
        \\  abi ralph improve           # Self-improvement pass
        \\  abi ralph skills            # Show stored skills
        \\
        \\Multi-Ralph: Lock-free RalphBus (ralph_multi) + parallel swarm (ralph_swarm) — Zig-native, fast multithreading.
        \\
        \\Run 'abi ralph <subcommand> help' for subcommand-specific help.
        \\
    , .{});
}
