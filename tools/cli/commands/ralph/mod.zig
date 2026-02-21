//! Ralph orchestrator command — native Zig 0.16 replacement for shell/Python scripts.
//!
//! Subcommands:
//!   ralph init     — Create workspace: ralph.yml, .ralph/, PROMPT.md
//!   ralph run      — Execute iterative loop via Abbey engine
//!   ralph status   — Show loop state, skills stored, last run stats
//!   ralph gate     — Native quality gate (replaces check_ralph_gate.sh + score_ralph_results.py)
//!   ralph improve  — Self-improvement: analyze source, identify issues, extract lessons
//!   ralph skills   — List/add/clear stored skills
//!   ralph super    — Init if needed, run, optional gate (power one-shot)
//!   ralph multi    — Zig-native multithreaded multi-agent (ThreadPool + RalphBus)

const std = @import("std");
const utils = @import("../../utils/mod.zig");

const init_mod = @import("init.zig");
const run_mod = @import("run_loop.zig");
const status_mod = @import("status.zig");
const gate_mod = @import("gate.zig");
const improve_mod = @import("improve.zig");
const skills_mod = @import("skills.zig");
const super_mod = @import("super.zig");
const multi_mod = @import("multi.zig");

// ============================================================================
// Subcommand dispatch (mirrors config.zig pattern)
// ============================================================================

fn ralphInit(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try init_mod.runInit(allocator, parser.remaining());
}
fn ralphRun(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try run_mod.runRun(allocator, parser.remaining());
}
fn ralphStatus(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try status_mod.runStatus(allocator, parser.remaining());
}
fn ralphGate(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try gate_mod.runGate(allocator, parser.remaining());
}
fn ralphImprove(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try improve_mod.runImprove(allocator, parser.remaining());
}
fn ralphSkills(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try skills_mod.runSkills(allocator, parser.remaining());
}
fn ralphSuper(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try super_mod.runSuper(allocator, parser.remaining());
}
fn ralphMulti(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try multi_mod.runMulti(allocator, parser.remaining());
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
