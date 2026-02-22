//! Ralph orchestrator command — native Zig 0.16 replacement for shell/Python scripts.
//!
//! Subcommands:
//!   ralph init     — Create workspace: ralph.yml, .ralph/, PROMPT.md
//!   ralph run      — Execute iterative loop via Abbey engine
//!   ralph status   — Show loop state, skills stored, last run stats
//!   ralph gate     — Native quality gate (replaces check_ralph_gate.sh + score_ralph_results.py)
//!   ralph improve  — Autonomous apply loop with per-iteration verify-all + commits
//!   ralph skills   — List/add/clear persisted skills (.ralph/skills.jsonl)
//!   ralph super    — Init if needed, run, optional gate (power one-shot)
//!   ralph multi    — Zig-native multithreaded multi-agent (ThreadPool + RalphBus)

const std = @import("std");
const command_mod = @import("../../command.zig");
const utils = @import("../../utils/mod.zig");

const init_mod = @import("init.zig");
const run_mod = @import("run_loop.zig");
const status_mod = @import("status.zig");
const gate_mod = @import("gate.zig");
const improve_mod = @import("improve.zig");
const skills_mod = @import("skills.zig");
const super_mod = @import("super.zig");
const multi_mod = @import("multi.zig");

// Wrapper functions for comptime children dispatch
fn wrapInit(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try ralphInit(allocator, &parser);
}
fn wrapRun(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try ralphRun(allocator, &parser);
}
fn wrapSuper(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try ralphSuper(allocator, &parser);
}
fn wrapMulti(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try ralphMulti(allocator, &parser);
}
fn wrapStatus(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try ralphStatus(allocator, &parser);
}
fn wrapGate(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try ralphGate(allocator, &parser);
}
fn wrapImprove(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try ralphImprove(allocator, &parser);
}
fn wrapSkills(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try ralphSkills(allocator, &parser);
}

pub const meta: command_mod.Meta = .{
    .name = "ralph",
    .description = "Ralph orchestrator (init, run, super, multi, status, gate, improve, skills)",
    .subcommands = &.{ "init", "run", "super", "multi", "status", "gate", "improve", "skills", "help" },
    .children = &.{
        .{ .name = "init", .description = "Create workspace: ralph.yml, .ralph/, PROMPT.md", .handler = .{ .basic = wrapInit } },
        .{ .name = "run", .description = "Execute the Ralph iterative loop", .handler = .{ .basic = wrapRun } },
        .{ .name = "super", .description = "Init if needed, run, optional gate (power one-shot)", .handler = .{ .basic = wrapSuper } },
        .{ .name = "super-ralph", .description = "Init if needed, run, optional gate (power one-shot)", .handler = .{ .basic = wrapSuper } },
        .{ .name = "multi", .description = "Zig-native multithreaded multi-agent", .handler = .{ .basic = wrapMulti } },
        .{ .name = "swarm", .description = "Zig-native multithreaded multi-agent", .handler = .{ .basic = wrapMulti } },
        .{ .name = "status", .description = "Show loop state, skills stored, last run stats", .handler = .{ .basic = wrapStatus } },
        .{ .name = "gate", .description = "Native quality gate", .handler = .{ .basic = wrapGate } },
        .{ .name = "improve", .description = "Autonomous self-improvement loop", .handler = .{ .basic = wrapImprove } },
        .{ .name = "skills", .description = "List/add/clear persisted skills", .handler = .{ .basic = wrapSkills } },
        .{ .name = "skill", .description = "List/add/clear persisted skills", .handler = .{ .basic = wrapSkills } },
    },
};

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
const ralph_subcommands = [_][]const u8{
    "init", "run", "super", "multi", "status", "gate", "improve", "skills", "help",
};

/// Entry point called by CLI dispatcher.
/// Only reached when no child matches (help / unknown).
pub fn run(_: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printHelp();
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp();
        return;
    }
    // Unknown subcommand
    std.debug.print("Unknown ralph subcommand: {s}\n", .{cmd});
    if (utils.args.suggestCommand(cmd, &ralph_subcommands)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
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
        \\  improve    Autonomous self-improvement loop (analysis-only or apply mode)
        \\  skills     List/add/clear persisted skills
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
