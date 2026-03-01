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
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");

const init_mod = @import("init.zig");
const run_mod = @import("run_loop.zig");
const status_mod = @import("status.zig");
const gate_mod = @import("gate.zig");
const improve_mod = @import("improve.zig");
const skills_mod = @import("skills.zig");
const super_mod = @import("super.zig");
const multi_mod = @import("multi.zig");

pub const meta: command_mod.Meta = .{
    .name = "ralph",
    .description = "Ralph orchestrator (init, run, super, multi, status, gate, improve, skills)",
    .kind = .group,
    .subcommands = &.{ "init", "run", "super", "multi", "status", "gate", "improve", "skills", "help" },
    .children = &.{
        .{ .name = "init", .description = "Create workspace: ralph.yml, .ralph/, PROMPT.md", .handler = command_mod.contextParserHandler(ralphInit) },
        .{ .name = "run", .description = "Execute the Ralph iterative loop", .handler = command_mod.contextParserHandler(ralphRun) },
        .{ .name = "super", .description = "Init if needed, run, optional gate (power one-shot)", .handler = command_mod.contextParserHandler(ralphSuper) },
        .{ .name = "super-ralph", .description = "Init if needed, run, optional gate (power one-shot)", .handler = command_mod.contextParserHandler(ralphSuper) },
        .{ .name = "multi", .description = "Zig-native multithreaded multi-agent", .handler = command_mod.contextParserHandler(ralphMulti) },
        .{ .name = "swarm", .description = "Zig-native multithreaded multi-agent", .handler = command_mod.contextParserHandler(ralphMulti) },
        .{ .name = "status", .description = "Show loop state, skills stored, last run stats", .handler = command_mod.contextParserHandler(ralphStatus) },
        .{ .name = "gate", .description = "Native quality gate", .handler = command_mod.contextParserHandler(ralphGate) },
        .{ .name = "improve", .description = "Autonomous self-improvement loop", .handler = command_mod.contextParserHandler(ralphImprove) },
        .{ .name = "skills", .description = "List/add/clear persisted skills", .handler = command_mod.contextParserHandler(ralphSkills) },
        .{ .name = "skill", .description = "List/add/clear persisted skills", .handler = command_mod.contextParserHandler(ralphSkills) },
    },
};

// ============================================================================
// Subcommand dispatch (mirrors config.zig pattern)
// ============================================================================

fn ralphInit(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try init_mod.runInit(ctx, parser.remaining());
}
fn ralphRun(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try run_mod.runRun(ctx, parser.remaining());
}
fn ralphStatus(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try status_mod.runStatus(ctx, parser.remaining());
}
fn ralphGate(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try gate_mod.runGate(ctx, parser.remaining());
}
fn ralphImprove(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try improve_mod.runImprove(ctx, parser.remaining());
}
fn ralphSkills(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try skills_mod.runSkills(ctx, parser.remaining());
}
fn ralphSuper(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try super_mod.runSuper(ctx.allocator, parser.remaining());
}
fn ralphMulti(ctx: *const context_mod.CommandContext, parser: *utils.args.ArgParser) !void {
    try multi_mod.runMulti(ctx.allocator, parser.remaining());
}

/// Entry point called by CLI dispatcher.
/// Only reached when no child matches (help / unknown).
pub fn run(_: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
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
    utils.output.printError("Unknown ralph subcommand: {s}", .{cmd});
    if (command_mod.suggestSubcommand(meta, cmd)) |suggestion| {
        utils.output.printInfo("Did you mean: {s}", .{suggestion});
    }
}

// ============================================================================
// Top-level help
// ============================================================================

fn printHelp() void {
    utils.output.println(
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
    , .{});
}

test {
    std.testing.refAllDecls(@This());
}
