//! ralph status — Show loop state, skills stored, last run stats

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const cfg = @import("config.zig");

pub fn runStatus(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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

    if (!cfg.fileExists(io, cfg.STATE_FILE)) {
        std.debug.print("No Ralph workspace found. Run 'abi ralph init' first.\n", .{});
        return;
    }

    const state = cfg.readState(allocator, io);
    const has_lock = cfg.fileExists(io, cfg.LOCK_FILE);

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
