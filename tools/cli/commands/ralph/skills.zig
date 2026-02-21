//! ralph skills — list/add/clear stored skills

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

pub fn runSkills(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
