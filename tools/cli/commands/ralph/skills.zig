//! ralph skills — list/add/clear persisted skills.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const skills_store = @import("skills_store.zig");
const cfg = @import("config.zig");

pub fn runSkills(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    if (args.len == 0) return skillsList(allocator, io);

    const subcmd = std.mem.sliceTo(args[0], 0);
    if (std.mem.eql(u8, subcmd, "list") or std.mem.eql(u8, subcmd, "ls")) {
        return skillsList(allocator, io);
    } else if (std.mem.eql(u8, subcmd, "add")) {
        if (args.len < 2) {
            utils.output.println("Usage: abi ralph skills add <skill text>", .{});
            return;
        }
        return skillsAdd(allocator, io, std.mem.sliceTo(args[1], 0));
    } else if (std.mem.eql(u8, subcmd, "clear")) {
        return skillsClear(allocator, io);
    } else if (utils.args.matchesAny(subcmd, &[_][]const u8{ "--help", "-h", "help" })) {
        utils.output.println(
            \\Usage: abi ralph skills <subcommand>
            \\
            \\Manage persisted Ralph skills in .ralph/skills.jsonl.
            \\
            \\Subcommands:
            \\  list           Show persisted skills (default)
            \\  add <text>     Add a skill
            \\  clear          Remove all persisted skills
        , .{});
    } else {
        utils.output.printError("Unknown skills subcommand: {s}", .{subcmd});
    }
}

fn skillsList(allocator: std.mem.Allocator, io: std.Io) !void {
    const count = skills_store.countSkills(allocator, io);
    const list_text = try skills_store.listSkills(allocator, io);
    defer allocator.free(list_text);

    utils.output.printHeader("Ralph persisted skills");
    utils.output.printKeyValue("File", cfg.SKILLS_FILE);
    utils.output.printKeyValueFmt("Count", "{d}", .{count});
    utils.output.println("", .{});
    if (list_text.len == 0) {
        utils.output.println("No skills stored.", .{});
    } else {
        utils.output.print("{s}", .{list_text});
    }
}

fn skillsAdd(allocator: std.mem.Allocator, io: std.Io, text: []const u8) !void {
    try skills_store.appendSkill(allocator, io, text, null, 1.0);
    utils.output.printSuccess("Skill stored: {s}", .{text});
}

fn skillsClear(allocator: std.mem.Allocator, io: std.Io) !void {
    try skills_store.clearSkills(allocator, io);
    utils.output.printSuccess("Cleared persisted skills ({s}).", .{cfg.SKILLS_FILE});
}
