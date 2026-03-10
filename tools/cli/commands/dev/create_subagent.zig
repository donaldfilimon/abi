//! Create Cursor subagent files (.cursor/agents/*.md or ~/.cursor/agents/*.md).
//!
//! Writes a markdown file with YAML frontmatter (name, description) and an optional body.
//! Project scope: .cursor/agents/<name>.md. User scope: ~/.cursor/agents/<name>.md.

const std = @import("std");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");

pub const meta: command_mod.Meta = .{
    .name = "create-subagent",
    .description = "Create a Cursor subagent file (project or user scope) with name, description, and optional prompt body",
    .aliases = &.{"subagent"},
    .subcommands = &.{ "help", "project", "user" },
};

const default_body =
    \\You are a specialized subagent. When invoked, follow the workflow below.
    \\
    \\When invoked:
    \\1. Understand the user's request
    \\2. Follow the steps needed to complete the task
    \\3. Provide clear, actionable output
    \\
;

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    var project_scope = true; // default: project
    var name_buf: []const u8 = "";
    var desc_buf: []const u8 = "Custom subagent for specialized tasks.";
    var body: []const u8 = default_body;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (std.mem.eql(u8, arg, "--user") or std.mem.eql(u8, arg, "-u")) {
            project_scope = false;
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--project") or std.mem.eql(u8, arg, "-p")) {
            project_scope = true;
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--name") or std.mem.eql(u8, arg, "-n")) {
            i += 1;
            if (i < args.len) {
                name_buf = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, arg, "--description") or std.mem.eql(u8, arg, "-d")) {
            i += 1;
            if (i < args.len) {
                desc_buf = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, arg, "--body")) {
            i += 1;
            if (i < args.len) {
                body = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (arg.len > 0 and arg[0] != '-') {
            if (name_buf.len == 0) name_buf = arg;
        }
        i += 1;
    }

    if (name_buf.len == 0) {
        utils.output.printError("Missing subagent name. Use --name <name> or pass name as first argument.", .{});
        printHelp();
        return error.MissingName;
    }

    // Normalize name: lowercase, hyphens only
    const name_owned = try normalizeName(allocator, name_buf);
    defer allocator.free(name_owned);

    const dir_path = if (project_scope) try getProjectAgentsDir(allocator) else try getUserAgentsDir(allocator);
    defer allocator.free(dir_path);

    const io = ctx.io;
    const file_path = try std.fmt.allocPrint(allocator, "{s}/{s}.md", .{ dir_path, name_owned });
    defer allocator.free(file_path);

    const content = try std.fmt.allocPrint(allocator,
        \\---
        \\name: {s}
        \\description: {s}
        \\---
        \\
        \\{s}
        \\
    , .{ name_owned, desc_buf, body });
    defer allocator.free(content);

    var mkdir_child = try std.process.spawn(io, .{
        .argv = &.{ "mkdir", "-p", dir_path },
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .ignore,
    });
    const mkdir_term = try mkdir_child.wait(io);
    switch (mkdir_term) {
        .exited => |code| if (code != 0) return error.MkdirFailed,
        else => return error.MkdirFailed,
    }

    const dir = std.Io.Dir.cwd();
    try dir.writeFile(io, .{ .sub_path = file_path, .data = content });

    utils.output.printInfo("Created subagent: {s}", .{file_path});
}

fn normalizeName(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    // Count valid chars first (strip path separators and dots to prevent traversal)
    var count: usize = 0;
    for (raw) |c| {
        if (c != '/' and c != '\\' and c != '.') count += 1;
    }
    if (count == 0) return error.MissingName;

    const out = try allocator.alloc(u8, count);
    var i: usize = 0;
    for (raw) |c| {
        if (c == '/' or c == '\\' or c == '.') continue;
        out[i] = if (c == ' ' or c == '_') '-' else std.ascii.toLower(c);
        i += 1;
    }
    return out;
}

fn getProjectAgentsDir(allocator: std.mem.Allocator) ![]const u8 {
    return std.fs.path.join(allocator, &.{ ".cursor", "agents" });
}

fn getUserAgentsDir(allocator: std.mem.Allocator) ![]const u8 {
    const home_ptr = std.c.getenv("HOME") orelse std.c.getenv("USERPROFILE") orelse return error.NoHome;
    const home = std.mem.sliceTo(home_ptr, 0);
    return std.fs.path.join(allocator, &.{ home, ".cursor", "agents" });
}

fn printHelp() void {
    std.debug.print(
        \\Usage: abi create-subagent [options] [name]
        \\       abi subagent [options] [name]
        \\
        \\Create a Cursor subagent .md file with YAML frontmatter (name, description) and body.
        \\
        \\Options:
        \\  --project, -p     Write to .cursor/agents/ (default)
        \\  --user, -u       Write to ~/.cursor/agents/
        \\  --name, -n NAME  Subagent name (lowercase, hyphens)
        \\  --description, -d DESC  When to delegate (required for good discovery)
        \\  --body TEXT      Optional prompt body (default template if omitted)
        \\  --help, -h       Show this help
        \\
        \\Example:
        \\  abi create-subagent -n my-helper -d "Use for X and Y. Use proactively when Z."
        \\
    , .{});
}
