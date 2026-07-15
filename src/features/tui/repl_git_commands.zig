//! Git and session slash-command handlers for the REPL.
//!
//! Extracted from `repl.zig` so the interactive loop stays focused on I/O
//! and dispatch. These free functions take the needed fields as parameters
//! (allocator, state pointer, io) and perform the actual subprocess / file
//! work for `/diff`, `/commit`, `/sync-clis`, `/open`, and `/sessions`.
//! `repl.zig` calls them from thin wrapper methods on `ReplLoop`.

const std = @import("std");
const builtin = @import("builtin");
const env = @import("../../foundation/env.zig");
const cmds = @import("repl_commands.zig");
const repl_types = @import("repl_types.zig");
const repl_session = @import("repl_session.zig");
const file_context = @import("../ai/file_context.zig");

/// `/sync-clis`: execute the central sync-clis launcher from the operator's
/// Grok skill dir. Resolves the launcher path from the OS-appropriate home
/// env var and never executes a missing script.
pub fn runSyncClis(allocator: std.mem.Allocator, io: std.Io) !void {
    const home_var = cmds.homeEnvVarName(builtin.target.os.tag);
    const home = env.get(home_var) orelse {
        std.debug.print("sync-clis: HOME not set; cannot locate launcher\n", .{});
        return;
    };
    const maybe_launch_path = try cmds.syncClisLauncherPath(allocator, home);
    if (maybe_launch_path == null) {
        std.debug.print("sync-clis: HOME not set; cannot locate launcher\n", .{});
        return;
    }
    const launch_path = maybe_launch_path.?;
    defer allocator.free(launch_path);
    std.Io.Dir.cwd().access(io, launch_path, .{}) catch {
        std.debug.print("sync-clis: launcher not found at {s}\n", .{launch_path});
        return;
    };
    std.debug.print("sync-clis: executing central sync (full targets via driver)...\n", .{});
    var child = try std.process.spawn(io, .{
        .argv = &[_][]const u8{launch_path},
        .cwd = .inherit,
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    defer child.kill(io);
    const term = try child.wait(io);
    std.debug.print("sync-clis done (exit {any})\n", .{term});
}

/// `/open <path>`: read a file into the prompt context. The content is stored
/// in `state.file_context_buf` (packed header + path + content) and injected
/// before the next completion.
pub fn runOpen(allocator: std.mem.Allocator, state: *repl_types.ReplState, path: []const u8, io: std.Io) !void {
    if (path.len == 0) {
        if (state.open_path.len > 0) {
            var buf: [256]u8 = undefined;
            std.debug.print("{s}\n", .{cmds.formatOpenStatus(&buf, state.open_path, state.open_content.len)});
        } else {
            std.debug.print("usage: /open <path>\n", .{});
        }
        return;
    }
    state.clearFileContext(allocator);

    const max_read: usize = repl_types.OPEN_FILE_BUDGET_BYTES;
    const content = file_context.readFileBounded(io, allocator, ".", path, max_read) catch |err| {
        std.debug.print("open: cannot read '{s}': {s}\n", .{ path, @errorName(err) });
        return;
    };

    const path_bytes = std.mem.trim(u8, path, " \t\r");
    const header_len = @sizeOf(usize);
    const total_len = header_len + path_bytes.len + content.len;
    const buf = try allocator.alloc(u8, total_len);
    @memcpy(buf[0..header_len], std.mem.asBytes(&path_bytes.len));
    @memcpy(buf[header_len..][0..path_bytes.len], path_bytes);
    @memcpy(buf[header_len + path_bytes.len ..], content);

    state.file_context_buf = buf;
    const stored_path_len = std.mem.bytesAsSlice(usize, buf[0..header_len])[0];
    state.open_path = buf[header_len..][0..stored_path_len];
    state.open_content = buf[header_len + stored_path_len ..];

    var status_buf: [256]u8 = undefined;
    std.debug.print("{s}\n", .{cmds.formatOpenStatus(&status_buf, state.open_path, state.open_content.len)});
}

/// `/diff`: run `git diff` and print the output. When `--stat` is passed,
/// shows a summary of changed files. Output is colorized with ANSI codes for
/// added (+) lines in green and removed (-) lines in red.
pub fn runDiff(allocator: std.mem.Allocator, io: std.Io) !void {
    const arg = cmds.specialArg("/diff ");
    const want_stat = cmds.diffWantsStat(arg);
    const argv = cmds.diffArgv(want_stat);

    var child = std.process.spawn(io, .{
        .argv = argv,
        .cwd = .inherit,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .ignore,
    }) catch |err| {
        std.debug.print("diff: failed to run git diff: {s}\n", .{@errorName(err)});
        return;
    };
    defer child.kill(io);

    var output = std.ArrayListUnmanaged(u8).empty;
    defer output.deinit(allocator);
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = std.Io.File.readStreaming(child.stdout.?, io, &.{&buf}) catch break;
        if (n == 0) break;
        try output.appendSlice(allocator, buf[0..n]);
    }
    const term = try child.wait(io);

    if (output.items.len == 0) {
        std.debug.print("diff: working tree is clean\n", .{});
    } else if (want_stat) {
        std.debug.print("{s}\n", .{output.items});
    } else {
        const colored = try cmds.colorizeDiff(allocator, output.items);
        defer allocator.free(colored);
        std.debug.print("{s}", .{colored});
    }
    if (term != .exited or term.exited != 0) {
        std.debug.print("diff: git diff terminated: {any}\n", .{term});
    }
}

/// `/commit`: stage all changes and create a commit. Prompts for a commit
/// message interactively via the next REPL input line.
pub fn runCommit(allocator: std.mem.Allocator, io: std.Io) !void {
    var add_child = std.process.spawn(io, .{
        .argv = &[_][]const u8{ "git", "add", "-A" },
        .cwd = .inherit,
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .ignore,
    }) catch |err| {
        std.debug.print("commit: failed to stage: {s}\n", .{@errorName(err)});
        return;
    };
    _ = try add_child.wait(io);

    std.debug.print("commit message (end with Ctrl-D or empty line to cancel):\n> ", .{});

    var msg_buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &msg_buf);
    var msg_lines = std.ArrayListUnmanaged(u8).empty;
    defer msg_lines.deinit(allocator);

    while (true) {
        const maybe_line = stdin_reader.interface.takeDelimiter('\n') catch break;
        const line = (maybe_line orelse break);
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0 and msg_lines.items.len == 0) {
            std.debug.print("commit cancelled\n", .{});
            return;
        }
        if (trimmed.len == 0 and msg_lines.items.len > 0) break;
        if (msg_lines.items.len > 0) try msg_lines.append(allocator, '\n');
        try msg_lines.appendSlice(allocator, trimmed);
    }

    if (msg_lines.items.len == 0) {
        std.debug.print("commit cancelled\n", .{});
        return;
    }

    var commit_child = std.process.spawn(io, .{
        .argv = &[_][]const u8{ "git", "commit", "-m", msg_lines.items },
        .cwd = .inherit,
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    }) catch |err| {
        std.debug.print("commit: failed: {s}\n", .{@errorName(err)});
        return;
    };
    const term = try commit_child.wait(io);
    std.debug.print("commit done (exit {any})\n", .{term});
}

/// `/sessions`: list saved session files in `~/.abi/sessions/`.
pub fn listSessions(allocator: std.mem.Allocator, io: std.Io) !void {
    const sessions_dir = repl_session.sessionsDir(allocator) catch |err| {
        std.debug.print("sessions: cannot resolve sessions dir: {s}\n", .{@errorName(err)});
        return;
    };
    defer allocator.free(sessions_dir);

    var dir = std.Io.Dir.cwd().openDir(io, sessions_dir, .{}) catch |err| {
        std.debug.print("sessions: no sessions directory ({s})\n", .{@errorName(err)});
        return;
    };
    defer dir.close(io);

    std.debug.print("Saved sessions:\n", .{});
    var iter = dir.iterate();
    var count: usize = 0;
    while (try iter.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;
        const name = entry.name[0 .. entry.name.len - 5];
        std.debug.print("  {s}\n", .{name});
        count += 1;
    }
    if (count == 0) {
        std.debug.print("  (none)\n", .{});
    }
}

test {
    std.testing.refAllDecls(@This());
}
