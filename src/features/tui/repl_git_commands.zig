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
const test_helpers = @import("../../foundation/test_helpers.zig");

/// `/sync-clis`: execute the central sync-clis launcher. Prefers the in-repo
/// canonical launcher, then the synced `.claude` copy, then the operator's
/// `~/.grok` skill dir (via the OS-appropriate home env var). Never executes
/// a missing script.
pub fn runSyncClis(allocator: std.mem.Allocator, io: std.Io) !void {
    const candidates = [_][]const u8{
        ".agents/skills/sync-clis/launch.sh",
        ".claude/skills/sync-clis/launch.sh",
    };
    var launch_owned: ?[]const u8 = null;
    defer if (launch_owned) |p| allocator.free(p);

    var launch_path: ?[]const u8 = null;
    for (candidates) |rel| {
        std.Io.Dir.cwd().access(io, rel, .{}) catch continue;
        launch_path = rel;
        break;
    }
    if (launch_path == null) {
        const home_var = cmds.homeEnvVarName(builtin.target.os.tag);
        const maybe_grok = try cmds.syncClisLauncherPath(allocator, env.get(home_var));
        if (maybe_grok) |grok| {
            std.Io.Dir.cwd().access(io, grok, .{}) catch {
                allocator.free(grok);
                std.debug.print("sync-clis: launcher not found (tried .agents/, .claude/, ~/.grok/)\n", .{});
                return;
            };
            launch_owned = grok;
            launch_path = grok;
        } else {
            std.debug.print("sync-clis: launcher not found (tried .agents/, .claude/; HOME unset)\n", .{});
            return;
        }
    }

    std.debug.print("sync-clis: executing {s}...\n", .{launch_path.?});
    var child = try std.process.spawn(io, .{
        .argv = &[_][]const u8{launch_path.?},
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
    // `content` is copied into the packed buffer below; without this free the
    // transient read buffer leaked on every successful `/open`.
    defer allocator.free(content);

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

/// `/diff [--stat]`: run `git diff` and print the output. `arg` is the
/// argument text after the slash command (e.g. `--stat` from `/diff --stat`).
/// When `--stat` is passed, shows a summary of changed files. Output is
/// colorized with ANSI codes for added (+) lines in green and removed (-)
/// lines in red.
pub fn runDiff(allocator: std.mem.Allocator, arg: []const u8, io: std.Io) !void {
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

// ── Tests ─────────────────────────────────────────────────────────────────
//
// These cover the file-context side of the module (`runOpen`), which needs no
// subprocess. The pure argv/format helpers (`diffArgv`, `diffWantsStat`,
// `commitArgvFor`, `formatOpenStatus`, `homeEnvVarName`, `syncClisLauncherPath`)
// are already unit-tested in `repl_commands.zig` and are intentionally not
// duplicated here. The spawn paths (`runDiff`, `runCommit`, `runSyncClis`)
// stay untested by design: exercising them would fork git or the sync-clis
// launcher (which exists in this repo's cwd), and the launcher-not-found /
// HOME-unset branch would require mutating global process env.

test "runOpen loads a real file and packs path + content into file_context_buf" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const path = "zig-out/repl-open-roundtrip.txt";
    const content = "hello from /open\nline two\n";
    try std.Io.Dir.createDirPath(.cwd(), io, "zig-out");
    defer test_helpers.deleteTestFileIfExists(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = content });

    var state = repl_types.ReplState.init(.{});
    defer state.clearFileContext(allocator);

    try runOpen(allocator, &state, path, io);

    try std.testing.expect(state.file_context_buf != null);
    try std.testing.expectEqualStrings(path, state.open_path);
    try std.testing.expectEqualStrings(content, state.open_content);

    // The packed layout is [usize path_len][path bytes][content bytes], and
    // both public slices must alias the single owned backing buffer (never the
    // caller's transient `path` argument, which dangles after the REPL line).
    const buf = state.file_context_buf.?;
    const header_len = @sizeOf(usize);
    try std.testing.expectEqual(path.len, std.mem.bytesAsSlice(usize, buf[0..header_len])[0]);
    try std.testing.expectEqual(header_len + path.len + content.len, buf.len);
    try std.testing.expectEqual(@intFromPtr(buf.ptr) + header_len, @intFromPtr(state.open_path.ptr));
    try std.testing.expectEqual(@intFromPtr(buf.ptr) + header_len + path.len, @intFromPtr(state.open_content.ptr));
}

test "runOpen with empty path and nothing loaded takes the usage branch untouched" {
    const allocator = std.testing.allocator;
    var state = repl_types.ReplState.init(.{});

    try runOpen(allocator, &state, "", std.testing.io);

    try std.testing.expectEqual(@as(usize, 0), state.open_path.len);
    try std.testing.expectEqual(@as(usize, 0), state.open_content.len);
    try std.testing.expect(state.file_context_buf == null);
}

test "runOpen with empty path reports status and keeps the loaded context" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const path = "zig-out/repl-open-status.txt";
    try std.Io.Dir.createDirPath(.cwd(), io, "zig-out");
    defer test_helpers.deleteTestFileIfExists(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "status body" });

    var state = repl_types.ReplState.init(.{});
    defer state.clearFileContext(allocator);
    try runOpen(allocator, &state, path, io);
    const buf_before = state.file_context_buf.?;

    // Empty path with a loaded file must print status without reloading,
    // freeing, or replacing the existing context.
    try runOpen(allocator, &state, "", io);

    try std.testing.expectEqual(@intFromPtr(buf_before.ptr), @intFromPtr(state.file_context_buf.?.ptr));
    try std.testing.expectEqualStrings(path, state.open_path);
    try std.testing.expectEqualStrings("status body", state.open_content);
}

test "runOpen on an unreadable path clears prior context and returns cleanly" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const path = "zig-out/repl-open-cleared.txt";
    try std.Io.Dir.createDirPath(.cwd(), io, "zig-out");
    defer test_helpers.deleteTestFileIfExists(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "soon stale" });

    var state = repl_types.ReplState.init(.{});
    defer state.clearFileContext(allocator);
    try runOpen(allocator, &state, path, io);
    try std.testing.expect(state.file_context_buf != null);

    // The failed read happens after clearFileContext, so the stale context is
    // gone and the error is reported without raising.
    try runOpen(allocator, &state, "zig-out/definitely-missing-repl-open.txt", io);

    try std.testing.expectEqual(@as(usize, 0), state.open_path.len);
    try std.testing.expectEqual(@as(usize, 0), state.open_content.len);
    try std.testing.expect(state.file_context_buf == null);
}

test "runOpen replaces a previously opened file without leaking the old buffer" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const first = "zig-out/repl-open-first.txt";
    const second = "zig-out/repl-open-second.txt";
    try std.Io.Dir.createDirPath(.cwd(), io, "zig-out");
    defer test_helpers.deleteTestFileIfExists(first);
    defer test_helpers.deleteTestFileIfExists(second);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = first, .data = "first contents" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = second, .data = "second contents" });

    var state = repl_types.ReplState.init(.{});
    defer state.clearFileContext(allocator);

    try runOpen(allocator, &state, first, io);
    try std.testing.expectEqualStrings("first contents", state.open_content);

    // std.testing.allocator fails the test if the first backing buffer leaks.
    try runOpen(allocator, &state, second, io);
    try std.testing.expectEqualStrings(second, state.open_path);
    try std.testing.expectEqualStrings("second contents", state.open_content);
}

test "runOpen truncates content to the OPEN_FILE_BUDGET_BYTES budget" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const path = "zig-out/repl-open-budget.txt";
    try std.Io.Dir.createDirPath(.cwd(), io, "zig-out");
    defer test_helpers.deleteTestFileIfExists(path);

    const oversized = try allocator.alloc(u8, repl_types.OPEN_FILE_BUDGET_BYTES + 128);
    defer allocator.free(oversized);
    @memset(oversized, 'x');
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = oversized });

    var state = repl_types.ReplState.init(.{});
    defer state.clearFileContext(allocator);
    try runOpen(allocator, &state, path, io);

    try std.testing.expectEqual(repl_types.OPEN_FILE_BUDGET_BYTES, state.open_content.len);
}

test {
    std.testing.refAllDecls(@This());
}
