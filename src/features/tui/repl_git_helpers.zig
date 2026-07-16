//! Pure, unit-testable git/diff/commit helper functions extracted from
//! `repl_commands.zig`. Zero dependency on `repl_types` — these are pure
//! string/argv builders and can be tested without constructing a live store.

const std = @import("std");
const builtin = @import("builtin");
const utils = @import("../../foundation/utils.zig");

// ── colorizeDiff ──────────────────────────────────────────────────────────

/// Render a git diff with ANSI color codes: green for additions, red for
/// deletions, cyan for hunk headers, bold for file headers (`+++`/`---`/diff).
/// Returns an owned slice; caller frees. Pure: no IO, no terminal.
pub fn colorizeDiff(allocator: std.mem.Allocator, diff: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(allocator);
    var iter = std.mem.splitScalar(u8, diff, '\n');
    while (iter.next()) |line| {
        if (line.len > 0) {
            switch (line[0]) {
                '+' => {
                    if (!std.mem.startsWith(u8, line, "+++")) {
                        try out.appendSlice(allocator, "\x1b[32m");
                        try out.appendSlice(allocator, line);
                        try out.appendSlice(allocator, "\x1b[0m\n");
                    } else {
                        try out.appendSlice(allocator, "\x1b[1m");
                        try out.appendSlice(allocator, line);
                        try out.appendSlice(allocator, "\x1b[0m\n");
                    }
                },
                '-' => {
                    if (!std.mem.startsWith(u8, line, "---")) {
                        try out.appendSlice(allocator, "\x1b[31m");
                        try out.appendSlice(allocator, line);
                        try out.appendSlice(allocator, "\x1b[0m\n");
                    } else {
                        try out.appendSlice(allocator, "\x1b[1m");
                        try out.appendSlice(allocator, line);
                        try out.appendSlice(allocator, "\x1b[0m\n");
                    }
                },
                '@' => {
                    try out.appendSlice(allocator, "\x1b[36m");
                    try out.appendSlice(allocator, line);
                    try out.appendSlice(allocator, "\x1b[0m\n");
                },
                'd' => if (std.mem.startsWith(u8, line, "diff")) {
                    try out.appendSlice(allocator, "\x1b[1m");
                    try out.appendSlice(allocator, line);
                    try out.appendSlice(allocator, "\x1b[0m\n");
                } else {
                    try out.appendSlice(allocator, line);
                    try out.appendSlice(allocator, "\n");
                },
                else => {
                    try out.appendSlice(allocator, line);
                    try out.appendSlice(allocator, "\n");
                },
            }
        } else {
            try out.appendSlice(allocator, "\n");
        }
    }
    return out.toOwnedSlice(allocator);
}

// ── diffArgv / commitArgv ────────────────────────────────────────────────

/// Return the git argv for /diff. Pure; no spawn.
pub fn diffArgv(want_stat: bool) []const []const u8 {
    return if (want_stat)
        &[_][]const u8{ "git", "diff", "--stat" }
    else
        &[_][]const u8{ "git", "diff", "--color=never" };
}

/// Parse the /diff argument token and return whether --stat was requested. Pure.
pub fn diffWantsStat(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "--stat");
}

/// The constant git-add argv used by /commit. Pure.
pub fn commitAddArgv() []const []const u8 {
    return &[_][]const u8{ "git", "add", "-A" };
}

/// Build the git-commit argv for a given (owned) message. Pure; caller owns msg lifetime.
pub fn commitArgvFor(msg: []const u8) [4][]const u8 {
    return .{ "git", "commit", "-m", msg };
}

// ── accumulateCommitMessage ──────────────────────────────────────────────

pub const CommitMessageOutcome = union(enum) {
    cancelled,
    message: []u8, // owned; caller frees
};

/// Accumulate a multi-line commit message from pre-read, pre-trimmed lines.
/// First empty line cancels; a subsequent empty line after non-empty content
/// submits. Pure: no stdin, no spawn. Caller frees .message.
pub fn accumulateCommitMessage(allocator: std.mem.Allocator, lines: []const []const u8) !CommitMessageOutcome {
    var msg = std.ArrayListUnmanaged(u8).empty;
    defer msg.deinit(allocator);

    for (lines) |line| {
        if (line.len == 0) {
            if (msg.items.len == 0) return .cancelled;
            break;
        }
        if (msg.items.len > 0) try msg.append(allocator, '\n');
        try msg.appendSlice(allocator, line);
    }

    if (msg.items.len == 0) return .cancelled;
    return .{ .message = try msg.toOwnedSlice(allocator) };
}

// ── homeEnvVarName / syncClisLauncherPath ────────────────────────────────

/// Return the OS-appropriate home environment variable name. Pure.
pub fn homeEnvVarName(os_tag: std.Target.Os.Tag) []const u8 {
    return if (os_tag == .windows) "USERPROFILE" else "HOME";
}

/// Join `home` with the sync-clis launcher path. Returns null when home is null.
/// Pure: no FS access. Caller frees non-null result.
pub fn syncClisLauncherPath(allocator: std.mem.Allocator, home: ?[]const u8) !?[]u8 {
    if (home == null) return null;
    const joined = try utils.pathJoin(home.?, ".grok/skills/sync-clis/launch.sh", allocator);
    return @constCast(joined);
}

// ── Tests ─────────────────────────────────────────────────────────────────

test "colorizeDiff marks added lines green and removed lines red" {
    const input = "+add\n-del\n";
    const result = try colorizeDiff(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "\x1b[32m+add\x1b[0m") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\x1b[31m-del\x1b[0m") != null);
}

test "colorizeDiff leaves +++ and --- file headers bold not colored" {
    const input = "+++ b/file\n--- a/file\n";
    const result = try colorizeDiff(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "\x1b[32m") == null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\x1b[31m") == null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\x1b[1m") != null);
}

test "colorizeDiff colors hunk headers cyan and diff headers bold" {
    const input = "@@ -1,2 +1,3 @@\ndiff --git a/f b/f\n";
    const result = try colorizeDiff(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "\x1b[36m@@ -1,2 +1,3 @@") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\x1b[1mdiff --git") != null);
}

test "colorizeDiff passes context lines through unchanged" {
    const input = " context line\n";
    const result = try colorizeDiff(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "\x1b[") == null);
    try std.testing.expect(std.mem.indexOf(u8, result, "context line") != null);
}

test "colorizeDiff preserves empty lines as bare newlines" {
    const input = "\n\n";
    const result = try colorizeDiff(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "\x1b[") == null);
}

test "diffArgv selects --stat argv when requested" {
    const argv = diffArgv(true);
    try std.testing.expectEqualStrings("git", argv[0]);
    try std.testing.expectEqualStrings("diff", argv[1]);
    try std.testing.expectEqualStrings("--stat", argv[2]);
}

test "diffArgv selects colorless argv by default" {
    const argv = diffArgv(false);
    try std.testing.expectEqualStrings("git", argv[0]);
    try std.testing.expectEqualStrings("diff", argv[1]);
    try std.testing.expectEqualStrings("--color=never", argv[2]);
}

test "diffWantsStat is true only for the exact --stat token" {
    try std.testing.expect(diffWantsStat("--stat"));
    try std.testing.expect(!diffWantsStat(""));
    try std.testing.expect(!diffWantsStat("--stats"));
    try std.testing.expect(!diffWantsStat("stat"));
}

test "commitAddArgv stages all changes" {
    const argv = commitAddArgv();
    try std.testing.expectEqualStrings("git", argv[0]);
    try std.testing.expectEqualStrings("add", argv[1]);
    try std.testing.expectEqualStrings("-A", argv[2]);
}

test "commitArgvFor embeds the message as the -m argument" {
    const argv = commitArgvFor("fix: x");
    try std.testing.expectEqualStrings("git", argv[0]);
    try std.testing.expectEqualStrings("commit", argv[1]);
    try std.testing.expectEqualStrings("-m", argv[2]);
    try std.testing.expectEqualStrings("fix: x", argv[3]);
}

test "accumulateCommitMessage cancels on leading empty line" {
    const result = try accumulateCommitMessage(std.testing.allocator, &.{""});
    try std.testing.expect(result == .cancelled);
}

test "accumulateCommitMessage cancels on empty input" {
    const result = try accumulateCommitMessage(std.testing.allocator, &.{});
    try std.testing.expect(result == .cancelled);
}

test "accumulateCommitMessage submits a single non-empty line" {
    const result = try accumulateCommitMessage(std.testing.allocator, &.{"fix: thing"});
    defer std.testing.allocator.free(result.message);
    try std.testing.expectEqualStrings("fix: thing", result.message);
}

test "accumulateCommitMessage joins multiple lines with newlines" {
    const result = try accumulateCommitMessage(std.testing.allocator, &.{ "a", "b", "c" });
    defer std.testing.allocator.free(result.message);
    try std.testing.expectEqualStrings("a\nb\nc", result.message);
}

test "accumulateCommitMessage submits on the first empty line after content" {
    const result = try accumulateCommitMessage(std.testing.allocator, &.{ "a", "b", "" });
    defer std.testing.allocator.free(result.message);
    try std.testing.expectEqualStrings("a\nb", result.message);
}

test "accumulateCommitMessage ignores trailing empties beyond the first submit boundary" {
    const result = try accumulateCommitMessage(std.testing.allocator, &.{ "a", "", "b" });
    defer std.testing.allocator.free(result.message);
    try std.testing.expectEqualStrings("a", result.message);
}

test "homeEnvVarName selects HOME on posix and USERPROFILE on windows" {
    try std.testing.expectEqualStrings("HOME", homeEnvVarName(.macos));
    try std.testing.expectEqualStrings("HOME", homeEnvVarName(.linux));
    try std.testing.expectEqualStrings("USERPROFILE", homeEnvVarName(.windows));
}

test "syncClisLauncherPath returns null when home is null" {
    const result = try syncClisLauncherPath(std.testing.allocator, null);
    try std.testing.expect(result == null);
}

test "syncClisLauncherPath joins home with the grok skill launcher path" {
    const result = try syncClisLauncherPath(std.testing.allocator, "/Users/x");
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);
    try std.testing.expect(std.mem.endsWith(u8, result.?, "sync-clis/launch.sh"));
}

test {
    std.testing.refAllDecls(@This());
}
