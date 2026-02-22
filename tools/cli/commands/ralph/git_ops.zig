//! Ralph git operations for autonomous iteration commits.

const std = @import("std");

fn runGit(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: []const u8,
    argv: []const []const u8,
) !std.process.RunResult {
    return std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .stdout_limit = .limited(8 * 1024 * 1024),
        .stderr_limit = .limited(8 * 1024 * 1024),
    });
}

pub fn isGitRepo(allocator: std.mem.Allocator, io: std.Io, cwd: []const u8) bool {
    const result = runGit(allocator, io, cwd, &.{ "git", "rev-parse", "--is-inside-work-tree" }) catch return false;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    return result.term == .exited and result.term.exited == 0;
}

pub fn hasChanges(allocator: std.mem.Allocator, io: std.Io, cwd: []const u8) bool {
    const result = runGit(allocator, io, cwd, &.{ "git", "status", "--porcelain" }) catch return false;
    defer {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
    if (result.term != .exited or result.term.exited != 0) return false;
    return std.mem.trim(u8, result.stdout, " \t\r\n").len > 0;
}

pub fn ensureRunBranch(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: []const u8,
    branch_name: []const u8,
) void {
    var create = runGit(allocator, io, cwd, &.{ "git", "checkout", "-b", branch_name }) catch return;
    defer {
        allocator.free(create.stdout);
        allocator.free(create.stderr);
    }
    if (create.term == .exited and create.term.exited == 0) return;

    var checkout = runGit(allocator, io, cwd, &.{ "git", "checkout", branch_name }) catch return;
    defer {
        allocator.free(checkout.stdout);
        allocator.free(checkout.stderr);
    }
}

pub fn commitAllIfChanged(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: []const u8,
    run_id: []const u8,
    iteration: usize,
    gate_passed: bool,
    provider_label: []const u8,
) bool {
    var add = runGit(allocator, io, cwd, &.{ "git", "add", "-A" }) catch return false;
    defer {
        allocator.free(add.stdout);
        allocator.free(add.stderr);
    }
    if (add.term != .exited or add.term.exited != 0) return false;

    var diff = runGit(allocator, io, cwd, &.{ "git", "diff", "--cached", "--quiet" }) catch return false;
    defer {
        allocator.free(diff.stdout);
        allocator.free(diff.stderr);
    }
    if (diff.term == .exited and diff.term.exited == 0) {
        return false;
    }

    const msg = std.fmt.allocPrint(
        allocator,
        "ralph: iteration {d} [{s}] gate={s} provider={s}",
        .{ iteration, run_id, if (gate_passed) "pass" else "fail", provider_label },
    ) catch return false;
    defer allocator.free(msg);

    var commit = runGit(allocator, io, cwd, &.{ "git", "commit", "-m", msg, "--no-verify" }) catch return false;
    defer {
        allocator.free(commit.stdout);
        allocator.free(commit.stderr);
    }
    return commit.term == .exited and commit.term.exited == 0;
}
