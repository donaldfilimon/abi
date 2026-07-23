//! `abi agent os <dry-run|execute --confirm> <cmd> [args...]`
//!
//! Runs an OS-control command request through the os_control policy gate.
//! `execute` requires an explicit `--confirm`; without it (or for `dry-run`)
//! the command is only audited, never run. Arbitrary user argv is classified
//! `.unknown` intent — the real gate is the allow-list plus workspace containment.

const std = @import("std");
const abi = @import("abi");
const usage_mod = @import("../usage.zig");

const OS_ALLOWED_COMMANDS = &.{ "true", "pwd", "ls", "whoami", "date" };

/// Returns the process exit code.
pub fn handleAgentOs(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 4) return usage_mod.usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");
    const os_cmd = args[3];
    const execute = std.mem.eql(u8, os_cmd, "execute");
    const dry_run = std.mem.eql(u8, os_cmd, "dry-run");
    if (!execute and !dry_run) return usage_mod.usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");
    if (execute and (args.len < 6 or !std.mem.eql(u8, args[4], "--confirm"))) {
        return usage_mod.usageError("usage: abi agent os execute --confirm <cmd> [args...]");
    }
    const start: usize = if (execute) 5 else 4;
    if (start >= args.len) return usage_mod.usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");

    const cwd_z = try std.process.currentPathAlloc(io, allocator);
    defer allocator.free(cwd_z);
    const policy = abi.features.os_control.Policy{
        .workspace_root = cwd_z,
        .dry_run_only = dry_run,
        .allow_execution = execute,
        .allowed_commands = OS_ALLOWED_COMMANDS,
        .require_confirmation = true,
    };
    const request = abi.features.os_control.CommandRequest{
        .intent = .unknown,
        .argv = args[start..],
        .cwd = cwd_z,
        .confirm_execution = execute,
    };

    if (dry_run) {
        const rendered = abi.features.os_control.renderDryRun(allocator, io, request, policy) catch |err| switch (err) {
            error.CommandDenied => {
                std.debug.print("error: command denied by os-control policy\n", .{});
                return 1;
            },
            else => return err,
        };
        defer allocator.free(rendered);
        std.debug.print("{s}\n", .{rendered});
        return 0;
    }

    const result = abi.features.os_control.executeConfirmed(allocator, io, request, policy) catch |err| switch (err) {
        error.CommandDenied => {
            std.debug.print("error: command denied by os-control policy\n", .{});
            return 1;
        },
        else => return err,
    };
    std.debug.print("{s}\n", .{result.message});
    return result.exit_code orelse 0;
}

test "agent os dry-run denies commands outside the policy allow-list" {
    const allocator = std.testing.allocator;
    const t = std.testing.io;
    try std.testing.expectEqual(@as(u8, 1), try handleAgentOs(t, allocator, &.{ "abi", "agent", "os", "dry-run", "rm" }));
}

test "agent os dry-run allow-listed command succeeds without executing" {
    // Skip when os_control is stubbed out (-Dfeat-os-control=false) or the
    // target has no trusted executable table (non-macOS/Linux): both make
    // every request deny by design.
    if (abi.features.os_control.trustedCommandSpec("pwd") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const t = std.testing.io;
    // The handler resolves the ambient cwd itself and uses it as both the
    // policy workspace root and the request cwd, so containment holds in any
    // real directory. `renderDryRun` applies the full policy gate and renders
    // the escaped plan — it never spawns a child process.
    try std.testing.expectEqual(@as(u8, 0), try handleAgentOs(t, allocator, &.{ "abi", "agent", "os", "dry-run", "pwd" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgentOs(t, allocator, &.{ "abi", "agent", "os", "dry-run", "true" }));
}

test {
    std.testing.refAllDecls(@This());
}
