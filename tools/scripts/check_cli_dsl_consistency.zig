const std = @import("std");
const util = @import("util.zig");

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var violations: usize = 0;

    const checks = [_]struct {
        command: []const u8,
        description: []const u8,
        violation_when_matched: bool,
    }{
        .{
            .command = "rg -n 'const catalog_items = \\[_\\]MenuItem\\{' tools/cli/ui/launcher/launcher_catalog.zig",
            .description = "launcher catalog must be descriptor-driven, not manually enumerated",
            .violation_when_matched = true,
        },
        .{
            .command = "rg -n 'const command_modules = \\.\\{' tools/cli/commands/mod.zig",
            .description = "commands/mod.zig must source modules from generated registry",
            .violation_when_matched = true,
        },
        .{
            .command = "rg -n '@import\\(\"../../ui/dsl/mod.zig\"\\)' tools/cli/commands/ui/model.zig tools/cli/commands/ui/bench.zig tools/cli/commands/ui/db.zig tools/cli/commands/ui/network.zig tools/cli/commands/ui/streaming.zig",
            .description = "repetitive UI dashboards must use shared DSL helper",
            .violation_when_matched = false,
        },
        .{
            .command = "rg -n 'fn wrap[A-Z]' tools/cli/commands/profile.zig tools/cli/commands/network.zig tools/cli/commands/task.zig tools/cli/commands/toolchain.zig tools/cli/commands/lsp.zig tools/cli/commands/model.zig tools/cli/commands/train/mod.zig tools/cli/commands/ralph/mod.zig",
            .description = "migrated command families must not use legacy wrapX shim functions",
            .violation_when_matched = true,
        },
        .{
            .command = "rg -n 'const [a-z_]+_subcommands = \\[_\\]\\[\\]const u8' tools/cli/commands/profile.zig tools/cli/commands/network.zig tools/cli/commands/task.zig tools/cli/commands/toolchain.zig tools/cli/commands/lsp.zig tools/cli/commands/model.zig tools/cli/commands/train/mod.zig tools/cli/commands/ralph/mod.zig",
            .description = "migrated command families must derive suggestions from command metadata",
            .violation_when_matched = true,
        },
        .{
            .command = "rg -n '\\.aliases\\s*=\\s*&\\.\\{\\s*\"' tools/cli/commands/ui/mod.zig",
            .description = "ui command must not expose legacy top-level aliases",
            .violation_when_matched = true,
        },
    };

    for (checks) |check| {
        const result = try util.captureCommand(allocator, check.command);
        defer allocator.free(result.output);

        if (result.exit_code != 0 and result.exit_code != 1) {
            std.debug.print("ERROR: DSL check command failed: {s}\n", .{check.command});
            if (result.output.len > 0) std.debug.print("{s}", .{result.output});
            std.process.exit(1);
        }

        const matched = result.exit_code == 0;
        const violation = if (check.violation_when_matched) matched else !matched;
        if (violation) {
            std.debug.print("VIOLATION: {s}\n", .{check.description});
            violations += 1;
        }
    }

    if (violations > 0) {
        std.debug.print("\nERROR: CLI DSL consistency failed with {d} violation(s).\n", .{violations});
        std.process.exit(1);
    }

    std.debug.print("OK: CLI DSL consistency checks passed.\n", .{});
}
