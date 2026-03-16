const std = @import("std");
const util = @import("util");

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var violations: usize = 0;

    const checks = [_]struct {
        command: []const u8,
        description: []const u8,
        violation_when_matched: bool,
    }{
        .{
            .command = "rg -n 'validateCatalog\\(&catalog_items\\);' tools/cli/terminal/launcher/launcher_catalog.zig",
            .description = "launcher catalog must validate descriptor definitions at compile-time",
            .violation_when_matched = false,
        },
        .{
            .command = "rg -n 'const command_modules = \\.\\{' tools/cli/commands/mod.zig",
            .description = "commands/mod.zig must source modules from generated registry",
            .violation_when_matched = true,
        },
        .{
            .command = "rg -n '@import\\(\"../../../terminal/dsl/mod.zig\"\\)' tools/cli/commands/core/ui/model.zig tools/cli/commands/core/ui/bench.zig tools/cli/commands/core/ui/db.zig tools/cli/commands/core/ui/network.zig tools/cli/commands/core/ui/streaming.zig",
            .description = "repetitive UI dashboards must use shared DSL helper",
            .violation_when_matched = false,
        },
        .{
            .command = "rg -n 'fn wrap[A-Z]' tools/cli/commands/core/profile.zig tools/cli/commands/infra/network.zig tools/cli/commands/dev/task.zig tools/cli/commands/dev/lsp.zig tools/cli/commands/ai/model.zig tools/cli/commands/ai/train/mod.zig tools/cli/commands/ai/ralph/mod.zig",
            .description = "migrated command families must not use legacy wrapX shim functions",
            .violation_when_matched = true,
        },
        .{
            .command = "rg -n 'const [a-z_]+_subcommands = \\[_\\]\\[\\]const u8' tools/cli/commands/core/profile.zig tools/cli/commands/infra/network.zig tools/cli/commands/dev/task.zig tools/cli/commands/dev/lsp.zig tools/cli/commands/ai/model.zig tools/cli/commands/ai/train/mod.zig tools/cli/commands/ai/ralph/mod.zig",
            .description = "migrated command families must derive suggestions from command metadata",
            .violation_when_matched = true,
        },
        .{
            .command = "rg -n '\\.aliases\\s*=\\s*&\\.\\{\\s*\"' tools/cli/commands/core/ui/mod.zig",
            .description = "ui command must not expose legacy top-level aliases",
            .violation_when_matched = true,
        },
    };

    for (checks) |check| {
        const result = try util.captureCommand(allocator, io, check.command);
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
