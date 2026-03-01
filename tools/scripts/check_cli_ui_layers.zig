const std = @import("std");
const util = @import("util.zig");

fn isAllowedCommandUiImport(file: []const u8) bool {
    return std.mem.startsWith(u8, file, "tools/cli/commands/ui/") or
        std.mem.eql(u8, file, "tools/cli/commands/train/monitor.zig") or
        std.mem.eql(u8, file, "tools/cli/commands/train/common.zig");
}

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var violations: usize = 0;

    const ui_import_scan_cmd =
        "rg -n --glob '*.zig' '@import\\(\".*ui/(core|panels|launcher|editor)/' tools/cli/commands";
    const ui_import_scan = try util.captureCommand(allocator, ui_import_scan_cmd);
    defer allocator.free(ui_import_scan.output);

    if (ui_import_scan.exit_code != 0 and ui_import_scan.exit_code != 1) {
        std.debug.print("ERROR: command/UI layer scan failed (exit={d})\n", .{ui_import_scan.exit_code});
        if (ui_import_scan.output.len > 0) std.debug.print("{s}", .{ui_import_scan.output});
        std.process.exit(1);
    }

    var lines = std.mem.splitScalar(u8, ui_import_scan.output, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const first_colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const file = line[0..first_colon];
        if (isAllowedCommandUiImport(file)) continue;
        std.debug.print("VIOLATION: {s}\n", .{line});
        violations += 1;
    }

    const ui_backedge_scan_cmd =
        "rg -n --glob '*.zig' '@import\\(\".*commands/' tools/cli/ui";
    const ui_backedge_scan = try util.captureCommand(allocator, ui_backedge_scan_cmd);
    defer allocator.free(ui_backedge_scan.output);

    if (ui_backedge_scan.exit_code != 0 and ui_backedge_scan.exit_code != 1) {
        std.debug.print("ERROR: UI backedge scan failed (exit={d})\n", .{ui_backedge_scan.exit_code});
        if (ui_backedge_scan.output.len > 0) std.debug.print("{s}", .{ui_backedge_scan.output});
        std.process.exit(1);
    }

    if (ui_backedge_scan.output.len > 0) {
        std.debug.print("{s}", .{ui_backedge_scan.output});
        violations += std.mem.count(u8, ui_backedge_scan.output, "\n");
    }

    if (violations > 0) {
        std.debug.print(
            "\nERROR: CLI/UI layering contract failed with {d} violation(s).\n",
            .{violations},
        );
        std.debug.print(
            "Allowed command->UI imports: tools/cli/commands/ui/** and train monitor/common only.\n",
            .{},
        );
        std.debug.print(
            "UI modules must not import tools/cli/commands/**.\n",
            .{},
        );
        std.process.exit(1);
    }

    std.debug.print("OK: CLI/UI layering contract passed.\n", .{});
}
