const std = @import("std");
const util = @import("util.zig");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    const cmd = "rg -n --glob '*.zig' '@import\\(\"abi\"\\)' src/features";
    const result = try util.captureCommand(allocator, cmd);
    defer allocator.free(result.output);

    if (result.exit_code != 0 and result.exit_code != 1) {
        std.debug.print("ERROR: import rule scan failed (exit={d})\n", .{result.exit_code});
        if (result.output.len > 0) std.debug.print("{s}", .{result.output});
        std.process.exit(1);
    }

    var violations: usize = 0;
    var lines = std.mem.splitScalar(u8, result.output, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;

        const first_colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const second_rel = std.mem.indexOfScalar(u8, line[first_colon + 1 ..], ':') orelse continue;
        const second_colon = first_colon + 1 + second_rel;

        const file = line[0..first_colon];
        const lineno = line[first_colon + 1 .. second_colon];
        const content = line[second_colon + 1 ..];
        const trimmed = std.mem.trim(u8, content, " \t");

        if (std.mem.startsWith(u8, trimmed, "//")) continue;

        std.debug.print("VIOLATION: {s}:{s}: {s}\n", .{ file, lineno, trimmed });
        violations += 1;
    }

    if (violations > 0) {
        std.debug.print("\nERROR: Found {d} @import(\"abi\") violation(s) in feature modules.\n", .{violations});
        std.debug.print("Feature modules must use relative imports to avoid circular dependencies.\n", .{});
        std.process.exit(1);
    }

    std.debug.print("OK: No @import(\"abi\") violations in feature modules.\n", .{});
}
