const std = @import("std");
const util = @import("util.zig");

fn scanForbidden(
    allocator: std.mem.Allocator,
    pattern: []const u8,
    label: []const u8,
    errors: *usize,
) !void {
    const cmd = try std.fmt.allocPrint(
        allocator,
        "rg -n --glob '*.zig' '{s}' src build tools",
        .{pattern},
    );
    defer allocator.free(cmd);

    const result = try util.captureCommand(allocator, cmd);
    defer allocator.free(result.output);

    if (result.exit_code == 0) {
        std.debug.print("ERROR: Found forbidden Zig 0.16 pattern: {s}\n", .{label});
        std.debug.print("{s}", .{result.output});
        errors.* += 1;
        return;
    }

    if (result.exit_code != 1) {
        std.debug.print("ERROR: Pattern scan failed for '{s}' (exit={d})\n", .{ pattern, result.exit_code });
        if (result.output.len > 0) std.debug.print("{s}", .{result.output});
        errors.* += 1;
    }
}

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var errors: usize = 0;

    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*std\\.fs\\.cwd\\(",
        "legacy cwd API usage; use std.Io.Dir.cwd()",
        &errors,
    );
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*std\\.io\\.fixedBufferStream\\(",
        "fixedBufferStream legacy API removed in Zig 0.16",
        &errors,
    );
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*std\\.time\\.nanoTimestamp\\(",
        "nanoTimestamp legacy API removed in Zig 0.16",
        &errors,
    );
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*std\\.time\\.sleep\\(",
        "legacy sleep API forbidden; use services/shared/time wrapper",
        &errors,
    );
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*std\\.process\\.getEnvVar\\(",
        "legacy process env API removed in Zig 0.16",
        &errors,
    );
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*@typeInfo\\([^)]*\\)\\.Fn",
        "@typeInfo(.Fn) -> @typeInfo(.@\"fn\")",
        &errors,
    );
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*std\\.ArrayList\\([^)]*\\)\\.init\\(",
        "legacy ArrayList init usage; prefer ArrayListUnmanaged patterns",
        &errors,
    );

    try scanForbidden(
        allocator,
        "std\\.(debug\\.print|log\\.[a-z]+)\\([^)]*@tagName\\(",
        "@tagName() used in print/log formatting context; use {t} instead",
        &errors,
    );
    try scanForbidden(
        allocator,
        "std\\.(debug\\.print|log\\.[a-z]+)\\([^)]*@errorName\\(",
        "@errorName() used in print/log formatting context; use {t} instead",
        &errors,
    );

    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*comptime[[:space:]]*\\{[[:space:]]*_[[:space:]]*=[[:space:]]*@import\\(",
        "legacy comptime-based test discovery detected; use test { _ = @import(...); }",
        &errors,
    );

    if (errors > 0) {
        std.debug.print("FAILED: Zig 0.16 pattern check found {d} issue(s)\n", .{errors});
        std.process.exit(1);
    }

    std.debug.print("OK: Zig 0.16 pattern checks passed\n", .{});
}
