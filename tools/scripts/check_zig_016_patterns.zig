const std = @import("std");
const util = @import("util.zig");

fn scanForbidden(
    allocator: std.mem.Allocator,
    pattern: []const u8,
    label: []const u8,
    errors: *usize,
) !void {
    return scanForbiddenFlags(allocator, pattern, label, errors, "");
}

fn scanForbiddenMultiline(
    allocator: std.mem.Allocator,
    pattern: []const u8,
    label: []const u8,
    errors: *usize,
) !void {
    return scanForbiddenFlags(allocator, pattern, label, errors, "-U ");
}

fn scanForbiddenFlags(
    allocator: std.mem.Allocator,
    pattern: []const u8,
    label: []const u8,
    errors: *usize,
    extra_flags: []const u8,
) !void {
    const cmd = try std.fmt.allocPrint(
        allocator,
        "rg -n {s}--glob '*.zig' --glob '!check_zig_016_patterns.zig' '{s}' src build tools",
        .{ extra_flags, pattern },
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

    // S5: Pre-check that rg (ripgrep) is available
    const rg_check = try util.captureCommand(allocator, "rg --version");
    defer allocator.free(rg_check.output);
    if (rg_check.exit_code != 0) {
        std.debug.print("ERROR: 'rg' (ripgrep) is required but not found in PATH.\n", .{});
        std.debug.print("Install with: brew install ripgrep (macOS) or cargo install ripgrep\n", .{});
        std.process.exit(1);
    }

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
        "^[[:space:]]*[^/].*std\\.io\\.AnyWriter",
        "legacy AnyWriter path from std.io; use std.Io writer abstractions in Zig 0.16",
        &errors,
    );
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*getStd(Out|Err)\\(",
        "legacy std.io.getStdOut/getStdErr API usage removed in Zig 0.16",
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

    // S6: Additional deprecated patterns
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*std\\.json\\.stringifyAlloc\\(",
        "legacy std.json.stringifyAlloc; use std.json.Stringify.valueAlloc in Zig 0.16",
        &errors,
    );
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*std\\.process\\.argsAlloc\\(",
        "legacy std.process.argsAlloc; use init.minimal.args.toSlice(arena) in Zig 0.16",
        &errors,
    );
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*std\\.crypto\\.random",
        "legacy std.crypto.random; use std.c.arc4random_buf in Zig 0.16",
        &errors,
    );
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*std\\.posix\\.getenv\\(",
        "legacy std.posix.getenv; use std.c.getenv in Zig 0.16",
        &errors,
    );

    // #3: defer + free immediately before bare `return varname;` (use-after-free)
    // Narrow pattern: only flags direct identifier returns, not function-call returns.
    try scanForbiddenMultiline(
        allocator,
        "defer[[:space:]]+[a-z_.]+\\.free\\([^)]+\\);[[:space:]]*\\n[[:space:]]*return [a-z_]+;",
        "defer+free before bare return (use-after-free); use errdefer instead",
        &errors,
    );

    // #6: @field(build_options) — not checked by regex. Zig enforces comptime at compile
    // time, and all codebase uses are correctly marked `comptime`. False positives would
    // fire on every valid isCompileTimeEnabled() call.

    // #9: HashMap.init(allocator)
    try scanForbidden(
        allocator,
        "HashMap\\([^)]*\\)\\.init\\(",
        "legacy HashMap.init(allocator); use .empty + per-call allocator in Zig 0.16",
        &errors,
    );

    // #15: std.os.chdir()
    try scanForbidden(
        allocator,
        "std\\.os\\.chdir\\(",
        "legacy std.os.chdir; use std.Io.Threaded.chdir() in Zig 0.16",
        &errors,
    );

    // #16: std.cstr. removed; allocPrintZ renamed
    try scanForbidden(
        allocator,
        "std\\.cstr\\.|allocPrintZ\\(",
        "std.cstr removed in Zig 0.16; use std.fmt.allocPrintSentinel(alloc, fmt, args, 0)",
        &errors,
    );

    // #17: .writer(io) missing required buffer argument
    try scanForbidden(
        allocator,
        "\\.writer\\(io\\)[^,]",
        "file.writer(io) missing buffer arg; use file.writer(io, &buf) + flush() in Zig 0.16",
        &errors,
    );

    // #20: dupe(u8, @tagName(...)) directly — @tagName returns [*:0]const u8, not []const u8
    // Narrow pattern: only flags the exact wrong form `.dupe(u8, @tagName(`, not the correct sliceTo variant.
    try scanForbidden(
        allocator,
        "\\.dupe\\(u8, @tagName\\(",
        "@tagName returns [*:0]const u8; use std.mem.sliceTo(@tagName(x), 0) before dupe(u8, ...)",
        &errors,
    );

    // #11: pub fn main() !void — missing Init parameter (old 0.14/0.15 signature)
    try scanForbiddenFlags(
        allocator,
        "pub fn main\\(\\)\\s*!?\\s*void",
        "old main() signature; use pub fn main(init: std.process.Init) !void in Zig 0.16",
        &errors,
        "--glob '!tools/gendocs/wasm/exports.zig' ",
    );

    // #7/#19: Removed facade aliases from v0.4.0 (abi.ai_core, abi.inference, etc.)
    try scanForbidden(
        allocator,
        "^[[:space:]]*[^/].*abi\\.(ai_core|inference|training|reasoning)\\b",
        "removed v0.4.0 facade alias; use abi.ai.core, abi.ai.llm, abi.ai.training, abi.ai.orchestration",
        &errors,
    );

    // #4 extended: @tagName/@errorName with {s} in any writer.print (not just std.debug.print)
    try scanForbidden(
        allocator,
        "\\.print\\([^)]*@(tagName|errorName)\\([^)]*\\)[^)]*\\{s\\}",
        "@tagName/@errorName with {s} in print; use {t} format specifier instead",
        &errors,
    );

    if (errors > 0) {
        std.debug.print("FAILED: Zig 0.16 pattern check found {d} issue(s)\n", .{errors});
        std.process.exit(1);
    }

    std.debug.print("OK: Zig 0.16 pattern checks passed\n", .{});
}
