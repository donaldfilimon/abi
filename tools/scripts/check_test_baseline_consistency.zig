const std = @import("std");
const baseline = @import("baseline.zig");
const util = @import("util.zig");

/// Verify test baseline consistency:
/// - The feature test manifest in build/test_discovery.zig references
///   only files that exist under src/.
/// - The main test root src/services/tests/mod.zig exists.
/// - The WDBX fast test root exists if referenced.
pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var errors: usize = 0;

    // ── Main test root ──────────────────────────────────────────────────
    const main_test_root = "src/services/tests/mod.zig";
    if (!util.fileExists(io, main_test_root)) {
        std.debug.print("ERROR: main test root missing: {s}\n", .{main_test_root});
        errors += 1;
    }

    // ── Feature test manifest ───────────────────────────────────────────
    const manifest_path = "build/test_discovery.zig";
    if (!util.fileExists(io, manifest_path)) {
        std.debug.print("ERROR: feature test manifest missing: {s}\n", .{manifest_path});
        errors += 1;
    } else {
        const manifest_content = util.readFileAlloc(allocator, io, manifest_path, 256 * 1024) catch |err| blk: {
            std.debug.print("ERROR: failed to read {s}: {t}\n", .{ manifest_path, err });
            errors += 1;
            break :blk null;
        };

        if (manifest_content) |content| {
            defer allocator.free(content);

            // Extract .path = "..." entries from the manifest and verify
            // that src/<path> exists on disk.
            var pos: usize = 0;
            var checked: usize = 0;
            const needle = ".path = \"";
            while (std.mem.indexOfPos(u8, content, pos, needle)) |start| {
                const path_start = start + needle.len;
                if (std.mem.indexOfScalarPos(u8, content, path_start, '"')) |path_end| {
                    const rel_path = content[path_start..path_end];
                    const full_path = std.fmt.allocPrint(allocator, "src/{s}", .{rel_path}) catch {
                        errors += 1;
                        pos = path_end + 1;
                        continue;
                    };
                    defer allocator.free(full_path);

                    if (!util.fileExists(io, full_path)) {
                        std.debug.print("ERROR: feature test manifest references missing file: {s}\n", .{full_path});
                        errors += 1;
                    }
                    checked += 1;
                    pos = path_end + 1;
                } else {
                    break;
                }
            } else {
                // No more matches
            }

            if (checked == 0) {
                std.debug.print("WARNING: no .path entries found in test manifest — may be malformed\n", .{});
            }
        }
    }

    // ── Stub surface check ──────────────────────────────────────────────
    const stub_check = "build/validate/stub_surface_check.zig";
    if (!util.fileExists(io, stub_check)) {
        std.debug.print("ERROR: stub surface check missing: {s}\n", .{stub_check});
        errors += 1;
    }

    // ── WDBX fast test root ─────────────────────────────────────────────
    const wdbx_fast_root = "src/wdbx_fast_tests_root.zig";
    if (util.fileExists(io, wdbx_fast_root)) {
        // If it exists, make sure the WDBX engine file exists too
        const wdbx_engine = "src/wdbx/wdbx.zig";
        if (!util.fileExists(io, wdbx_engine)) {
            std.debug.print("ERROR: WDBX fast test root exists but engine missing: {s}\n", .{wdbx_engine});
            errors += 1;
        }
    }

    // ── CLI TUI test root ───────────────────────────────────────────────
    const cli_tui_root = "build/cli_tui_tests_root.zig";
    if (!util.fileExists(io, cli_tui_root)) {
        std.debug.print("WARNING: CLI TUI test root missing: {s}\n", .{cli_tui_root});
    }

    // ── Flags validation file ───────────────────────────────────────────
    const flags_file = "build/flags.zig";
    if (!util.fileExists(io, flags_file)) {
        std.debug.print("ERROR: flag validation matrix missing: {s}\n", .{flags_file});
        errors += 1;
    }

    if (errors > 0) {
        std.debug.print("FAILED: test baseline consistency check found {d} issue(s)\n", .{errors});
        std.process.exit(1);
    }

    std.debug.print("OK: test baseline consistency checks passed\n", .{});
}
