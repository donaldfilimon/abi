const std = @import("std");
const util = @import("util");

/// Known test roots that pull in test blocks from the source tree.
/// Files reachable (via imports) from any of these roots are covered.
const test_roots = [_][]const u8{
    "src/services/tests/mod.zig",
    "src/root.zig",
    "src/feature_parity_tests.zig",
};

/// Files that are expected to have test blocks but are NOT reachable from
/// the standard test roots (standalone test files, build-layer tests, etc.).
/// These are excluded from the orphan check.
const allowed_orphans = [_][]const u8{
    // Build-layer test files (compiled via separate build steps)
    "build/",
    "tools/",
    "tests/",
    "benchmarks/",
    // Database-specific test roots (compiled separately by build.zig)
    "src/database_wdbx_tests_root.zig",
    "src/database_fast_tests_root.zig",
    // The test roots themselves
    "src/services/tests/mod.zig",
    "src/feature_parity_tests.zig",
};

fn captureScan(allocator: std.mem.Allocator, io: std.Io, cmd: []const u8) ![]u8 {
    const result = try util.captureCommand(allocator, io, cmd);
    defer allocator.free(result.output);

    // exit code 1 means "no matches" for grep/rg — not an error
    if (result.exit_code != 0 and result.exit_code != 1) {
        std.debug.print("ERROR: scan command failed (exit={d}): {s}\n", .{ result.exit_code, cmd });
        if (result.output.len > 0) std.debug.print("{s}", .{result.output});
        std.process.exit(1);
    }

    return allocator.dupe(u8, result.output);
}

fn isAllowedOrphan(path: []const u8) bool {
    for (allowed_orphans) |prefix| {
        if (std.mem.startsWith(u8, path, prefix)) return true;
    }
    return false;
}

/// Check if a file is imported by any other .zig file under src/.
/// Uses `rg --fixed-strings` to avoid regex metacharacter issues.
/// NOTE: Searches by basename, which can produce false negatives when
/// two unrelated files share the same name (e.g. session.zig in
/// different directories).  This is acceptable for an advisory tool.
fn isImportedAnywhere(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_basename: []const u8,
) !bool {
    // Use --fixed-strings to avoid regex metacharacter issues with
    // filenames containing dots, hyphens, etc.
    const cmd = try std.fmt.allocPrint(
        allocator,
        "rg -l --fixed-strings --glob '*.zig' '{s}' src/",
        .{file_basename},
    );
    defer allocator.free(cmd);

    const output = try captureScan(allocator, io, cmd);
    defer allocator.free(output);

    return std.mem.trim(u8, output, " \t\r\n").len > 0;
}

fn basename(path: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |idx| {
        return path[idx + 1 ..];
    }
    return path;
}

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    // Step 1: Find all .zig files under src/ that contain test blocks.
    // Match both `test {` (unnamed) and `test "name" {` (named) patterns.
    const test_files_output = try captureScan(
        allocator,
        io,
        "rg -l --glob '*.zig' '(^|\\s)test\\s+(\\{|\"[^\"]*\"\\s*\\{)' src/",
    );
    defer allocator.free(test_files_output);

    var orphan_count: usize = 0;
    var total_test_files: usize = 0;
    var checked_files: usize = 0;

    var lines = std.mem.splitScalar(u8, test_files_output, '\n');
    while (lines.next()) |line| {
        const path = std.mem.trim(u8, line, " \t\r\n");
        if (path.len == 0) continue;
        total_test_files += 1;

        // Skip files in allowed orphan directories
        if (isAllowedOrphan(path)) continue;
        checked_files += 1;

        // Check if this file is imported by any other file in src/
        const file_base = basename(path);
        const imported = isImportedAnywhere(allocator, io, file_base) catch false;

        if (!imported) {
            // Double-check: is this file a mod.zig that would be imported
            // by name via the parent directory convention?
            if (std.mem.eql(u8, file_base, "mod.zig") or
                std.mem.eql(u8, file_base, "root.zig") or
                std.mem.eql(u8, file_base, "types.zig") or
                std.mem.eql(u8, file_base, "stub.zig"))
            {
                // These are structural files — almost certainly imported
                // by the parent's mod.zig. Skip the false positive.
                continue;
            }

            std.debug.print("ORPHAN: {s} has test blocks but is not imported by any src/ file\n", .{path});
            orphan_count += 1;
        }
    }

    if (orphan_count > 0) {
        std.debug.print(
            "\nERROR: {d} orphaned test file(s) found ({d} total files with tests, {d} checked)\n",
            .{ orphan_count, total_test_files, checked_files },
        );
        std.debug.print("These files contain test blocks but are not imported from any test root.\n", .{});
        std.debug.print("Fix: import the file from the parent mod.zig or a test root.\n", .{});
        std.process.exit(1);
    }

    std.debug.print(
        "OK: {d} src/ test files checked, no orphans ({d} total files with tests)\n",
        .{ checked_files, total_test_files },
    );
}
