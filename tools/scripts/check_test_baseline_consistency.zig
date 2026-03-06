const std = @import("std");
const baseline = @import("baseline.zig");
const util = @import("util.zig");

fn checkContains(content: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, content, needle) != null;
}

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var errors: usize = 0;

    var readme: []const u8 = "";
    var has_readme = false;
    defer if (has_readme) allocator.free(readme);

    if (util.fileExists(io, "README.md")) {
        if (util.readFileAlloc(allocator, io, "README.md", 8 * 1024 * 1024)) |content| {
            readme = content;
            has_readme = true;
        } else |_| {
            std.debug.print("ERROR: failed to read README.md\n", .{});
            errors += 1;
        }
    } else {
        std.debug.print("ERROR: README.md missing\n", .{});
        errors += 1;
    }

    const readme_badge = try std.fmt.allocPrint(allocator, "tests-{d}_passing", .{baseline.test_main_pass});
    defer allocator.free(readme_badge);
    if (has_readme and !checkContains(readme, readme_badge)) {
        std.debug.print("ERROR: README.md missing expected baseline marker for README badge main pass\n", .{});
        errors += 1;
    }

    const readme_narrative = try std.fmt.allocPrint(
        allocator,
        "{d} tests ({d} passing, {d} skip)",
        .{ baseline.test_main_total, baseline.test_main_pass, baseline.test_main_skip },
    );
    defer allocator.free(readme_narrative);
    if (has_readme and !checkContains(readme, readme_narrative)) {
        std.debug.print("ERROR: README.md missing expected baseline marker for README narrative baseline\n", .{});
        errors += 1;
    }

    if (errors > 0) {
        std.debug.print("FAILED: Test baseline consistency check found {d} issue(s)\n", .{errors});
        std.process.exit(1);
    }

    std.debug.print("OK: Test baseline consistency checks passed\n", .{});
}
