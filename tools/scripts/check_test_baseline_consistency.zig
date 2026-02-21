const std = @import("std");
const baseline = @import("baseline.zig");
const util = @import("util.zig");

fn checkContains(content: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, content, needle) != null;
}

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var errors: usize = 0;

    const readme = util.readFileAlloc(allocator, io, "README.md", 8 * 1024 * 1024) catch {
        std.debug.print("ERROR: README.md missing\n", .{});
        std.process.exit(1);
    };
    defer allocator.free(readme);

    const claude = util.readFileAlloc(allocator, io, "CLAUDE.md", 16 * 1024 * 1024) catch {
        std.debug.print("ERROR: CLAUDE.md missing\n", .{});
        std.process.exit(1);
    };
    defer allocator.free(claude);

    const zig_rules = util.readFileAlloc(allocator, io, ".claude/rules/zig.md", 8 * 1024 * 1024) catch {
        std.debug.print("ERROR: .claude/rules/zig.md missing\n", .{});
        std.process.exit(1);
    };
    defer allocator.free(zig_rules);

    const readme_badge = try std.fmt.allocPrint(allocator, "tests-{d}_passing", .{baseline.test_main_pass});
    defer allocator.free(readme_badge);
    if (!checkContains(readme, readme_badge)) {
        std.debug.print("ERROR: README.md missing expected baseline marker for README badge main pass\n", .{});
        errors += 1;
    }

    const readme_narrative = try std.fmt.allocPrint(
        allocator,
        "{d} tests ({d} passing, {d} skip)",
        .{ baseline.test_main_total, baseline.test_main_pass, baseline.test_main_skip },
    );
    defer allocator.free(readme_narrative);
    if (!checkContains(readme, readme_narrative)) {
        std.debug.print("ERROR: README.md missing expected baseline marker for README narrative baseline\n", .{});
        errors += 1;
    }

    const main_baseline = try std.fmt.allocPrint(
        allocator,
        "{d} pass, {d} skip ({d} total)",
        .{ baseline.test_main_pass, baseline.test_main_skip, baseline.test_main_total },
    );
    defer allocator.free(main_baseline);

    const feature_baseline = try std.fmt.allocPrint(
        allocator,
        "{d} pass ({d} total)",
        .{ baseline.test_feature_pass, baseline.test_feature_total },
    );
    defer allocator.free(feature_baseline);

    if (!checkContains(claude, main_baseline)) {
        std.debug.print("ERROR: CLAUDE.md missing expected baseline marker for CLAUDE main baseline\n", .{});
        errors += 1;
    }
    if (!checkContains(claude, feature_baseline)) {
        std.debug.print("ERROR: CLAUDE.md missing expected baseline marker for CLAUDE feature baseline\n", .{});
        errors += 1;
    }
    if (!checkContains(zig_rules, main_baseline)) {
        std.debug.print("ERROR: .claude/rules/zig.md missing expected baseline marker for zig.md main baseline\n", .{});
        errors += 1;
    }
    if (!checkContains(zig_rules, feature_baseline)) {
        std.debug.print("ERROR: .claude/rules/zig.md missing expected baseline marker for zig.md feature baseline\n", .{});
        errors += 1;
    }

    const stale_markers = [_][]const u8{
        "1220/1225",
        "1220 pass",
        "1153 pass",
        "1198 pass",
        "1251 pass",
        "1257 pass",
        "1262 total",
        "1213 pass",
        "671 pass",
        "1252 pass",
        "1512 pass",
        "1534 pass",
    };

    for (stale_markers) |marker| {
        if (checkContains(readme, marker)) {
            std.debug.print("ERROR: README.md contains stale baseline marker '{s}'\n", .{marker});
            errors += 1;
        }
        if (checkContains(claude, marker)) {
            std.debug.print("ERROR: CLAUDE.md contains stale baseline marker '{s}'\n", .{marker});
            errors += 1;
        }
        if (checkContains(zig_rules, marker)) {
            std.debug.print("ERROR: .claude/rules/zig.md contains stale baseline marker '{s}'\n", .{marker});
            errors += 1;
        }
    }

    if (errors > 0) {
        std.debug.print("FAILED: Test baseline consistency check found {d} issue(s)\n", .{errors});
        std.process.exit(1);
    }

    std.debug.print("OK: Test baseline consistency checks passed\n", .{});
}
