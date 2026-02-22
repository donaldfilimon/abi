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
    var claude: []const u8 = "";
    var zig_rules: []const u8 = "";
    var has_readme = false;
    var has_claude = false;
    var has_zig_rules = false;
    defer if (has_readme) allocator.free(readme);
    defer if (has_claude) allocator.free(claude);
    defer if (has_zig_rules) allocator.free(zig_rules);

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

    if (util.fileExists(io, "CLAUDE.md")) {
        if (util.readFileAlloc(allocator, io, "CLAUDE.md", 16 * 1024 * 1024)) |content| {
            claude = content;
            has_claude = true;
        } else |_| {
            std.debug.print("ERROR: failed to read CLAUDE.md\n", .{});
            errors += 1;
        }
    } else {
        std.debug.print("ERROR: CLAUDE.md missing\n", .{});
        errors += 1;
    }

    if (util.fileExists(io, ".claude/rules/zig.md")) {
        if (util.readFileAlloc(allocator, io, ".claude/rules/zig.md", 8 * 1024 * 1024)) |content| {
            zig_rules = content;
            has_zig_rules = true;
        } else |_| {
            std.debug.print("ERROR: failed to read .claude/rules/zig.md\n", .{});
            errors += 1;
        }
    } else {
        std.debug.print("INFO: optional baseline file missing: .claude/rules/zig.md\n", .{});
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

    if (has_claude and !checkContains(claude, main_baseline)) {
        std.debug.print("ERROR: CLAUDE.md missing expected baseline marker for CLAUDE main baseline\n", .{});
        errors += 1;
    }
    if (has_claude and !checkContains(claude, feature_baseline)) {
        std.debug.print("ERROR: CLAUDE.md missing expected baseline marker for CLAUDE feature baseline\n", .{});
        errors += 1;
    }
    if (has_zig_rules and !checkContains(zig_rules, main_baseline)) {
        std.debug.print("ERROR: .claude/rules/zig.md missing expected baseline marker for zig.md main baseline\n", .{});
        errors += 1;
    }
    if (has_zig_rules and !checkContains(zig_rules, feature_baseline)) {
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
        "1976 pass",
        "1535 pass",
        "1975 pass",
        "1996 pass",
        "1997 pass",
        "2001 total",
        "2044 pass",
        "2048 total",
        "2060 pass",
        "2064 total",
        "1254 pass",
        "1259 total",
        "2080 pass",
        "2084 total",
    };

    for (stale_markers) |marker| {
        if (has_readme and checkContains(readme, marker)) {
            std.debug.print("ERROR: README.md contains stale baseline marker '{s}'\n", .{marker});
            errors += 1;
        }
        if (has_claude and checkContains(claude, marker)) {
            std.debug.print("ERROR: CLAUDE.md contains stale baseline marker '{s}'\n", .{marker});
            errors += 1;
        }
        if (has_zig_rules and checkContains(zig_rules, marker)) {
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
