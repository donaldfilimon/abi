const std = @import("std");
const baseline = @import("baseline.zig");
const util = @import("util.zig");

const Mode = enum {
    all,
    main_only,
    feature_only,
};

const Counts = struct {
    pass: usize = 0,
    skip: usize = 0,
    total: usize = 0,
};

fn parseLeadingInt(text: []const u8) ?usize {
    if (text.len == 0) return null;
    for (text) |ch| {
        if (!std.ascii.isDigit(ch)) return null;
    }
    return std.fmt.parseInt(usize, text, 10) catch null;
}

fn findCountBeforeWord(output: []const u8, word: []const u8) usize {
    var search_from: usize = 0;

    while (std.mem.indexOfPos(u8, output, search_from, word)) |word_idx| {
        var end = word_idx;
        while (end > 0 and output[end - 1] == ' ') : (end -= 1) {}

        var start = end;
        while (start > 0 and std.ascii.isDigit(output[start - 1])) : (start -= 1) {}

        if (start < end) {
            if (parseLeadingInt(output[start..end])) |value| {
                return value;
            }
        }

        search_from = word_idx + word.len;
    }

    return 0;
}

fn parseSummary(output: []const u8) Counts {
    return .{
        .pass = findCountBeforeWord(output, "pass"),
        .skip = findCountBeforeWord(output, "skip"),
        .total = findCountBeforeWord(output, "total"),
    };
}

fn validateMain(allocator: std.mem.Allocator, errors: *usize) !void {
    std.debug.print("Running main tests...\n", .{});

    const result = try util.captureCommand(allocator, "zig build test --summary all");
    defer allocator.free(result.output);

    const actual = parseSummary(result.output);

    std.debug.print(
        "  Expected: {d} pass, {d} skip ({d} total)\n",
        .{ baseline.test_main_pass, baseline.test_main_skip, baseline.test_main_total },
    );
    std.debug.print(
        "  Actual:   {d} pass, {d} skip ({d} total)\n",
        .{ actual.pass, actual.skip, actual.total },
    );

    if (actual.pass < baseline.test_main_pass) {
        std.debug.print(
            "  ERROR: Main test pass count regressed ({d} < {d})\n",
            .{ actual.pass, baseline.test_main_pass },
        );
        errors.* += 1;
    } else if (actual.pass > baseline.test_main_pass) {
        std.debug.print(
            "  NOTICE: Main pass count increased — update tools/scripts/baseline.zig ({d} > {d})\n",
            .{ actual.pass, baseline.test_main_pass },
        );
        errors.* += 1;
    }

    if (actual.total != baseline.test_main_total) {
        std.debug.print(
            "  NOTICE: Main total changed ({d} != {d}) — update tools/scripts/baseline.zig\n",
            .{ actual.total, baseline.test_main_total },
        );
        errors.* += 1;
    }
}

fn validateFeature(allocator: std.mem.Allocator, errors: *usize) !void {
    std.debug.print("Running feature tests...\n", .{});

    const result = try util.captureCommand(allocator, "zig build feature-tests --summary all");
    defer allocator.free(result.output);

    const actual = parseSummary(result.output);

    std.debug.print(
        "  Expected: {d} pass ({d} total)\n",
        .{ baseline.test_feature_pass, baseline.test_feature_total },
    );
    std.debug.print(
        "  Actual:   {d} pass ({d} total)\n",
        .{ actual.pass, actual.total },
    );

    if (actual.pass < baseline.test_feature_pass) {
        std.debug.print(
            "  ERROR: Feature test pass count regressed ({d} < {d})\n",
            .{ actual.pass, baseline.test_feature_pass },
        );
        errors.* += 1;
    } else if (actual.pass > baseline.test_feature_pass) {
        std.debug.print(
            "  NOTICE: Feature pass count increased — update tools/scripts/baseline.zig ({d} > {d})\n",
            .{ actual.pass, baseline.test_feature_pass },
        );
        errors.* += 1;
    }
}

fn parseMode(args: []const []const u8) !Mode {
    if (args.len == 0) return .all;

    const arg = args[0];
    if (std.mem.eql(u8, arg, "--main-only")) return .main_only;
    if (std.mem.eql(u8, arg, "--feature-only")) return .feature_only;
    if (std.mem.eql(u8, arg, "all")) return .all;

    std.debug.print("Usage: zig run tools/scripts/validate_test_counts.zig -- [--main-only | --feature-only]\n", .{});
    return error.InvalidMode;
}

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var argv = try std.process.argsWithAllocator(allocator);
    defer argv.deinit();

    var args = std.ArrayListUnmanaged([]const u8).empty;
    defer args.deinit(allocator);

    var index: usize = 0;
    while (argv.next()) |arg| : (index += 1) {
        if (index == 0) continue;
        try args.append(allocator, arg);
    }

    const mode = parseMode(args.items) catch {
        std.process.exit(1);
    };

    var errors: usize = 0;

    switch (mode) {
        .main_only => try validateMain(allocator, &errors),
        .feature_only => try validateFeature(allocator, &errors),
        .all => {
            try validateMain(allocator, &errors);
            try validateFeature(allocator, &errors);
        },
    }

    if (errors > 0) {
        std.debug.print("\nFAILED: Test baseline validation found {d} issue(s)\n", .{errors});
        std.debug.print("If counts increased, update tools/scripts/baseline.zig and sync docs.\n", .{});
        std.process.exit(1);
    }

    std.debug.print("\nOK: Test counts match baseline\n", .{});
}
