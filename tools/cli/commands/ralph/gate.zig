//! ralph gate — native replacement for check_ralph_gate.sh + score_ralph_results.py

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const cfg = @import("config.zig");
const workspace = @import("workspace.zig");

const ScoringRule = struct {
    name: []const u8,
    keywords: []const []const u8,
    min_len: usize,
};

const scoring_rules = [_]ScoringRule{
    // Legacy migration rules (baseline)
    .{
        .name = "compat_policy",
        .keywords = &.{ "compat", "legacy", "migration", "deprecation" },
        .min_len = 80,
    },
    .{
        .name = "toolchain_determinism",
        .keywords = &.{ "zvm", ".zigversion", "PATH", "zig version" },
        .min_len = 80,
    },
    .{
        .name = "mod_stub_drift",
        .keywords = &.{ "mod", "stub", "parity", "compile" },
        .min_len = 80,
    },
    .{
        .name = "split_parity_validation",
        .keywords = &.{ "parity", "test", "behavior", "module" },
        .min_len = 80,
    },
    .{
        .name = "migration_mapping",
        .keywords = &.{ "Framework", "App", "Config", "Capability" },
        .min_len = 70,
    },
    // Active plan rules (RM-001 through RM-009)
    .{
        .name = "build_health",
        .keywords = &.{ "pass", "skip", "test", "summary", "feature-tests" },
        .min_len = 40,
    },
    .{
        .name = "cli_framework",
        .keywords = &.{ "command", "descriptor", "provider", "router", "fallback" },
        .min_len = 60,
    },
    .{
        .name = "tui_quality",
        .keywords = &.{ "panel", "render", "layout", "dashboard", "unicode" },
        .min_len = 60,
    },
    .{
        .name = "feature_structure",
        .keywords = &.{ "feature", "module", "import", "catalog", "restructure" },
        .min_len = 60,
    },
    .{
        .name = "skill_learning",
        .keywords = &.{ "skill", "domain", "lesson", "improve", "iterate" },
        .min_len = 40,
    },
};

const ScoreResult = struct {
    score: f64 = 0.0,
    reasons: [8][]const u8 = undefined,
    reason_count: usize = 0,

    fn appendReason(self: *ScoreResult, r: []const u8) void {
        if (self.reason_count < 8) {
            self.reasons[self.reason_count] = r;
            self.reason_count += 1;
        }
    }
};

fn keywordScore(text: []const u8, keywords: []const []const u8) f64 {
    if (keywords.len == 0) return 0.0;
    var hits: f64 = 0.0;
    for (keywords) |kw| {
        if (cfg.containsIgnoreCase(text, kw)) hits += 1.0;
    }
    return hits / @as(f64, @floatFromInt(keywords.len));
}

fn scoreItem(
    output: []const u8,
    notes: []const u8,
    rule: ScoringRule,
    require_live: bool,
) ScoreResult {
    var res = ScoreResult{};

    const is_live = std.mem.indexOf(u8, notes, "provider=openai") != null and
        std.mem.indexOf(u8, notes, "placeholder") == null and
        std.mem.indexOf(u8, notes, "dry_run") == null;

    if (require_live and !is_live) {
        res.appendReason("not_live_openai_output");
    }

    const trimmed_len = std.mem.trim(u8, output, " \t\r\n").len;
    if (trimmed_len < rule.min_len) {
        res.appendReason("output_too_short");
    }

    var score = keywordScore(output, rule.keywords);
    if (trimmed_len >= rule.min_len) {
        score = (score + 1.0) / 2.0;
    }
    if (require_live and !is_live) {
        score *= 0.2;
    }
    res.score = score;
    return res;
}

pub fn runGate(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var in_path: []const u8 = "reports/ralph_upgrade_results_openai.json";
    var out_path: []const u8 = "reports/ralph_upgrade_summary.md";
    var min_average: f64 = 0.75;
    var require_live = false;
    var explicit_input = false;
    var in_path_owned: ?[]u8 = null;
    defer if (in_path_owned) |p| allocator.free(p);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (utils.args.matchesAny(arg, &[_][]const u8{ "--in", "-i" })) {
            i += 1;
            if (i < args.len) {
                in_path = std.mem.sliceTo(args[i], 0);
                explicit_input = true;
            }
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--out", "-o" })) {
            i += 1;
            if (i < args.len) out_path = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--min-average")) {
            i += 1;
            if (i < args.len) min_average = std.fmt.parseFloat(f64, std.mem.sliceTo(args[i], 0)) catch min_average;
        } else if (std.mem.eql(u8, arg, "--require-live")) {
            require_live = true;
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            utils.output.print(
                \\Usage: abi ralph gate [options]
                \\
                \\Native quality gate — replaces check_ralph_gate.sh + score_ralph_results.py.
                \\Scores JSON results against 5 keyword rules, writes Markdown summary.
                \\Exits 0 on pass, 2 on fail.
                \\
                \\Options:
                \\  -i, --in <path>         Input JSON (default: latest .ralph run report, fallback legacy path)
                \\  -o, --out <path>        Output Markdown (default: reports/ralph_upgrade_summary.md)
                \\      --min-average <f>   Minimum average score (default: 0.75)
                \\      --require-live      Require live OpenAI provider outputs
                \\  -h, --help              Show this help
                \\
            , .{});
            return;
        }
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    if (!explicit_input) {
        if (workspace.latestReportPath(allocator, io)) |path| {
            in_path_owned = path;
            in_path = path;
        }
    }

    // Read input JSON
    const json_contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        in_path,
        allocator,
        .limited(16 * 1024 * 1024),
    ) catch {
        utils.output.printError("missing live Ralph results: {s}", .{in_path});
        utils.output.println("Run:\n  abi ralph run --task \"...\" --auto-skill", .{});
        return error.ExecutionFailed;
    };
    defer allocator.free(json_contents);

    // Parse JSON
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json_contents,
        .{},
    ) catch {
        utils.output.printError("Invalid JSON in {s}", .{in_path});
        return error.ExecutionFailed;
    };
    defer parsed.deinit();

    // New run-report format: object with last_gate_passed + last_gate_exit.
    switch (parsed.value) {
        .object => |obj| {
            if (obj.get("last_gate_passed") != null) {
                const gate_passed = if (obj.get("last_gate_passed")) |v| switch (v) {
                    .bool => |b| b,
                    else => false,
                } else false;
                const gate_exit = if (obj.get("last_gate_exit")) |v| switch (v) {
                    .integer => |code| code,
                    else => 1,
                } else 1;
                const run_id = if (obj.get("run_id")) |v| switch (v) {
                    .string => |s| s,
                    else => "",
                } else "";
                const summary = try std.fmt.allocPrint(allocator,
                    \\# Ralph Gate (Run Report)
                    \\
                    \\- Input: `{s}`
                    \\- Run ID: `{s}`
                    \\- Result: **{s}**
                    \\- Exit code: {d}
                    \\
                , .{
                    in_path,
                    run_id,
                    if (gate_passed) "PASS" else "FAIL",
                    gate_exit,
                });
                defer allocator.free(summary);
                if (std.fs.path.dirname(out_path)) |dir| cfg.ensureDir(io, dir);
                cfg.writeFile(allocator, io, out_path, summary) catch {};
                utils.output.println("{s}", .{summary});
                if (!gate_passed) return error.ExecutionFailed;
                utils.output.printSuccess("Ralph gate passed ({s}).", .{out_path});
                return;
            }
        },
        else => {},
    }

    const items = switch (parsed.value) {
        .array => |a| a.items,
        else => {
            utils.output.printError("Results JSON must be a top-level array.", .{});
            return error.ExecutionFailed;
        },
    };

    if (items.len == 0) {
        utils.output.printError("Results file is empty.", .{});
        return error.ExecutionFailed;
    }

    // Score each item against its matching rule
    const count = @min(items.len, scoring_rules.len);
    var total: f64 = 0.0;

    var rows = std.ArrayListUnmanaged(u8).empty;
    defer rows.deinit(allocator);

    for (0..count) |idx| {
        const item = switch (items[idx]) {
            .object => |o| o,
            else => continue,
        };
        const rule = scoring_rules[idx];

        const output_str = if (item.get("output")) |v| switch (v) {
            .string => |s| s,
            else => "",
        } else "";
        const notes_str = if (item.get("notes")) |v| switch (v) {
            .string => |s| s,
            else => "",
        } else "";

        const scored = scoreItem(output_str, notes_str, rule, require_live);
        total += scored.score;

        // Append markdown row
        try rows.appendSlice(allocator, "- `");
        try rows.appendSlice(allocator, rule.name);
        var score_buf: [32]u8 = undefined;
        const score_str = std.fmt.bufPrint(&score_buf, "`: {d:.3} (", .{scored.score}) catch "`: ? (";
        try rows.appendSlice(allocator, score_str);
        if (scored.reason_count == 0) {
            try rows.appendSlice(allocator, "ok");
        } else {
            for (scored.reasons[0..scored.reason_count], 0..) |r, ri| {
                if (ri > 0) try rows.appendSlice(allocator, ", ");
                try rows.appendSlice(allocator, r);
            }
        }
        try rows.appendSlice(allocator, ")\n");
    }

    const avg = total / @as(f64, @floatFromInt(@max(count, 1)));
    const passed = avg >= min_average;
    const status_str = if (passed) "PASS" else "FAIL";
    const live_text = if (require_live) "required" else "optional";

    const summary = try std.fmt.allocPrint(allocator,
        \\# Ralph Upgrade Score
        \\
        \\- Input: `{s}`
        \\- Live OpenAI outputs: {s}
        \\- Average score: {d:.3}
        \\- Threshold: {d:.3}
        \\- Result: **{s}**
        \\
        \\## Item scores
        \\{s}
    , .{ in_path, live_text, avg, min_average, status_str, rows.items });
    defer allocator.free(summary);

    // Ensure output parent directory exists, then write
    if (std.fs.path.dirname(out_path)) |dir| cfg.ensureDir(io, dir);
    cfg.writeFile(allocator, io, out_path, summary) catch |err| {
        utils.output.printWarning("could not write {s}: {t}", .{ out_path, err });
    };

    utils.output.println("{s}", .{summary});

    if (!passed) {
        utils.output.printError("Ralph gate did not pass (avg={d:.3} < threshold={d:.3})", .{ avg, min_average });
        return error.ExecutionFailed;
    }
    utils.output.printSuccess("Ralph gate passed ({s}).", .{out_path});
}
