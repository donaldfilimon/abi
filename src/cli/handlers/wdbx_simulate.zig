//! `abi wdbx simulate ...` — bounded multiway rewriting experiments.
//!
//! CLI front-end for `features.wdbx.multiway`: inline rules, rule files,
//! JSON config files, canonical JSON / Graphviz DOT export, optional WDBX
//! persistence, resume from a persisted export, dry-run validation, SIGINT
//! cancellation, and quiet/verbose modes. All exploration is explicitly
//! bounded; a bounded result is only reported "complete" when the frontier
//! drained with no resource bound tripped.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const features = @import("abi").features;

const wdbx = features.wdbx;

const format = @import("wdbx_simulate_format.zig");
const options = @import("wdbx_simulate_options.zig");

pub const help = format.help;
const diag = format.diag;
const printSummary = format.printSummary;
const Options = options.Options;
const parseRuleLine = options.parseRuleLine;
const loadRulesFile = options.loadRulesFile;
const loadConfigFile = options.loadConfigFile;
const parseUintFlag = options.parseUintFlag;
const stringFlag = options.stringFlag;
const MAX_RESUME_FILE_BYTES = options.MAX_RESUME_FILE_BYTES;

var sigint_cancel = std.atomic.Value(bool).init(false);

fn onSignal(sig: @TypeOf(std.posix.SIG.INT)) callconv(.c) void {
    _ = sig;
    sigint_cancel.store(true, .release);
}

fn installCancelHandler() void {
    switch (builtin.os.tag) {
        // Windows has no POSIX sigaction; Ctrl-C falls back to default
        // process termination there (documented gap, same as the MCP server).
        .windows => {},
        else => {
            const posix = std.posix;
            const handler = posix.Sigaction{
                .handler = .{ .handler = onSignal },
                .mask = posix.sigemptyset(),
                .flags = 0,
            };
            posix.sigaction(posix.SIG.INT, &handler, null);
        },
    }
}

/// Entry point; `args` is the full argv (`abi wdbx simulate ...`).
pub fn handleSimulate(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var opts = Options{};
    const flags = args[3..];
    var i: usize = 0;
    while (i < flags.len) : (i += 1) {
        const flag = flags[i];
        const result: ?u8 = blk: {
            if (std.mem.eql(u8, flag, "--initial")) {
                const value = stringFlag(flags, &i, flag) catch break :blk 2;
                opts.initial.append(arena, arena.dupe(u8, value) catch break :blk 2) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--rule")) {
                const value = stringFlag(flags, &i, flag) catch break :blk 2;
                break :blk try parseRuleLine(arena, &opts, value, "--rule");
            } else if (std.mem.eql(u8, flag, "--rules-file")) {
                const value = stringFlag(flags, &i, flag) catch break :blk 2;
                break :blk try loadRulesFile(io, arena, &opts, value);
            } else if (std.mem.eql(u8, flag, "--config")) {
                const value = stringFlag(flags, &i, flag) catch break :blk 2;
                break :blk try loadConfigFile(io, arena, &opts, value);
            } else if (std.mem.eql(u8, flag, "--depth")) {
                opts.max_depth = parseUintFlag(u32, flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--max-states")) {
                opts.max_states = parseUintFlag(u32, flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--max-events")) {
                opts.max_events = parseUintFlag(u32, flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--max-payload")) {
                opts.max_payload = parseUintFlag(u32, flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--deadline-ms")) {
                opts.max_duration_ms = parseUintFlag(u64, flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--max-memory")) {
                opts.max_memory_bytes = parseUintFlag(u64, flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--seed")) {
                opts.seed = parseUintFlag(u64, flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--workers")) {
                const workers = parseUintFlag(u32, flags, &i, flag) catch break :blk 2;
                if (workers == 0) break :blk diag("--workers must be >= 1", .{});
                opts.workers = workers;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--format")) {
                const value = stringFlag(flags, &i, flag) catch break :blk 2;
                if (std.mem.eql(u8, value, "summary")) {
                    opts.format = .summary;
                } else if (std.mem.eql(u8, value, "json")) {
                    opts.format = .json;
                } else if (std.mem.eql(u8, value, "dot")) {
                    opts.format = .dot;
                } else {
                    break :blk diag("--format must be summary, json, or dot (got '{s}')", .{value});
                }
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--output")) {
                opts.output = stringFlag(flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--store")) {
                opts.store = stringFlag(flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--resume")) {
                opts.resume_file = stringFlag(flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--resume-wdbx")) {
                opts.resume_wdbx = stringFlag(flags, &i, flag) catch break :blk 2;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--dry-run")) {
                opts.dry_run = true;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--quiet")) {
                opts.quiet = true;
                break :blk null;
            } else if (std.mem.eql(u8, flag, "--verbose")) {
                opts.verbose = true;
                break :blk null;
            }
            break :blk diag("unknown flag '{s}' (see `abi wdbx simulate --help`)", .{flag});
        };
        if (result) |code| return code;
    }

    if (opts.resume_file != null and opts.resume_wdbx != null) {
        return diag("--resume and --resume-wdbx are mutually exclusive", .{});
    }

    // Assemble the effective config.
    const config = wdbx.multiway.Config{
        .initial = opts.initial.items,
        .rules = opts.rules.items,
        .max_depth = opts.max_depth orelse 5,
        .max_states = opts.max_states orelse 10_000,
        .max_events = opts.max_events orelse 100_000,
        .max_payload = opts.max_payload orelse 4096,
        .max_duration_ms = opts.max_duration_ms orelse 0,
        .max_memory_bytes = opts.max_memory_bytes orelse 0,
        .seed = opts.seed orelse 0,
        .workers = opts.workers orelse 1,
    };

    wdbx.multiway.validateConfig(config) catch |err| {
        return switch (err) {
            error.NoInitialStates => diag("no initial states; pass --initial or a config file", .{}),
            error.NoRules => diag("no rules; pass --rule, --rules-file, or a config file", .{}),
            error.TooManyRules => diag("too many rules (max {d})", .{wdbx.multiway.MAX_RULES}),
            error.EmptyLhs => diag("a rule has an empty left-hand side", .{}),
            error.InitialPayloadTooLarge => diag("an initial state exceeds --max-payload ({d} bytes)", .{config.max_payload}),
            error.ZeroBound => diag("--depth, --max-states, --max-events, and --max-payload must all be >= 1", .{}),
        };
    };

    if (opts.dry_run) {
        std.debug.print(
            "dry-run OK: {d} initial state(s), {d} rule(s), depth<={d}, states<={d}, events<={d}, payload<={d}B, deadline={d}ms, memory<={d}B, seed={d}, workers={d}\n",
            .{ config.initial.len, config.rules.len, config.max_depth, config.max_states, config.max_events, config.max_payload, config.max_duration_ms, config.max_memory_bytes, config.seed, config.workers },
        );
        if (opts.verbose) {
            for (config.rules, 0..) |rule, idx| {
                std.debug.print("  rule {d}: '{s}' -> '{s}'\n", .{ idx, rule.lhs, rule.rhs });
            }
        }
        return 0;
    }

    sigint_cancel.store(false, .release);
    installCancelHandler();

    var result = blk: {
        if (opts.resume_file) |path| {
            const export_json = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(MAX_RESUME_FILE_BYTES)) catch |err| {
                return diag("cannot read resume export '{s}': {s}", .{ path, @errorName(err) });
            };
            break :blk resumeOrDiag(arena, export_json, config);
        }
        if (opts.resume_wdbx) |path| {
            const export_json = wdbx.multiway.loadExportFromWdbx(io, arena, path, null) catch |err| {
                return diag("cannot load experiment from WDBX checkpoint '{s}': {s}", .{ path, @errorName(err) });
            };
            break :blk resumeOrDiag(arena, export_json, config);
        }
        break :blk try wdbx.multiway.run(arena, config, &sigint_cancel);
    } orelse return 2;

    var metrics = try wdbx.multiway.computeMetrics(arena, &result);
    const export_json = try wdbx.multiway.exportCanonicalJson(arena, config, &result, &metrics);
    const export_hash = wdbx.multiway.exportHashHex(export_json);

    if (!opts.quiet) {
        printSummary(&result, &metrics, &export_hash, opts.verbose);
    }

    switch (opts.format) {
        .summary => {},
        .json => {
            if (opts.output) |path| {
                std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = export_json }) catch |err| {
                    std.debug.print("simulate: cannot write '{s}': {s}\n", .{ path, @errorName(err) });
                    return 1;
                };
                if (!opts.quiet) std.debug.print("canonical JSON export written to {s}\n", .{path});
            } else {
                std.debug.print("{s}\n", .{export_json});
            }
        },
        .dot => {
            const dot = try wdbx.multiway.exportDot(arena, config, &result);
            if (opts.output) |path| {
                std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = dot }) catch |err| {
                    std.debug.print("simulate: cannot write '{s}': {s}\n", .{ path, @errorName(err) });
                    return 1;
                };
                if (!opts.quiet) std.debug.print("DOT export written to {s}\n", .{path});
            } else {
                std.debug.print("{s}", .{dot});
            }
        },
    }

    if (opts.store) |path| {
        wdbx.multiway.persistToWdbx(io, arena, path, config, &result, export_json) catch |err| {
            std.debug.print("simulate: WDBX persistence to '{s}' failed: {s}\n", .{ path, @errorName(err) });
            return 1;
        };
        if (!opts.quiet) {
            const cfg_hash = try wdbx.multiway.configHash(arena, config);
            std.debug.print("persisted to WDBX checkpoint {s} (config {s})\n", .{ path, std.fmt.bytesToHex(cfg_hash, .lower) });
        }
    }

    return 0;
}

/// Wraps `resume_` with actionable diagnostics; null means "already reported,
/// exit 2".
fn resumeOrDiag(allocator: std.mem.Allocator, export_json: []const u8, config: wdbx.multiway.Config) ?wdbx.multiway.Result {
    return wdbx.multiway.resume_(allocator, export_json, config, &sigint_cancel) catch |err| {
        _ = switch (err) {
            error.AlreadyComplete => diag("experiment already ran to completion; nothing to resume", .{}),
            error.ConfigMismatch => diag("resume config mismatch: rules/initial states must match the persisted experiment (only bounds may change)", .{}),
            error.UnsupportedFormat => diag("unsupported export format (expected {s})", .{wdbx.multiway.FORMAT_VERSION}),
            error.MalformedExport => diag("malformed export document", .{}),
            error.OutOfMemory => diag("out of memory while parsing export", .{}),
        };
        return null;
    };
}

const handleWdbx = @import("wdbx.zig").handleWdbx;
const test_helpers = @import("abi").foundation.test_helpers;

fn cleanupArtifacts(paths: []const []const u8) void {
    for (paths) |path| test_helpers.deleteTestFileIfExists(path);
}

test "simulate dry-run validates and reports the plan" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{
        "abi",       "wdbx", "simulate", "--initial", "A", "--rule", "A->AB",
        "--dry-run",
    }));
}

test "simulate rejects invalid rules and impossible bounds with exit 2" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    // Missing arrow.
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "simulate", "--initial", "A", "--rule", "AB" }));
    // Empty LHS.
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "simulate", "--initial", "A", "--rule", "->B" }));
    // No rules at all.
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "simulate", "--initial", "A" }));
    // No initial state.
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "simulate", "--rule", "A->B" }));
    // Zero bound.
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "simulate", "--initial", "A", "--rule", "A->B", "--depth", "0" }));
    // Malformed numeric flag.
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "simulate", "--initial", "A", "--rule", "A->B", "--depth", "notanumber" }));
    // Unknown flag.
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "simulate", "--initial", "A", "--rule", "A->B", "--frobnicate" }));
    // workers 0.
    try std.testing.expectEqual(@as(u8, 2), try handleWdbx(std.testing.io, allocator, &.{ "abi", "wdbx", "simulate", "--initial", "A", "--rule", "A->B", "--workers", "0" }));
}

test "simulate runs the reference experiment and writes canonical JSON" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const out_path = "zig-out/multiway-cli-ref.json";
    cleanupArtifacts(&.{out_path});
    defer cleanupArtifacts(&.{out_path});

    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{
        "abi",          "wdbx",         "simulate",
        "--initial",    "A",            "--rule",
        "A->AB",        "--rule",       "A->BA",
        "--rule",       "BB->A",        "--depth",
        "5",            "--max-states", "500",
        "--max-events", "5000",         "--format",
        "json",         "--output",     out_path,
        "--quiet",
    }));

    const content = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, out_path, allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(content);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"format\":\"abi-multiway-v1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"unique_states\":31") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"unique_transitions\":62") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"termination\":\"max_depth\"") != null);
}

test "simulate DOT export and rules-file input" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const rules_path = "zig-out/multiway-cli-rules.txt";
    const dot_path = "zig-out/multiway-cli-graph.dot";
    cleanupArtifacts(&.{ rules_path, dot_path });
    defer cleanupArtifacts(&.{ rules_path, dot_path });

    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = rules_path, .data = "# reference rules\nA->AB\nA->BA\nBB->A\n" });
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{
        "abi",     "wdbx", "simulate", "--initial", "A",        "--rules-file", rules_path,
        "--depth", "3",    "--format", "dot",       "--output", dot_path,       "--quiet",
    }));
    const dot = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, dot_path, allocator, .limited(1024 * 1024));
    defer allocator.free(dot);
    try std.testing.expect(std.mem.indexOf(u8, dot, "digraph multiway") != null);
}

test "simulate config file + WDBX persistence + resume round-trip" {
    if (!build_options.feat_wdbx) return;
    const allocator = std.testing.allocator;
    const config_path = "zig-out/multiway-cli-config.json";
    const export_path = "zig-out/multiway-cli-export.json";
    const export2_path = "zig-out/multiway-cli-export2.json";
    const store_path = "zig-out/multiway-cli-store.jsonl";
    cleanupArtifacts(&.{ config_path, export_path, export2_path });
    cleanupStore(store_path);
    defer cleanupArtifacts(&.{ config_path, export_path, export2_path });
    defer cleanupStore(store_path);

    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = config_path, .data =
        \\{"initial":["A"],"rules":["A->AB","A->BA","BB->A"],"max_depth":3,"max_states":500,"max_events":5000,"max_payload":64}
    });

    // Bounded run persisted to WDBX, canonical export written to a file.
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{
        "abi",      "wdbx",      "simulate", "--config", config_path, "--format", "json",
        "--output", export_path, "--store",  store_path, "--quiet",
    }));

    // Resume from the JSON export with deeper limits.
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{
        "abi",      "wdbx",      "simulate", "--config", config_path, "--depth", "5",
        "--resume", export_path, "--quiet",
    }));

    // Resume from the WDBX checkpoint too.
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{
        "abi",           "wdbx",     "simulate", "--config", config_path, "--depth",    "5",
        "--resume-wdbx", store_path, "--format", "json",     "--output",  export2_path, "--quiet",
    }));

    // The resumed-run export must equal a direct depth-5 run's export.
    const direct_path = "zig-out/multiway-cli-direct.json";
    cleanupArtifacts(&.{direct_path});
    defer cleanupArtifacts(&.{direct_path});
    try std.testing.expectEqual(@as(u8, 0), try handleWdbx(std.testing.io, allocator, &.{
        "abi",      "wdbx", "simulate", "--config",  config_path, "--depth", "5",
        "--format", "json", "--output", direct_path, "--quiet",
    }));
    const resumed = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, export2_path, allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(resumed);
    const direct = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, direct_path, allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(direct);
    try std.testing.expectEqualStrings(direct, resumed);
}

fn cleanupStore(path: []const u8) void {
    var buf: [256]u8 = undefined;
    test_helpers.deleteTestFileIfExists(path);
    if (std.fmt.bufPrint(&buf, "{s}.wal", .{path})) |wp| test_helpers.deleteTestFileIfExists(wp) else |_| {}
    if (std.fmt.bufPrint(&buf, "{s}.manifest", .{path})) |mp| test_helpers.deleteTestFileIfExists(mp) else |_| {}
    var epoch: u64 = 0;
    while (epoch < 8) : (epoch += 1) {
        if (std.fmt.bufPrint(&buf, "{s}.seg.{d}.jsonl", .{ path, epoch })) |sp| test_helpers.deleteTestFileIfExists(sp) else |_| {}
    }
}

test {
    std.testing.refAllDecls(@This());
}
