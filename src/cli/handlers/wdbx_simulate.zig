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

const MAX_RULE_FILE_BYTES = 1 * 1024 * 1024;
const MAX_CONFIG_FILE_BYTES = 1 * 1024 * 1024;
const MAX_RESUME_FILE_BYTES = 256 * 1024 * 1024;

pub fn help() u8 {
    std.debug.print(
        \\usage: abi wdbx simulate [options]
        \\
        \\Run a bounded multiway (Wolfram-style) string-rewriting experiment.
        \\Simulates a finite, explicitly bounded slice of rule space; it does not
        \\enumerate "the ruliad" and makes no physics claims.
        \\
        \\Rules & initial states
        \\  --initial <STATE>        Initial state (repeatable; at least one required)
        \\  --rule '<LHS->RHS>'      Inline rewriting rule (repeatable)
        \\  --rules-file <PATH>      One rule per line; blank lines and # comments ignored
        \\  --config <PATH>          JSON experiment config (flags override file values)
        \\
        \\Bounds (hard limits; partial results are returned when one is reached)
        \\  --depth <N>              Maximum depth (default 5)
        \\  --max-states <N>         Maximum unique states (default 10000)
        \\  --max-events <N>         Maximum events (default 100000)
        \\  --max-payload <N>        Maximum state payload bytes (default 4096)
        \\  --deadline-ms <N>        Wall-clock budget in ms (0 = unlimited)
        \\  --max-memory <N>         Approximate engine memory budget in bytes (0 = unlimited)
        \\
        \\Reproducibility
        \\  --seed <N>               Recorded random seed (engine is deterministic)
        \\  --workers <N>            Recorded worker count (expansion is single-threaded)
        \\
        \\Output & persistence
        \\  --format <summary|json|dot>  Output format (default summary)
        \\  --output <PATH>          Write json/dot export to a file instead of printing
        \\  --store <PATH>           Persist experiment into a WDBX checkpoint at PATH
        \\  --resume <PATH>          Resume from a canonical JSON export file
        \\  --resume-wdbx <PATH>     Resume from the latest experiment in a WDBX checkpoint
        \\  --dry-run                Validate configuration and print the plan, then exit
        \\  --quiet                  Suppress the human summary
        \\  --verbose                Print per-depth tables and extra detail
        \\
        \\Ctrl-C cancels a running experiment; the partial result is still
        \\summarized/exported with termination reason "cancelled".
        \\
        \\example:
        \\  abi wdbx simulate --initial A --rule 'A->AB' --rule 'A->BA' --rule 'BB->A' \
        \\      --depth 5 --max-states 500 --max-events 5000 --format json --output experiment.json
        \\
    , .{});
    return 0;
}

fn diag(comptime fmt: []const u8, args: anytype) u8 {
    std.debug.print("simulate: " ++ fmt ++ "\n", .{} ++ args);
    return 2;
}

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

const Options = struct {
    initial: std.ArrayListUnmanaged([]const u8) = .empty,
    rules: std.ArrayListUnmanaged(wdbx.multiway.Rule) = .empty,
    max_depth: ?u32 = null,
    max_states: ?u32 = null,
    max_events: ?u32 = null,
    max_payload: ?u32 = null,
    max_duration_ms: ?u64 = null,
    max_memory_bytes: ?u64 = null,
    seed: ?u64 = null,
    workers: ?u32 = null,
    format: enum { summary, json, dot } = .summary,
    output: ?[]const u8 = null,
    store: ?[]const u8 = null,
    resume_file: ?[]const u8 = null,
    resume_wdbx: ?[]const u8 = null,
    dry_run: bool = false,
    quiet: bool = false,
    verbose: bool = false,
};

fn parseRuleLine(allocator: std.mem.Allocator, opts: *Options, text: []const u8, origin: []const u8) !?u8 {
    const rule = wdbx.multiway.parseRule(allocator, text) catch |err| switch (err) {
        error.MissingArrow => return diag("invalid rule '{s}' ({s}): expected 'LHS->RHS'", .{ text, origin }),
        error.EmptyLhs => return diag("invalid rule '{s}' ({s}): left-hand side must be non-empty", .{ text, origin }),
        error.OutOfMemory => return error.OutOfMemory,
    };
    try opts.rules.append(allocator, rule);
    return null;
}

fn loadRulesFile(io: std.Io, allocator: std.mem.Allocator, opts: *Options, path: []const u8) !?u8 {
    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(MAX_RULE_FILE_BYTES)) catch |err| {
        return diag("cannot read rules file '{s}': {s}", .{ path, @errorName(err) });
    };
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        if (try parseRuleLine(allocator, opts, line, path)) |code| return code;
    }
    return null;
}

fn jsonUint(value: std.json.Value) ?u64 {
    return switch (value) {
        .integer => |n| if (n < 0) null else @intCast(n),
        else => null,
    };
}

fn loadConfigFile(io: std.Io, allocator: std.mem.Allocator, opts: *Options, path: []const u8) !?u8 {
    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(MAX_CONFIG_FILE_BYTES)) catch |err| {
        return diag("cannot read config file '{s}': {s}", .{ path, @errorName(err) });
    };
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, allocator, content, .{}) catch {
        return diag("config file '{s}' is not valid JSON", .{path});
    };
    const root = switch (parsed) {
        .object => |obj| obj,
        else => return diag("config file '{s}': top level must be a JSON object", .{path}),
    };
    if (root.get("initial")) |value| {
        const arr = switch (value) {
            .array => |a| a,
            else => return diag("config '{s}': \"initial\" must be an array of strings", .{path}),
        };
        for (arr.items) |item| {
            const text = switch (item) {
                .string => |s| s,
                else => return diag("config '{s}': \"initial\" entries must be strings", .{path}),
            };
            try opts.initial.append(allocator, try allocator.dupe(u8, text));
        }
    }
    if (root.get("rules")) |value| {
        const arr = switch (value) {
            .array => |a| a,
            else => return diag("config '{s}': \"rules\" must be an array", .{path}),
        };
        for (arr.items) |item| {
            switch (item) {
                .string => |text| {
                    if (try parseRuleLine(allocator, opts, text, path)) |code| return code;
                },
                .object => |obj| {
                    const text = switch (obj.get("rule") orelse return diag("config '{s}': rule object missing \"rule\"", .{path})) {
                        .string => |s| s,
                        else => return diag("config '{s}': rule object \"rule\" must be a string", .{path}),
                    };
                    if (try parseRuleLine(allocator, opts, text, path)) |code| return code;
                    const rule = &opts.rules.items[opts.rules.items.len - 1];
                    if (obj.get("weight")) |weight_value| {
                        rule.weight = switch (weight_value) {
                            .float => |f| f,
                            .integer => |n| @floatFromInt(n),
                            else => return diag("config '{s}': rule \"weight\" must be a number", .{path}),
                        };
                    }
                    if (obj.get("family")) |family_value| {
                        rule.family = switch (family_value) {
                            .string => |s| try allocator.dupe(u8, s),
                            else => return diag("config '{s}': rule \"family\" must be a string", .{path}),
                        };
                    }
                },
                else => return diag("config '{s}': \"rules\" entries must be strings or objects", .{path}),
            }
        }
    }
    const uint_fields = .{
        .{ "max_depth", "max_depth" },
        .{ "max_states", "max_states" },
        .{ "max_events", "max_events" },
        .{ "max_payload", "max_payload" },
    };
    inline for (uint_fields) |field| {
        if (root.get(field[0])) |value| {
            const n = jsonUint(value) orelse return diag("config '{s}': \"{s}\" must be a non-negative integer", .{ path, field[0] });
            if (n > std.math.maxInt(u32)) return diag("config '{s}': \"{s}\" too large", .{ path, field[0] });
            @field(opts, field[1]) = @intCast(n);
        }
    }
    if (root.get("max_duration_ms")) |value| {
        opts.max_duration_ms = jsonUint(value) orelse return diag("config '{s}': \"max_duration_ms\" must be a non-negative integer", .{path});
    }
    if (root.get("max_memory_bytes")) |value| {
        opts.max_memory_bytes = jsonUint(value) orelse return diag("config '{s}': \"max_memory_bytes\" must be a non-negative integer", .{path});
    }
    if (root.get("seed")) |value| {
        opts.seed = jsonUint(value) orelse return diag("config '{s}': \"seed\" must be a non-negative integer", .{path});
    }
    if (root.get("workers")) |value| {
        const n = jsonUint(value) orelse return diag("config '{s}': \"workers\" must be a non-negative integer", .{path});
        if (n == 0 or n > std.math.maxInt(u32)) return diag("config '{s}': \"workers\" out of range", .{path});
        opts.workers = @intCast(n);
    }
    return null;
}

fn parseUintFlag(comptime T: type, args: []const []const u8, index: *usize, flag: []const u8) !?T {
    if (index.* + 1 >= args.len) {
        _ = diag("{s} requires a value", .{flag});
        return error.Reported;
    }
    index.* += 1;
    return std.fmt.parseInt(T, args[index.*], 10) catch {
        _ = diag("{s}: '{s}' is not a valid non-negative integer", .{ flag, args[index.*] });
        return error.Reported;
    };
}

fn stringFlag(args: []const []const u8, index: *usize, flag: []const u8) ![]const u8 {
    if (index.* + 1 >= args.len) {
        _ = diag("{s} requires a value", .{flag});
        return error.Reported;
    }
    index.* += 1;
    return args[index.*];
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

fn printSummary(result: *const wdbx.multiway.Result, metrics: *const wdbx.multiway.Metrics, export_hash: *const [64]u8, verbose: bool) void {
    const elapsed_ms = @as(f64, @floatFromInt(result.elapsed_ns)) / @as(f64, std.time.ns_per_ms);
    const elapsed_s = @max(elapsed_ms / 1000.0, 1e-9);
    std.debug.print(
        \\multiway simulation
        \\  states (unique):        {d}
        \\  events:                 {d}
        \\  transitions (unique):   {d}
        \\  termination:            {s}
        \\  exhaustive in domain:   {}
        \\  mean out-degree:        {d:.3} (unique transitions / unique states)
        \\  max out-degree:         {d}
        \\  median out-degree:      {d:.1}
        \\  convergent states:      {d} (distinct-predecessor in-degree > 1)
        \\  self-loops:             {d}
        \\  cycle present:          {}
        \\  weakly connected comps: {d}
        \\  payload bytes max/mean: {d}/{d:.2}
        \\  runtime:                {d:.3} ms ({d:.0} states/s, {d:.0} events/s)
        \\  export sha256:          {s}
        \\
    , .{
        metrics.unique_states,
        metrics.event_count,
        metrics.unique_transitions,
        metrics.termination.label(),
        metrics.exhaustive,
        metrics.mean_out_degree,
        metrics.max_out_degree,
        metrics.median_out_degree,
        metrics.convergent_states,
        metrics.self_loops,
        metrics.has_cycle,
        metrics.weakly_connected_components,
        metrics.max_payload_bytes,
        metrics.mean_payload_bytes,
        elapsed_ms,
        @as(f64, @floatFromInt(metrics.unique_states)) / elapsed_s,
        @as(f64, @floatFromInt(metrics.event_count)) / elapsed_s,
        export_hash,
    });
    if (verbose) {
        std.debug.print("  depth  states  events  growth\n", .{});
        for (metrics.states_per_depth, 0..) |states, depth| {
            const growth: f64 = if (depth >= 1 and depth - 1 < metrics.growth_rates.len) metrics.growth_rates[depth - 1] else 0.0;
            std.debug.print("  {d: >5}  {d: >6}  {d: >6}  {d: >6.2}\n", .{ depth, states, metrics.events_per_depth[depth], growth });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (run through the public wdbx handler surface)
// ---------------------------------------------------------------------------

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
