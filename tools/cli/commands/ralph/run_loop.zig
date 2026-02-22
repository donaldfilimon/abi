//! ralph run — Execute iterative loop via Abbey engine or provider router

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const cfg = @import("config.zig");
const skills_store = @import("skills_store.zig");

const providers = abi.ai.llm.providers;
const ProviderId = providers.ProviderId;

pub fn runRun(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var task_override: ?[]const u8 = null;
    var iter_override: ?usize = null;
    var auto_skill = false;
    var store_skill: ?[]const u8 = null;
    var config_path: []const u8 = cfg.CONFIG_FILE;
    var backend_override: ?ProviderId = null;
    var model_override: ?[]const u8 = null;
    var strict_backend_flag = false;
    var plugin_override: ?[]const u8 = null;
    var fallback_buf = std.ArrayListUnmanaged(ProviderId).empty;
    defer fallback_buf.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (utils.args.matchesAny(arg, &[_][]const u8{ "--task", "-t" })) {
            i += 1;
            if (i < args.len) task_override = std.mem.sliceTo(args[i], 0);
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--iterations", "-i" })) {
            i += 1;
            if (i < args.len) iter_override = std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10) catch null;
        } else if (std.mem.eql(u8, arg, "--auto-skill")) {
            auto_skill = true;
        } else if (std.mem.eql(u8, arg, "--store-skill")) {
            i += 1;
            if (i < args.len) store_skill = std.mem.sliceTo(args[i], 0);
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--config", "-c" })) {
            i += 1;
            if (i < args.len) config_path = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i < args.len) {
                const raw = std.mem.sliceTo(args[i], 0);
                backend_override = ProviderId.fromString(raw) orelse {
                    std.debug.print("Unknown provider backend: {s}\n", .{raw});
                    return;
                };
            }
        } else if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i < args.len) model_override = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--fallback")) {
            i += 1;
            if (i < args.len) {
                fallback_buf.clearRetainingCapacity();
                try appendProvidersCsv(allocator, &fallback_buf, std.mem.sliceTo(args[i], 0));
            }
        } else if (std.mem.eql(u8, arg, "--strict-backend")) {
            strict_backend_flag = true;
        } else if (std.mem.eql(u8, arg, "--plugin")) {
            i += 1;
            if (i < args.len) plugin_override = std.mem.sliceTo(args[i], 0);
        } else if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            printHelp();
            return;
        }
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Parse ralph.yml
    var ralph_cfg = cfg.RalphConfig{};
    const yml_contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        config_path,
        allocator,
        .limited(64 * 1024),
    ) catch |err| {
        std.debug.print("Cannot read {s}: {t}\n", .{ config_path, err });
        std.debug.print("Run 'abi ralph init' to create a workspace.\n", .{});
        return;
    };
    defer allocator.free(yml_contents);
    cfg.parseRalphYamlInto(yml_contents, &ralph_cfg);

    const max_iterations = iter_override orelse ralph_cfg.max_iterations;
    const model = model_override orelse ralph_cfg.llm_model;
    const plugin = plugin_override orelse ralph_cfg.llm_plugin;
    const strict_backend = strict_backend_flag or ralph_cfg.llm_strict_backend;

    // Merge fallback: CLI flag wins, otherwise use config value
    if (fallback_buf.items.len == 0 and ralph_cfg.llm_fallback.len > 0) {
        appendProvidersCsv(allocator, &fallback_buf, ralph_cfg.llm_fallback) catch {};
    }
    const fallback_slice = try fallback_buf.toOwnedSlice(allocator);
    defer allocator.free(fallback_slice);

    // Resolve the effective backend: CLI flag > config > null (auto)
    const effective_backend: ?ProviderId = backend_override orelse
        ProviderId.fromString(ralph_cfg.llm_backend);

    // Determine goal: --task flag or PROMPT.md contents
    var prompt_owned: ?[]u8 = null;
    defer if (prompt_owned) |p| allocator.free(p);

    const goal: []const u8 = if (task_override) |t|
        t
    else blk: {
        const content = std.Io.Dir.cwd().readFileAlloc(
            io,
            ralph_cfg.prompt_file,
            allocator,
            .limited(256 * 1024),
        ) catch |err| {
            std.debug.print("Cannot read {s}: {t}\n", .{ ralph_cfg.prompt_file, err });
            std.debug.print("Use --task or edit {s}.\n", .{ralph_cfg.prompt_file});
            return;
        };
        prompt_owned = content;
        break :blk content;
    };

    std.debug.print("Starting Ralph loop (backend: {s}, max_iterations: {d})\n", .{
        if (effective_backend) |b| b.label() else "auto(abbey)",
        max_iterations,
    });
    std.debug.print("Goal: {s}\n\n", .{goal});

    // Decide execution path: use provider router when an explicit non-legacy
    // backend was requested via CLI flag, otherwise fall back to Abbey engine.
    const use_provider_router = backend_override != null;

    if (use_provider_router) {
        // --- Provider-routed generation mode ---
        const result = try runProviderLoop(allocator, .{
            .goal = goal,
            .max_iterations = max_iterations,
            .model = model,
            .backend = effective_backend,
            .fallback = fallback_slice,
            .strict_backend = strict_backend,
            .plugin = plugin,
            .completion_promise = ralph_cfg.completion_promise,
        });
        defer allocator.free(result);

        // Skill storage (no engine available in provider-routed mode)
        const skills_added = handleSkills(allocator, io, store_skill, auto_skill, goal, result);
        updateState(allocator, io, skills_added);
        std.debug.print("\n=== Ralph Run Complete ===\n{s}\n", .{result});
    } else {
        // --- Legacy Abbey engine path ---
        var engine = abi.ai.abbey.createEngine(allocator) catch |err| {
            std.debug.print("Failed to create Abbey engine: {t}\n", .{err});
            return;
        };
        defer engine.deinit();

        const result = engine.runRalphLoop(goal, max_iterations) catch |err| {
            std.debug.print("Ralph loop failed: {t}\n", .{err});
            return;
        };
        defer allocator.free(result);

        engine.recordRalphRun(goal, max_iterations, result.len, 1.0) catch {};

        // Skill storage (with engine support)
        var skills_added: u64 = 0;
        if (store_skill) |s| {
            _ = engine.storeSkill(s) catch {};
            skills_store.appendSkill(allocator, io, s, null, 1.0) catch |err| {
                std.debug.print("Warning: could not persist skill: {t}\n", .{err});
            };
            skills_added += 1;
            std.debug.print("Skill stored.\n", .{});
        }
        if (auto_skill) {
            const stored = engine.extractAndStoreSkill(goal, result) catch false;
            if (stored) {
                if (firstSentence(result)) |lesson| {
                    skills_store.appendSkill(allocator, io, lesson, null, 0.8) catch {};
                }
                skills_added += 1;
                std.debug.print("Auto-skill extracted and stored.\n", .{});
            }
        }

        updateState(allocator, io, skills_added);
        std.debug.print("\n=== Ralph Run Complete ===\n{s}\n", .{result});
    }
}

const ProviderLoopOpts = struct {
    goal: []const u8,
    max_iterations: usize,
    model: []const u8,
    backend: ?ProviderId,
    fallback: []const ProviderId,
    strict_backend: bool,
    plugin: ?[]const u8,
    completion_promise: []const u8,
};

/// Run the iterative loop via the provider router, returning the final
/// concatenated output (caller owns the allocation).
fn runProviderLoop(allocator: std.mem.Allocator, opts: ProviderLoopOpts) ![]u8 {
    var output_buf = std.ArrayListUnmanaged(u8).empty;
    defer output_buf.deinit(allocator);

    var iter: usize = 0;
    while (iter < opts.max_iterations) : (iter += 1) {
        std.debug.print("--- iteration {d}/{d} ---\n", .{ iter + 1, opts.max_iterations });

        var result = providers.generate(allocator, .{
            .model = opts.model,
            .prompt = opts.goal,
            .backend = opts.backend,
            .fallback = opts.fallback,
            .strict_backend = opts.strict_backend,
            .plugin_id = opts.plugin,
            .max_tokens = 1200,
            .temperature = 0.7,
        }) catch |err| {
            std.debug.print("Provider generate failed: {t}\n", .{err});
            return error.ProviderGenerateFailed;
        };
        defer result.deinit(allocator);

        std.debug.print("[{s}] ", .{result.provider.label()});
        std.debug.print("{s}\n\n", .{result.content});

        try output_buf.appendSlice(allocator, result.content);
        try output_buf.append(allocator, '\n');

        // Check for completion promise
        if (cfg.containsIgnoreCase(result.content, opts.completion_promise)) {
            std.debug.print("Completion promise \"{s}\" found — stopping.\n", .{opts.completion_promise});
            break;
        }
    }

    std.debug.print("--- provider loop finished ({d} iterations) ---\n", .{iter});
    return try output_buf.toOwnedSlice(allocator);
}

/// Handle skill persistence for provider-routed mode (no engine).
fn handleSkills(
    allocator: std.mem.Allocator,
    io: std.Io,
    store_skill: ?[]const u8,
    auto_skill: bool,
    _: []const u8,
    result: []const u8,
) u64 {
    _ = auto_skill; // auto-skill extraction requires Abbey engine
    var skills_added: u64 = 0;
    if (store_skill) |s| {
        skills_store.appendSkill(allocator, io, s, null, 1.0) catch |err| {
            std.debug.print("Warning: could not persist skill: {t}\n", .{err});
        };
        skills_added += 1;
        std.debug.print("Skill stored.\n", .{});
    }
    _ = result;
    return skills_added;
}

/// Update state.json with run count and skill count.
fn updateState(allocator: std.mem.Allocator, io: std.Io, skills_added: u64) void {
    var state = cfg.readState(allocator, io);
    state.runs += 1;
    state.skills += skills_added;
    state.last_run_ts = blk: {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);
        break :blk @intCast(ts.sec);
    };
    cfg.writeState(allocator, io, state);
}

fn appendProvidersCsv(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(ProviderId),
    csv: []const u8,
) !void {
    var split = std.mem.splitScalar(u8, csv, ',');
    while (split.next()) |raw_part| {
        const part = std.mem.trim(u8, raw_part, " \t\r\n");
        if (part.len == 0) continue;
        const provider = ProviderId.fromString(part) orelse continue;
        var already = false;
        for (out.items) |existing| {
            if (existing == provider) {
                already = true;
                break;
            }
        }
        if (already) continue;
        try out.append(allocator, provider);
    }
}

fn printHelp() void {
    std.debug.print(
        \\Usage: abi ralph run [options]
        \\
        \\Execute the Ralph iterative loop via the Abbey engine or a provider backend.
        \\
        \\Options:
        \\  -t, --task <text>        Override task (default: reads PROMPT.md)
        \\  -i, --iterations <n>     Override max_iterations from ralph.yml
        \\  -c, --config <path>      Config file (default: ralph.yml)
        \\      --auto-skill         Extract and store a skill after the run
        \\      --store-skill <s>    Manually store a skill string after run
        \\      --backend <id>       Provider backend id (llama_cpp, mlx, ollama, lm_studio, vllm, anthropic, openai, plugin_http, plugin_native)
        \\      --fallback <csv>     Fallback provider chain (comma-separated)
        \\      --strict-backend     Disable fallback and fail fast on selected backend
        \\      --model <id|path>    Model id or local path
        \\      --plugin <id>        Plugin id for plugin providers
        \\  -h, --help               Show this help
        \\
    , .{});
}

fn firstSentence(text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return null;
    for (trimmed, 0..) |ch, idx| {
        if (ch == '.' or ch == '\n') return std.mem.trim(u8, trimmed[0 .. idx + 1], " \t\r\n");
    }
    return trimmed;
}
