//! Ralph autonomous improve runtime.

const std = @import("std");
const abi = @import("abi");
const providers = abi.ai.llm.providers;
const cfg = @import("config.zig");
const workspace = @import("workspace.zig");
const git_ops = @import("git_ops.zig");
const verification = @import("verification.zig");
const artifacts = @import("artifacts.zig");
const skills_store = @import("skills_store.zig");

pub const ImproveOptions = struct {
    task: []const u8,
    analysis_only: bool = false,
    max_iterations: usize = 5,
    max_fix_attempts: usize = 2,
    backend: ?providers.ProviderId = null,
    fallback: []const providers.ProviderId = &.{},
    strict_backend: bool = false,
    model: []const u8 = "llama3.2",
    plugin: ?[]const u8 = null,
    worktree: []const u8 = ".",
    require_clean_tree: bool = true,
    /// Gate command from ralph.yml (default: "zig build verify-all").
    gate_command: []const u8 = "zig build verify-all",
};

pub const RunSummary = struct {
    run_id: []u8,
    iterations: usize,
    passing_iterations: usize,
    last_gate_passed: bool,
    last_gate_exit: u8,
};

pub fn runImprove(
    allocator: std.mem.Allocator,
    io: std.Io,
    options: ImproveOptions,
) !RunSummary {
    const run_id = try workspace.generateRunId(allocator);
    errdefer allocator.free(run_id);
    const started_at = workspace.nowEpochSeconds();

    try workspace.ensureRunLayout(allocator, io, run_id);
    try workspace.acquireLoopLock(allocator, io, run_id, 60 * 60 * 4);
    defer workspace.releaseLoopLock(io);

    const in_repo = git_ops.isGitRepo(allocator, io, options.worktree);
    if (!options.analysis_only and options.require_clean_tree and in_repo and git_ops.hasChanges(allocator, io, options.worktree)) {
        std.debug.print("Refusing autonomous apply on dirty tree. Commit or stash changes, or use --analysis-only.\n", .{});
        return error.DirtyTree;
    }

    if (!options.analysis_only and in_repo and options.require_clean_tree and !git_ops.hasChanges(allocator, io, options.worktree)) {
        const branch = try std.fmt.allocPrint(allocator, "codex/ralph-{s}", .{run_id});
        defer allocator.free(branch);
        git_ops.ensureRunBranch(allocator, io, options.worktree, branch);
    }

    var tool_agent = try abi.ai.tool_agent.ToolAugmentedAgent.init(allocator, .{
        .agent = .{
            .name = "ralph-autonomous",
            .backend = .provider_router,
            .model = options.model,
            .temperature = 0.2,
            .max_tokens = 1200,
            .provider_backend = options.backend,
            .provider_fallback = options.fallback,
            .provider_strict_backend = options.strict_backend,
            .provider_plugin_id = options.plugin,
        },
        .max_tool_iterations = 14,
        .require_confirmation = false,
        .working_directory = options.worktree,
    });
    defer tool_agent.deinit();
    try tool_agent.registerAllAgentTools();

    const skills_context = try skills_store.loadSkillsContext(allocator, io, 2048);
    defer allocator.free(skills_context);

    var prompt = try std.fmt.allocPrint(
        allocator,
        "Task:\n{s}\n\nConstraints:\n- Work inside {s}\n- Prefer minimal, correct edits\n- Run and fix zig build verify-all\n- Use tools directly to edit files and execute commands\n{s}",
        .{ options.task, options.worktree, skills_context },
    );
    defer allocator.free(prompt);

    var passing_iterations: usize = 0;
    var last_gate_passed = false;
    var last_gate_exit: u8 = 1;
    var iterations_run: usize = 0;
    var skills_added: u64 = 0;
    var verify_log_path = try allocator.dupe(u8, "");
    defer allocator.free(verify_log_path);

    var iteration: usize = 0;
    while (iteration < options.max_iterations) : (iteration += 1) {
        iterations_run += 1;
        const response = tool_agent.processWithTools(prompt, allocator) catch |err| try std.fmt.allocPrint(
            allocator,
            "Tool-augmented iteration failed: {t}",
            .{err},
        );
        defer allocator.free(response);

        var changed = in_repo and git_ops.hasChanges(allocator, io, options.worktree);
        var committed = false;
        var verify_result_opt: ?verification.VerifyResult = null;
        // Ensure verify_result_opt is cleaned up if we exit the iteration early via try.
        errdefer if (verify_result_opt) |*vr| vr.deinit(allocator);

        if (!options.analysis_only) {
            verify_result_opt = try verification.runGateCommand(allocator, io, options.worktree, options.gate_command);
            last_gate_passed = verify_result_opt.?.passed;
            last_gate_exit = verify_result_opt.?.exit_code;

            if (verify_result_opt.?.passed) {
                passing_iterations += 1;
                if (in_repo and changed) {
                    committed = git_ops.commitAllIfChanged(
                        allocator,
                        io,
                        options.worktree,
                        run_id,
                        iteration + 1,
                        true,
                        providerLabel(options),
                    );
                    changed = in_repo and git_ops.hasChanges(allocator, io, options.worktree);
                }
            } else {
                var fix_attempt: usize = 0;
                while (fix_attempt < options.max_fix_attempts and !last_gate_passed) : (fix_attempt += 1) {
                    const fail_prompt = try std.fmt.allocPrint(
                        allocator,
                        "{s} failed (exit {d}). Apply direct code fixes and rerun.\n\nstdout:\n{s}\n\nstderr:\n{s}",
                        .{ options.gate_command, verify_result_opt.?.exit_code, verify_result_opt.?.stdout, verify_result_opt.?.stderr },
                    );
                    defer allocator.free(fail_prompt);
                    const fix_response = tool_agent.processWithTools(fail_prompt, allocator) catch break;
                    defer allocator.free(fix_response);

                    verify_result_opt.?.deinit(allocator);
                    verify_result_opt = null;
                    verify_result_opt = try verification.runGateCommand(allocator, io, options.worktree, options.gate_command);
                    last_gate_passed = verify_result_opt.?.passed;
                    last_gate_exit = verify_result_opt.?.exit_code;
                    changed = in_repo and git_ops.hasChanges(allocator, io, options.worktree);

                    if (last_gate_passed and in_repo and changed) {
                        committed = git_ops.commitAllIfChanged(
                            allocator,
                            io,
                            options.worktree,
                            run_id,
                            iteration + 1,
                            true,
                            providerLabel(options),
                        );
                        changed = in_repo and git_ops.hasChanges(allocator, io, options.worktree);
                        passing_iterations += 1;
                        break;
                    }
                }
            }
        } else {
            last_gate_passed = true;
            last_gate_exit = 0;
        }

        if (extractSkillSentence(response)) |sentence| {
            skills_store.appendSkill(allocator, io, sentence, run_id, if (last_gate_passed) 1.0 else 0.5) catch {};
            skills_added += 1;
        }

        try artifacts.writeIterationArtifact(
            allocator,
            io,
            run_id,
            iteration + 1,
            options.task,
            response,
            verify_result_opt,
            changed,
            committed,
        );

        if (verify_result_opt) |verify_result| {
            const new_log = try artifacts.appendVerifyLog(
                allocator,
                io,
                run_id,
                iteration + 1,
                verify_result,
            );
            allocator.free(verify_log_path);
            verify_log_path = new_log;
            verify_result_opt.?.deinit(allocator);
            verify_result_opt = null;
        }

        if (!options.analysis_only and last_gate_passed and !changed) break;

        const next_prompt = try std.fmt.allocPrint(
            allocator,
            "Iteration {d} complete (gate_passed={s}, changed={s}). Continue improving for task:\n{s}",
            .{
                iteration + 1,
                if (last_gate_passed) "true" else "false",
                if (changed) "true" else "false",
                options.task,
            },
        );
        allocator.free(prompt);
        prompt = next_prompt;
    }

    const ended_at = workspace.nowEpochSeconds();
    const fallback_text = try fallbackLabel(allocator, options);
    defer allocator.free(fallback_text);

    try artifacts.writeReport(allocator, io, .{
        .run_id = run_id,
        .worktree = options.worktree,
        .backend = providerLabel(options),
        .fallback = fallback_text,
        .strict_backend = options.strict_backend,
        .model = options.model,
        .plugin = options.plugin,
        .iterations = iterations_run,
        .passing_iterations = passing_iterations,
        .last_gate_passed = last_gate_passed,
        .last_gate_exit = last_gate_exit,
        .started_at = started_at,
        .ended_at = ended_at,
        .verify_log = verify_log_path,
        .duration_seconds = ended_at - started_at,
        .skills_added = skills_added,
        .gate_command = options.gate_command,
    });

    var state = cfg.readState(allocator, io);
    state.runs += 1;
    state.skills += skills_added;
    state.last_run_ts = ended_at;
    state.last_run_id = run_id;
    state.last_gate_passed = last_gate_passed;
    cfg.writeState(allocator, io, state);

    return .{
        .run_id = run_id,
        .iterations = iterations_run,
        .passing_iterations = passing_iterations,
        .last_gate_passed = last_gate_passed,
        .last_gate_exit = last_gate_exit,
    };
}

fn providerLabel(options: ImproveOptions) []const u8 {
    if (options.backend) |b| return b.label();
    return "auto(local-first)";
}

fn fallbackLabel(allocator: std.mem.Allocator, options: ImproveOptions) ![]u8 {
    if (options.fallback.len == 0) return allocator.dupe(u8, "");
    var buffer = std.ArrayListUnmanaged(u8).empty;
    defer buffer.deinit(allocator);
    for (options.fallback, 0..) |provider, idx| {
        if (idx > 0) try buffer.append(allocator, ',');
        try buffer.appendSlice(allocator, provider.label());
    }
    return buffer.toOwnedSlice(allocator);
}

/// Extract the most meaningful skill-worthy sentence from a response.
/// Prefers sentences containing action verbs and technical terms over generic openers.
fn extractSkillSentence(text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return null;

    // Collect up to 5 sentences
    var sentences: [5][]const u8 = undefined;
    var sentence_count: usize = 0;
    var start: usize = 0;

    for (trimmed, 0..) |ch, idx| {
        if (ch == '.' or ch == '\n') {
            const candidate = std.mem.trim(u8, trimmed[start .. idx + 1], " \t\r\n");
            if (candidate.len >= 15 and sentence_count < 5) {
                sentences[sentence_count] = candidate;
                sentence_count += 1;
            }
            start = idx + 1;
        }
    }

    // Fallback: if no sentences found, use first sentence
    if (sentence_count == 0) {
        for (trimmed, 0..) |ch, idx| {
            if (ch == '.' or ch == '\n') return std.mem.trim(u8, trimmed[0 .. idx + 1], " \t\r\n");
        }
        return if (trimmed.len >= 10) trimmed else null;
    }

    // Score sentences: prefer those with actionable technical content
    const action_keywords = [_][]const u8{
        "use",  "avoid",  "always",  "never",  "prefer", "ensure", "require",
        "must", "should", "instead", "rather", "fix",    "update", "replace",
    };
    const technical_keywords = [_][]const u8{
        "zig",    "build",  "test",   "compile", "allocat", "io",
        "mod",    "stub",   "import", "feature", "error",   "memory",
        "config", "module", "gate",   "flag",
    };

    var best_idx: usize = 0;
    var best_score: usize = 0;

    for (sentences[0..sentence_count], 0..) |sentence, idx| {
        var score: usize = 0;
        for (action_keywords) |kw| {
            if (std.mem.indexOf(u8, sentence, kw) != null) score += 2;
        }
        for (technical_keywords) |kw| {
            if (std.mem.indexOf(u8, sentence, kw) != null) score += 1;
        }
        // Penalize very short or very long
        if (sentence.len < 30) score = score / 2;
        if (sentence.len > 200) score = score / 2;

        if (score > best_score) {
            best_score = score;
            best_idx = idx;
        }
    }

    return sentences[best_idx];
}
