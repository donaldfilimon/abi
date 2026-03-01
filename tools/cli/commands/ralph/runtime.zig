//! Ralph autonomous improve runtime.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const providers = abi.features.ai.llm.providers;
const cfg = @import("config.zig");
const workspace = @import("workspace.zig");
const git_ops = @import("git_ops.zig");
const verification = @import("verification.zig");
const artifacts = @import("artifacts.zig");
const skills_store = @import("skills_store.zig");
const workflow_contract = @import("workflow_contract.zig");

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
    workflow_enable_contract: bool = true,
    workflow_todo_file: []const u8 = "tasks/todo.md",
    workflow_lessons_file: []const u8 = "tasks/lessons.md",
    workflow_strict_contract: bool = false,
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
        utils.output.printError("Refusing autonomous apply on dirty tree. Commit or stash changes, or use --analysis-only.", .{});
        return error.DirtyTree;
    }

    if (!options.analysis_only and in_repo and options.require_clean_tree and !git_ops.hasChanges(allocator, io, options.worktree)) {
        const branch = try std.fmt.allocPrint(allocator, "codex/ralph-{s}", .{run_id});
        defer allocator.free(branch);
        git_ops.ensureRunBranch(allocator, io, options.worktree, branch);
    }

    var tool_agent = try abi.features.ai.tool_agent.ToolAugmentedAgent.init(allocator, .{
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

    const contract_check = if (options.workflow_enable_contract)
        workflow_contract.inspectContractFiles(io, options.workflow_todo_file, options.workflow_lessons_file)
    else
        workflow_contract.ContractCheck{};
    var workflow_metrics = workflow_contract.RuntimeMetrics{
        .workflow_contract_passed = contract_check.passed,
        .workflow_warning_count = contract_check.warning_count,
    };
    if (options.workflow_enable_contract and options.workflow_strict_contract and !contract_check.passed) {
        utils.output.printError("Workflow contract strict mode failed (todo/lessons files not ready).", .{});
        return error.WorkflowContractFailed;
    }
    const contract_prompt = if (options.workflow_enable_contract) workflow_contract.contractPrompt() else "";

    var prompt = try std.fmt.allocPrint(
        allocator,
        "Task:\n{s}\n\nConstraints:\n- Work inside {s}\n- Prefer minimal, correct edits\n- Run and fix {s}\n- Use tools directly to edit files and execute commands\n{s}\n{s}",
        .{ options.task, options.worktree, options.gate_command, skills_context, contract_prompt },
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
        var verify_attempts: usize = 0;
        var verify_command_rejected = false;
        var correction_applied = false;
        var lessons_appended_iter: usize = 0;
        var trigger_kind: ?workflow_contract.ReplanTrigger = null;
        var trigger_label: ?[]const u8 = null;
        var trigger_logged = false;
        // Ensure verify_result_opt is cleaned up if we exit the iteration early via try.
        errdefer if (verify_result_opt) |*vr| vr.deinit(allocator);

        if (!options.analysis_only) {
            verify_result_opt = try verification.runGateCommand(allocator, io, options.worktree, options.gate_command);
            verify_attempts += 1;
            verify_command_rejected = workflow_contract.isVerifyCommandRejected(verify_result_opt.?.stderr);
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
                if (options.workflow_enable_contract) {
                    const trigger = workflow_contract.classifyTrigger(false, verify_command_rejected, false) orelse .verification_fail;
                    trigger_kind = trigger;
                    trigger_label = trigger.label();
                    const impact = try std.fmt.allocPrint(
                        allocator,
                        "Gate command '{s}' failed with exit={d}.",
                        .{ options.gate_command, verify_result_opt.?.exit_code },
                    );
                    defer allocator.free(impact);
                    const wrote_replan = workflow_contract.appendReplanNote(allocator, io, options.workflow_todo_file, .{
                        .trigger = trigger,
                        .impact = impact,
                        .plan_change = "Stop current path and switch to explicit fix-attempt loop.",
                        .verification_change = "Rerun the gate command after each fix attempt and log the result.",
                    }) catch false;
                    if (wrote_replan) {
                        trigger_logged = true;
                        workflow_metrics.replan_trigger_count += 1;
                    }
                }

                var fix_attempt: usize = 0;
                while (fix_attempt < options.max_fix_attempts and !last_gate_passed) : (fix_attempt += 1) {
                    const fail_prompt = try std.fmt.allocPrint(
                        allocator,
                        "{s} failed (exit {d}). Apply direct code fixes and rerun.\n\nstdout:\n{s}\n\nstderr:\n{s}\n\n{s}",
                        .{ options.gate_command, verify_result_opt.?.exit_code, verify_result_opt.?.stdout, verify_result_opt.?.stderr, contract_prompt },
                    );
                    defer allocator.free(fail_prompt);
                    const fix_response = tool_agent.processWithTools(fail_prompt, allocator) catch |err| {
                        if (options.workflow_enable_contract and trigger_kind == null) {
                            trigger_kind = workflow_contract.classifyTrigger(false, false, true) orelse .blocked_step;
                            trigger_label = trigger_kind.?.label();
                            const wrote_replan = workflow_contract.appendReplanNote(allocator, io, options.workflow_todo_file, .{
                                .trigger = trigger_kind.?,
                                .impact = "A tool execution failed during automatic fix attempt.",
                                .plan_change = "Pause current fix attempt and re-plan from current repository state.",
                                .verification_change = "Resume gate checks after the blocked step is resolved.",
                            }) catch false;
                            if (wrote_replan) {
                                trigger_logged = true;
                                workflow_metrics.replan_trigger_count += 1;
                            }
                        }
                        utils.output.printWarning("Fix attempt tool run failed: {t}", .{err});
                        break;
                    };
                    defer allocator.free(fix_response);

                    verify_result_opt.?.deinit(allocator);
                    verify_result_opt = null;
                    verify_result_opt = try verification.runGateCommand(allocator, io, options.worktree, options.gate_command);
                    verify_attempts += 1;
                    verify_command_rejected = verify_command_rejected or workflow_contract.isVerifyCommandRejected(verify_result_opt.?.stderr);
                    last_gate_passed = verify_result_opt.?.passed;
                    last_gate_exit = verify_result_opt.?.exit_code;
                    changed = in_repo and git_ops.hasChanges(allocator, io, options.worktree);

                    if (last_gate_passed and in_repo and changed) {
                        correction_applied = true;
                        workflow_metrics.correction_count += 1;
                        if (options.workflow_enable_contract) {
                            const root_cause = try std.fmt.allocPrint(
                                allocator,
                                "Gate '{s}' failed and required an automated correction cycle in iteration {d}.",
                                .{ options.gate_command, iteration + 1 },
                            );
                            defer allocator.free(root_cause);
                            const lesson_written = workflow_contract.appendCorrectionLesson(
                                allocator,
                                io,
                                options.workflow_lessons_file,
                                trigger_kind orelse .verification_fail,
                                root_cause,
                                "When verification fails, log a re-plan note and rerun the gate after each fix before continuing.",
                            ) catch false;
                            if (lesson_written) {
                                lessons_appended_iter += 1;
                                workflow_metrics.lessons_appended += 1;
                            }
                        }

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
            .{
                .contract_prompt_injected = options.workflow_enable_contract,
                .warning_count = workflow_metrics.workflow_warning_count,
                .trigger = trigger_label,
                .trigger_logged = trigger_logged,
                .correction_applied = correction_applied,
                .lessons_appended = lessons_appended_iter,
                .verify_attempts = verify_attempts,
                .verify_command_rejected = verify_command_rejected,
            },
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
            "Iteration {d} complete (gate_passed={s}, changed={s}). Continue improving for task:\n{s}\n\n{s}",
            .{
                iteration + 1,
                if (last_gate_passed) "true" else "false",
                if (changed) "true" else "false",
                options.task,
                contract_prompt,
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
        .workflow_contract_passed = workflow_metrics.workflow_contract_passed,
        .workflow_warning_count = workflow_metrics.workflow_warning_count,
        .replan_trigger_count = workflow_metrics.replan_trigger_count,
        .correction_count = workflow_metrics.correction_count,
        .lessons_appended = workflow_metrics.lessons_appended,
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

test {
    std.testing.refAllDecls(@This());
}
