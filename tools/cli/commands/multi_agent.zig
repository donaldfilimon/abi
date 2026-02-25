//! Multi‑Agent CLI command
//!
//! Provides an interface to the `MultiAgentCoordinator` API for running
//! multi-agent workflows, including DAG-based workflow execution.
//!
//! Subcommands:
//!   * `info`         - Show coordinator status and feature gating
//!   * `run`          - Execute a task using registered agents
//!   * `run-workflow`  - Execute a preset DAG workflow (code-review, research, implement-feature)
//!   * `list`         - List available workflow templates with DAG details
//!   * `create`       - Create a new workflow from template
//!   * `status`       - Show current coordinator status

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const app_paths = abi.shared.app_paths;

// Access the workflow DAG engine from the multi_agent module
const workflow_mod = abi.ai.multi_agent.workflow;
const WorkflowDef = workflow_mod.WorkflowDef;
const ExecutionTracker = workflow_mod.ExecutionTracker;
const StepStatus = workflow_mod.StepStatus;
const StepResult = workflow_mod.StepResult;

/// Predefined workflow templates
const WorkflowTemplate = struct {
    name: []const u8,
    description: []const u8,
    agents: []const []const u8,
};

const workflow_templates = [_]WorkflowTemplate{
    .{
        .name = "code-review",
        .description = "Multi-perspective code review workflow",
        .agents = &[_][]const u8{ "reviewer", "security", "performance" },
    },
    .{
        .name = "refactor",
        .description = "Coordinated refactoring workflow",
        .agents = &[_][]const u8{ "analyzer", "planner", "implementer" },
    },
    .{
        .name = "documentation",
        .description = "Documentation generation workflow",
        .agents = &[_][]const u8{ "analyzer", "writer", "reviewer" },
    },
    .{
        .name = "testing",
        .description = "Test generation and validation workflow",
        .agents = &[_][]const u8{ "analyzer", "test-writer", "validator" },
    },
};

/// DAG workflow presets — maps CLI names to WorkflowDef constants.
const DagPreset = struct {
    name: []const u8,
    def: WorkflowDef,
};

const dag_presets = [_]DagPreset{
    .{ .name = "code-review", .def = workflow_mod.preset_workflows.code_review },
    .{ .name = "research", .def = workflow_mod.preset_workflows.research },
    .{ .name = "implement-feature", .def = workflow_mod.preset_workflows.implement_feature },
};

fn findDagPreset(name: []const u8) ?WorkflowDef {
    for (dag_presets) |preset| {
        if (std.mem.eql(u8, preset.name, name)) return preset.def;
    }
    return null;
}

// Wrapper functions for comptime children dispatch
fn wrapMaInfo(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try runInfo(allocator);
}
fn wrapMaRun(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = utils.args.ArgParser.init(allocator, args);
    try runWorkflow(allocator, &parser);
}
fn wrapMaDagRun(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = utils.args.ArgParser.init(allocator, args);
    try runDagWorkflow(allocator, &parser);
}
fn wrapMaList(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try listWorkflows(allocator);
}
fn wrapMaCreate(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = utils.args.ArgParser.init(allocator, args);
    try createWorkflow(allocator, &parser);
}
fn wrapMaStatus(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    try showStatus(allocator);
}

fn findTemplate(name: []const u8) ?WorkflowTemplate {
    for (workflow_templates) |tmpl| {
        if (std.mem.eql(u8, tmpl.name, name)) return tmpl;
    }
    return null;
}

pub const meta: command_mod.Meta = .{
    .name = "multi-agent",
    .description = "Run multi-agent workflows",
    .subcommands = &.{ "info", "run", "run-workflow", "list", "create", "status" },
    .children = &.{
        .{ .name = "info", .description = "Show coordinator status and capabilities", .handler = wrapMaInfo },
        .{ .name = "run", .description = "Execute a task using the coordinator", .handler = wrapMaRun },
        .{ .name = "run-workflow", .description = "Execute a preset DAG workflow", .handler = wrapMaDagRun },
        .{ .name = "list", .description = "List available workflow templates", .handler = wrapMaList },
        .{ .name = "create", .description = "Create a new workflow configuration", .handler = wrapMaCreate },
        .{ .name = "status", .description = "Show current coordinator status", .handler = wrapMaStatus },
    },
};

const ma_subcommands = [_][]const u8{
    "info", "run", "run-workflow", "list", "create", "status", "help",
};

/// Entry point for the `multi-agent` command.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        printHelp(allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp(allocator);
        return;
    }
    utils.output.printError("Unknown multi-agent subcommand: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &ma_subcommands)) |suggestion| {
        utils.output.printInfo("Did you mean: {s}", .{suggestion});
    }
    utils.output.printInfo("Run 'abi multi-agent help' for usage.", .{});
}

fn runInfo(allocator: std.mem.Allocator) !void {
    // Initialise the framework to access runtime feature matrix.
    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    utils.output.printHeader("Multi‑Agent Coordinator");

    // Feature gating – only meaningful when AI is enabled.
    const ai_enabled = framework.isEnabled(.ai);
    utils.output.printKeyValue("AI Feature", utils.output.boolLabel(ai_enabled));

    if (!ai_enabled) {
        // When disabled, the stub returns error.AgentDisabled.
        utils.output.printInfo("Coordinator is unavailable (AI disabled)", .{});
        utils.output.printInfo("Rebuild with: zig build -Denable-ai=true", .{});
        return;
    }

    // Use the public API via `abi.ai.multi_agent`.
    const Coordinator = abi.ai.multi_agent.Coordinator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    utils.output.printKeyValueFmt("Registered Agents", "{d}", .{coord.agentCount()});

    // Show available templates
    utils.output.printKeyValueFmt("Workflow Templates", "{d}", .{workflow_templates.len});
    utils.output.printKeyValueFmt("DAG Workflow Presets", "{d}", .{dag_presets.len});

    utils.output.printSuccess("Coordinator ready.", .{});
}

fn runWorkflow(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.matches(&[_][]const u8{ "--help", "-h", "help" }) and parser.remaining().len == 1) {
        printRunHelp();
        return;
    }

    var workflow_name: ?[]const u8 = null;
    var task_description: ?[]const u8 = null;
    var unexpected_arg: ?[]const u8 = null;

    // Parse arguments
    while (parser.hasMore()) {
        if (parser.consumeFlag(&[_][]const u8{ "--help", "-h", "help" })) {
            printRunHelp();
            return;
        } else if (parser.consumeOption(&[_][]const u8{ "--workflow", "-w" })) |val| {
            workflow_name = val;
        } else if (parser.consumeOption(&[_][]const u8{ "--task", "-t" })) |val| {
            task_description = val;
        } else {
            // Positional argument is the task
            const positional = parser.next().?;
            if (task_description == null) {
                task_description = positional;
            } else if (unexpected_arg == null) {
                unexpected_arg = positional;
            }
        }
    }

    if (unexpected_arg) |arg| {
        utils.output.printError("Unexpected argument: {s}", .{arg});
        printRunHelp();
        return;
    }

    if (task_description == null) {
        utils.output.printError("No task specified", .{});
        printRunHelp();
        return;
    }

    // Check AI feature
    if (!abi.ai.isEnabled()) {
        utils.output.printError("AI feature is disabled", .{});
        utils.output.printInfo("Rebuild with: zig build -Denable-ai=true", .{});
        return;
    }

    utils.output.printHeader("Multi-Agent Workflow");

    const selected_template: ?WorkflowTemplate = blk: {
        if (workflow_name) |wf| {
            const tmpl = findTemplate(wf) orelse {
                utils.output.printError("Unknown workflow template: {s}", .{wf});
                std.debug.print("Available templates: ", .{});
                for (workflow_templates, 0..) |item, i| {
                    if (i > 0) std.debug.print(", ", .{});
                    std.debug.print("{s}", .{item.name});
                }
                std.debug.print("\n", .{});
                return;
            };
            break :blk tmpl;
        }
        break :blk null;
    };

    // Show workflow info
    if (selected_template) |tmpl| {
        utils.output.printKeyValue("Workflow", tmpl.name);
        utils.output.printInfo("Using template: {s}", .{tmpl.description});
        std.debug.print("Agents: ", .{});
        for (tmpl.agents, 0..) |agent, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{s}", .{agent});
        }
        std.debug.print("\n", .{});
    } else {
        utils.output.printKeyValue("Workflow", "default (sequential)");
    }

    utils.output.printKeyValue("Task", task_description.?);

    std.debug.print("\n", .{});

    // Initialize coordinator
    const Coordinator = abi.ai.multi_agent.Coordinator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    // Determine agent names from template or defaults
    const agent_names: []const []const u8 = if (selected_template) |tmpl|
        tmpl.agents
    else
        &[_][]const u8{ "agent-1", "agent-2", "agent-3" };

    // Create echo agents (safe default — no API keys required)
    const AgentType = abi.ai.agent.Agent;
    var agent_storage = allocator.alloc(AgentType, agent_names.len) catch {
        utils.output.printError("Failed to allocate agents", .{});
        return;
    };
    defer {
        for (agent_storage) |*ag| ag.deinit();
        allocator.free(agent_storage);
    }

    for (agent_names, 0..) |name, idx| {
        agent_storage[idx] = AgentType.init(allocator, .{
            .name = name,
            .backend = .echo,
        }) catch {
            utils.output.printError("Failed to create agent: {s}", .{name});
            return;
        };
        coord.register(&agent_storage[idx]) catch {
            utils.output.printError("Failed to register agent: {s}", .{name});
            return;
        };
    }

    utils.output.printKeyValueFmt("Agents", "{d} registered", .{coord.agentCount()});
    utils.output.printInfo("Executing workflow...", .{});

    const result = coord.runTask(task_description.?) catch |err| {
        utils.output.printError("Workflow execution failed: {t}", .{err});
        return;
    };
    defer allocator.free(result);

    std.debug.print("\n[Workflow Output]\n", .{});
    std.debug.print("{s}\n", .{result});

    // Show stats
    const stats = coord.getStats();
    utils.output.printKeyValueFmt("Results", "{d} total, {d} successful", .{ stats.result_count, stats.success_count });

    utils.output.printSuccess("Workflow complete", .{});
}

// ============================================================================
// DAG Workflow Execution
// ============================================================================

/// Execute a preset DAG workflow by name (run-workflow subcommand).
fn runDagWorkflow(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.matches(&[_][]const u8{ "--help", "-h", "help" }) and parser.remaining().len == 1) {
        printDagRunHelp();
        return;
    }

    var wf_name: ?[]const u8 = null;
    var task_input: ?[]const u8 = null;

    // Parse arguments: positional workflow name, optional --task
    while (parser.hasMore()) {
        if (parser.consumeFlag(&[_][]const u8{ "--help", "-h", "help" })) {
            printDagRunHelp();
            return;
        } else if (parser.consumeOption(&[_][]const u8{ "--task", "-t" })) |val| {
            task_input = val;
        } else {
            // First positional = workflow name, second = task
            const positional = parser.next().?;
            if (wf_name == null) {
                wf_name = positional;
            } else if (task_input == null) {
                task_input = positional;
            }
        }
    }

    if (wf_name == null) {
        utils.output.printError("No workflow name specified", .{});
        std.debug.print("\nAvailable DAG workflows:\n", .{});
        for (dag_presets) |preset| {
            std.debug.print("  {s:<25} {s}\n", .{ preset.name, preset.def.description });
        }
        std.debug.print("\n", .{});
        printDagRunHelp();
        return;
    }

    // Default task input
    const task_text = task_input orelse "Analyze the provided input";

    // Check AI feature
    if (!abi.ai.isEnabled()) {
        utils.output.printError("AI feature is disabled", .{});
        utils.output.printInfo("Rebuild with: zig build -Denable-ai=true", .{});
        return;
    }

    // Look up preset
    const wf_def = findDagPreset(wf_name.?) orelse {
        utils.output.printError("Unknown DAG workflow: {s}", .{wf_name.?});
        std.debug.print("\nAvailable DAG workflows:\n", .{});
        for (dag_presets) |preset| {
            std.debug.print("  {s:<25} {s}\n", .{ preset.name, preset.def.description });
        }
        return;
    };

    // Validate the workflow DAG
    const validation = wf_def.validate();
    if (!validation.valid) {
        utils.output.printError("Workflow validation failed: {s}", .{validation.error_message});
        return;
    }

    // Compute execution layers
    const layers = wf_def.computeLayers(allocator) catch |err| {
        utils.output.printError("Failed to compute workflow layers: {t}", .{err});
        return;
    };
    defer {
        for (layers) |layer| allocator.free(layer);
        allocator.free(layers);
    }

    // Print header
    std.debug.print("\n=== Workflow: {s} ===\n", .{wf_def.name});
    std.debug.print("Steps: {d} | Layers: {d}\n", .{ wf_def.steps.len, layers.len });
    std.debug.print("Task: {s}\n", .{task_text});
    std.debug.print("\n", .{});

    // Initialize coordinator and tracker
    const Coordinator = abi.ai.multi_agent.Coordinator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    // Create one echo agent per step
    const AgentType = abi.ai.agent.Agent;
    var agent_storage = allocator.alloc(AgentType, wf_def.steps.len) catch {
        utils.output.printError("Failed to allocate agents", .{});
        return;
    };
    defer {
        for (agent_storage) |*ag| ag.deinit();
        allocator.free(agent_storage);
    }

    for (wf_def.steps, 0..) |step, idx| {
        agent_storage[idx] = AgentType.init(allocator, .{
            .name = step.id,
            .backend = .echo,
        }) catch {
            utils.output.printError("Failed to create agent for step: {s}", .{step.id});
            return;
        };
        coord.register(&agent_storage[idx]) catch {
            utils.output.printError("Failed to register agent for step: {s}", .{step.id});
            return;
        };
    }

    // Initialize execution tracker
    var tracker = ExecutionTracker.init(allocator, wf_def) catch {
        utils.output.printError("Failed to initialize execution tracker", .{});
        return;
    };
    defer tracker.deinit();

    // Start overall timer
    var overall_timer = abi.shared.time.Timer.start() catch null;

    // Collect step results for final display
    var step_outputs: std.StringHashMapUnmanaged([]const u8) = .{};
    defer {
        var out_iter = step_outputs.iterator();
        while (out_iter.next()) |entry| {
            allocator.free(entry.value_ptr.*);
        }
        step_outputs.deinit(allocator);
    }

    var global_step_num: usize = 0;
    var total_failed: usize = 0;

    // Execute layer by layer
    for (layers, 0..) |layer_step_ids, layer_idx| {
        std.debug.print("[Layer {d}]\n", .{layer_idx + 1});

        for (layer_step_ids) |step_id| {
            global_step_num += 1;
            const step = wf_def.getStep(step_id) orelse continue;

            // Mark running
            tracker.markRunning(step_id);

            // Build the prompt from template + task input
            const prompt = std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ step.prompt_template, task_text }) catch {
                utils.output.printError("Failed to build prompt for step: {s}", .{step_id});
                continue;
            };
            defer allocator.free(prompt);

            // Execute via coordinator (uses first available agent with echo backend)
            var step_timer = abi.shared.time.Timer.start() catch null;
            const result_text = coord.runTask(prompt) catch |err| {
                const dur = if (step_timer) |*t| t.read() else 0;
                std.debug.print("  Step {d}/{d}: {s} ", .{ global_step_num, wf_def.steps.len, step_id });
                printDots(step_id.len);
                std.debug.print(" FAIL ({t})\n", .{err});
                tracker.markFailed(step_id, .{
                    .step_id = step_id,
                    .status = .failed,
                    .output = "",
                    .error_message = "execution failed",
                    .duration_ns = dur,
                    .assigned_persona = "",
                }) catch {};
                total_failed += 1;
                continue;
            };
            const dur = if (step_timer) |*t| t.read() else 0;

            // Save output
            step_outputs.put(allocator, step_id, result_text) catch {
                allocator.free(result_text);
                continue;
            };

            tracker.markCompleted(step_id, .{
                .step_id = step_id,
                .status = .completed,
                .output = result_text,
                .error_message = "",
                .duration_ns = dur,
                .assigned_persona = step.assigned_persona,
            }) catch {};

            // Print step status line
            std.debug.print("  Step {d}/{d}: {s} ", .{ global_step_num, wf_def.steps.len, step_id });
            printDots(step_id.len);
            std.debug.print(" OK (echo agent, {d}us)\n", .{dur / 1000});
        }

        std.debug.print("\n", .{});
    }

    // Print results section
    std.debug.print("=== Results ===\n", .{});
    for (wf_def.steps) |step| {
        std.debug.print("Step: {s}\n", .{step.id});
        if (step_outputs.get(step.id)) |output| {
            // Truncate long output for display
            const max_display: usize = 200;
            if (output.len > max_display) {
                std.debug.print("  Output: {s}...\n", .{output[0..max_display]});
            } else {
                std.debug.print("  Output: {s}\n", .{output});
            }
        } else {
            const status = tracker.getStepStatus(step.id);
            if (status) |s| {
                switch (s) {
                    .failed => std.debug.print("  Output: [FAILED]\n", .{}),
                    .skipped => std.debug.print("  Output: [SKIPPED - dependency failed]\n", .{}),
                    else => std.debug.print("  Output: [NO OUTPUT]\n", .{}),
                }
            } else {
                std.debug.print("  Output: [NO OUTPUT]\n", .{});
            }
        }
    }

    // Print stats section
    const overall_dur = if (overall_timer) |*t| t.read() else 0;
    const prog = tracker.progress();

    std.debug.print("\n=== Stats ===\n", .{});
    std.debug.print("Total steps: {d} | Completed: {d} | Failed: {d}\n", .{
        prog.total,
        prog.completed,
        prog.failed,
    });
    std.debug.print("Duration: {d}ms\n", .{overall_dur / std.time.ns_per_ms});

    if (total_failed == 0) {
        utils.output.printSuccess("DAG workflow completed successfully", .{});
    } else {
        utils.output.printError("DAG workflow completed with {d} failure(s)", .{total_failed});
    }
}

/// Print dots for alignment in step status display.
fn printDots(name_len: usize) void {
    const target_width: usize = 30;
    const dots = if (name_len < target_width) target_width - name_len else 2;
    var i: usize = 0;
    while (i < dots) : (i += 1) {
        std.debug.print(".", .{});
    }
}

fn printDagRunHelp() void {
    std.debug.print("Usage: abi multi-agent run-workflow <name> [--task \"...\"]\n", .{});
    std.debug.print("\nExecute a preset DAG workflow with dependency-ordered step execution.\n", .{});
    std.debug.print("\nAvailable workflows:\n", .{});
    std.debug.print("  code-review        Multi-perspective code review (4 steps, 2 layers)\n", .{});
    std.debug.print("  research           Research and analysis pipeline (3 steps, 3 layers)\n", .{});
    std.debug.print("  implement-feature  Feature implementation pipeline (4 steps, 4 layers)\n", .{});
    std.debug.print("\nExamples:\n", .{});
    std.debug.print("  abi multi-agent run-workflow code-review\n", .{});
    std.debug.print("  abi multi-agent run-workflow research --task \"Analyze Zig allocators\"\n", .{});
    std.debug.print("  abi multi-agent run-workflow implement-feature -t \"Add caching layer\"\n", .{});
}

// ============================================================================
// Enhanced List (with DAG details)
// ============================================================================

fn listWorkflows(allocator: std.mem.Allocator) !void {
    utils.output.printHeader("Available Workflow Templates");

    // Simple templates
    std.debug.print("\n{s}{s:<20} {s:<50}{s}\n", .{
        utils.output.Color.bold(),  "NAME", "DESCRIPTION",
        utils.output.Color.reset(),
    });
    std.debug.print("{s}\n", .{"-" ** 70});

    for (workflow_templates) |tmpl| {
        std.debug.print("{s:<20} {s:<50}\n", .{ tmpl.name, tmpl.description });
    }

    std.debug.print("\nTotal: {d} simple template(s)\n", .{workflow_templates.len});

    // DAG workflow presets
    std.debug.print("\n", .{});
    utils.output.printHeader("DAG Workflow Presets");

    for (dag_presets) |preset| {
        const wf = preset.def;

        // Compute layers for display
        const layers = wf.computeLayers(allocator) catch {
            std.debug.print("\n  {s}{s}{s} — {s}\n", .{
                utils.output.Color.cyan(), preset.name, utils.output.Color.reset(), wf.description,
            });
            std.debug.print("    (failed to compute layers)\n", .{});
            continue;
        };
        defer {
            for (layers) |layer| allocator.free(layer);
            allocator.free(layers);
        }

        std.debug.print("\n  {s}{s}{s} — {s}\n", .{
            utils.output.Color.cyan(), preset.name, utils.output.Color.reset(), wf.description,
        });
        std.debug.print("    Steps: {d} | Layers: {d}\n", .{ wf.steps.len, layers.len });

        // Show each step with deps, capabilities, and criticality
        for (wf.steps) |step| {
            const critical_marker: []const u8 = if (step.is_critical) " [critical]" else "";
            std.debug.print("    - {s}{s}\n", .{ step.id, critical_marker });
            std.debug.print("      {s}\n", .{step.description});

            // Dependencies
            if (step.depends_on.len > 0) {
                std.debug.print("      depends_on: ", .{});
                for (step.depends_on, 0..) |dep, di| {
                    if (di > 0) std.debug.print(", ", .{});
                    std.debug.print("{s}", .{dep});
                }
                std.debug.print("\n", .{});
            } else {
                std.debug.print("      depends_on: (none — root step)\n", .{});
            }

            // Capabilities
            if (step.required_capabilities.len > 0) {
                std.debug.print("      capabilities: ", .{});
                for (step.required_capabilities, 0..) |cap, ci| {
                    if (ci > 0) std.debug.print(", ", .{});
                    std.debug.print("{s}", .{@tagName(cap)});
                }
                std.debug.print("\n", .{});
            }
        }
    }

    std.debug.print("\nTotal: {d} DAG preset(s)\n", .{dag_presets.len});
    std.debug.print("\nUse 'abi multi-agent run-workflow <name>' to execute a DAG workflow\n", .{});
    std.debug.print("Use 'abi multi-agent run --workflow <name> --task \"...\"' for simple workflows\n", .{});
}

fn createWorkflow(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    if (parser.matches(&[_][]const u8{ "--help", "-h", "help" }) and parser.remaining().len == 1) {
        printCreateHelp();
        return;
    }

    const name = parser.next() orelse {
        utils.output.printError("No workflow name specified", .{});
        printCreateHelp();
        return;
    };
    if (parser.next()) |extra| {
        if (utils.args.matchesAny(extra, &.{ "--help", "-h", "help" })) {
            printCreateHelp();
            return;
        }
        utils.output.printError("Unexpected argument: {s}", .{extra});
        printCreateHelp();
        return;
    }

    utils.output.printHeader("Create Workflow");
    utils.output.printKeyValue("Name", name);

    // Check if template exists
    for (workflow_templates) |tmpl| {
        if (std.mem.eql(u8, tmpl.name, name)) {
            utils.output.printInfo("A template with this name already exists", .{});
            utils.output.printInfo("Template: {s}", .{tmpl.description});
            return;
        }
    }

    const workflows_dir = getPrimaryWorkflowsDir(allocator) catch |err| {
        utils.output.printError("Failed to resolve workflows directory: {t}", .{err});
        return;
    };
    defer allocator.free(workflows_dir);

    const workflow_file_name = std.fmt.allocPrint(allocator, "{s}.json", .{name}) catch {
        utils.output.printError("Failed to format workflow file name", .{});
        return;
    };
    defer allocator.free(workflow_file_name);
    const workflow_path = std.fs.path.join(allocator, &.{ workflows_dir, workflow_file_name }) catch {
        utils.output.printError("Failed to join workflow path", .{});
        return;
    };
    defer allocator.free(workflow_path);

    // Show creation guidance
    std.debug.print("\nTo create a custom workflow, define it in your configuration:\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("  {s}\n", .{workflow_path});
    std.debug.print("\n", .{});
    std.debug.print("Example workflow definition:\n", .{});
    std.debug.print("  {{\n", .{});
    std.debug.print("    \"name\": \"{s}\",\n", .{name});
    std.debug.print("    \"description\": \"Custom workflow\",\n", .{});
    std.debug.print("    \"agents\": [\"agent1\", \"agent2\"],\n", .{});
    std.debug.print("    \"mode\": \"sequential\"\n", .{});
    std.debug.print("  }}\n", .{});
    std.debug.print("\n", .{});

    utils.output.printSuccess("Workflow configuration guide shown", .{});
}

fn getPrimaryWorkflowsDir(allocator: std.mem.Allocator) ![]u8 {
    return app_paths.resolvePath(allocator, "workflows");
}

fn printRunHelp() void {
    std.debug.print("Usage: abi multi-agent run [--workflow <name>] --task \"...\"\n", .{});
    std.debug.print("       abi multi-agent run [--workflow <name>] \"...\"\n", .{});
}

fn printCreateHelp() void {
    std.debug.print("Usage: abi multi-agent create <name>\n", .{});
}

fn showStatus(allocator: std.mem.Allocator) !void {
    utils.output.printHeader("Multi-Agent Status");

    // Check AI feature
    const ai_enabled = abi.ai.isEnabled();
    utils.output.printKeyValue("AI Feature", utils.output.boolLabel(ai_enabled));

    if (!ai_enabled) {
        utils.output.printInfo("Multi-agent coordination requires AI feature", .{});
        utils.output.printInfo("Rebuild with: zig build -Denable-ai=true", .{});
        return;
    }

    // Initialize coordinator
    const Coordinator = abi.ai.multi_agent.Coordinator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    utils.output.printKeyValueFmt("Registered Agents", "{d}", .{coord.agentCount()});
    utils.output.printKeyValueFmt("Workflow Templates", "{d}", .{workflow_templates.len});
    utils.output.printKeyValueFmt("DAG Workflow Presets", "{d}", .{dag_presets.len});

    utils.output.printKeyValue("Status", "Ready");
    utils.output.printInfo("Use 'abi multi-agent list' to see available workflows", .{});
    utils.output.printInfo("Use 'abi multi-agent run --task \"...\"' to execute a task", .{});
    utils.output.printInfo("Use 'abi multi-agent run-workflow <name>' to run a DAG workflow", .{});
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi multi-agent <subcommand>", "[options]")
        .description("Coordinate multiple AI agents to work on complex tasks.")
        .section("Subcommands")
        .subcommand(.{ .name = "info", .description = "Show coordinator status and capabilities" })
        .subcommand(.{ .name = "run", .description = "Execute a task using the coordinator" })
        .subcommand(.{ .name = "run-workflow", .description = "Execute a preset DAG workflow" })
        .subcommand(.{ .name = "list", .description = "List available workflow templates and DAG presets" })
        .subcommand(.{ .name = "create", .description = "Create a new workflow configuration" })
        .subcommand(.{ .name = "status", .description = "Show current coordinator status" })
        .newline()
        .section("Run Options")
        .option(.{ .short = "-w", .long = "--workflow", .arg = "name", .description = "Workflow template to use" })
        .option(.{ .short = "-t", .long = "--task", .arg = "text", .description = "Task description to execute" })
        .newline()
        .section("DAG Workflows")
        .example("code-review", "Multi-perspective code review (4 steps, parallel + synthesis)")
        .example("research", "Research and analysis pipeline (3 steps, sequential)")
        .example("implement-feature", "Feature implementation (4 steps, plan → implement → test → review)")
        .newline()
        .section("Examples")
        .example("abi multi-agent info", "Show coordinator info")
        .example("abi multi-agent list", "List all workflow templates")
        .example("abi multi-agent run --task \"Review this code\"", "Run with default workflow")
        .example("abi multi-agent run -w code-review -t \"Review PR #123\"", "Run with specific template")
        .example("abi multi-agent run-workflow code-review", "Execute code-review DAG workflow")
        .example("abi multi-agent run-workflow research -t \"Zig allocators\"", "DAG workflow with custom task")
        .newline()
        .section("Options")
        .option(utils.help.common_options.help);

    builder.print();
}
