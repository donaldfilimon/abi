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
                utils.output.print("Available templates: ", .{});
                for (workflow_templates, 0..) |item, i| {
                    if (i > 0) utils.output.print(", ", .{});
                    utils.output.print("{s}", .{item.name});
                }
                utils.output.println("", .{});
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
        utils.output.print("  Agents: ", .{});
        for (tmpl.agents, 0..) |agent, i| {
            if (i > 0) utils.output.print(", ", .{});
            utils.output.print("{s}", .{agent});
        }
        utils.output.println("", .{});
    } else {
        utils.output.printKeyValue("Workflow", "default (sequential)");
    }

    utils.output.printKeyValue("Task", task_description.?);
    utils.output.println("", .{});

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

    utils.output.println("", .{});
    utils.output.printHeader("Workflow Output");
    utils.output.println("{s}", .{result});

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
        utils.output.println("\nAvailable DAG workflows:", .{});
        for (dag_presets) |preset| {
            utils.output.println("  {s:<25} {s}", .{ preset.name, preset.def.description });
        }
        utils.output.println("", .{});
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
        utils.output.println("\nAvailable DAG workflows:", .{});
        for (dag_presets) |preset| {
            utils.output.println("  {s:<25} {s}", .{ preset.name, preset.def.description });
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
    utils.output.printHeaderFmt("Workflow: {s}", .{wf_def.name});
    utils.output.printKeyValueFmt("Steps", "{d}", .{wf_def.steps.len});
    utils.output.printKeyValueFmt("Layers", "{d}", .{layers.len});
    utils.output.printKeyValue("Task", task_text);
    utils.output.println("", .{});

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
        utils.output.println("{s}[Layer {d}]{s}", .{ utils.output.Color.bold(), layer_idx + 1, utils.output.Color.reset() });

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
                utils.output.print("  Step {d}/{d}: {s} ", .{ global_step_num, wf_def.steps.len, step_id });
                printDots(step_id.len);
                utils.output.println(" {s}FAIL{s} ({t})", .{ utils.output.Color.red(), utils.output.Color.reset(), err });
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
            utils.output.print("  Step {d}/{d}: {s} ", .{ global_step_num, wf_def.steps.len, step_id });
            printDots(step_id.len);
            utils.output.println(" {s}OK{s} (echo agent, {d}us)", .{ utils.output.Color.green(), utils.output.Color.reset(), dur / 1000 });
        }

        utils.output.println("", .{});
    }

    // Print results section
    utils.output.printHeader("Results");
    for (wf_def.steps) |step| {
        utils.output.println("{s}{s}{s}:", .{ utils.output.Color.bold(), step.id, utils.output.Color.reset() });
        if (step_outputs.get(step.id)) |output| {
            // Truncate long output for display
            const max_display: usize = 200;
            if (output.len > max_display) {
                utils.output.println("  Output: {s}...", .{output[0..max_display]});
            } else {
                utils.output.println("  Output: {s}", .{output});
            }
        } else {
            const status = tracker.getStepStatus(step.id);
            if (status) |s| {
                switch (s) {
                    .failed => utils.output.println("  Output: {s}[FAILED]{s}", .{ utils.output.Color.red(), utils.output.Color.reset() }),
                    .skipped => utils.output.println("  Output: {s}[SKIPPED - dependency failed]{s}", .{ utils.output.Color.yellow(), utils.output.Color.reset() }),
                    else => utils.output.println("  Output: [NO OUTPUT]", .{}),
                }
            } else {
                utils.output.println("  Output: [NO OUTPUT]", .{});
            }
        }
    }

    // Print stats section
    const overall_dur = if (overall_timer) |*t| t.read() else 0;
    const prog = tracker.progress();

    utils.output.printHeader("Stats");
    utils.output.printKeyValueFmt("Total steps", "{d}", .{prog.total});
    utils.output.printKeyValueFmt("Completed", "{d}", .{prog.completed});
    utils.output.printKeyValueFmt("Failed", "{d}", .{prog.failed});
    utils.output.printKeyValueFmt("Duration", "{d}ms", .{overall_dur / std.time.ns_per_ms});

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
        utils.output.print(".", .{});
    }
}

fn printDagRunHelp() void {
    utils.output.println("Usage: abi multi-agent run-workflow <name> [--task \"...\"]", .{});
    utils.output.println("\nExecute a preset DAG workflow with dependency-ordered step execution.", .{});
    utils.output.println("\nAvailable workflows:", .{});
    utils.output.println("  code-review        Multi-perspective code review (4 steps, 2 layers)", .{});
    utils.output.println("  research           Research and analysis pipeline (3 steps, 3 layers)", .{});
    utils.output.println("  implement-feature  Feature implementation pipeline (4 steps, 4 layers)", .{});
    utils.output.println("\nExamples:", .{});
    utils.output.println("  abi multi-agent run-workflow code-review", .{});
    utils.output.println("  abi multi-agent run-workflow research --task \"Analyze Zig allocators\"", .{});
    utils.output.println("  abi multi-agent run-workflow implement-feature -t \"Add caching layer\"", .{});
}

// ============================================================================
// Enhanced List (with DAG details)
// ============================================================================

fn listWorkflows(allocator: std.mem.Allocator) !void {
    utils.output.printHeader("Available Workflow Templates");

    // Simple templates
    utils.output.println("\n{s}{s:<20} {s:<50}{s}", .{
        utils.output.Color.bold(),  "NAME", "DESCRIPTION",
        utils.output.Color.reset(),
    });
    utils.output.println("{s}", .{"-" ** 70});

    for (workflow_templates) |tmpl| {
        utils.output.println("{s:<20} {s:<50}", .{ tmpl.name, tmpl.description });
    }

    utils.output.println("\nTotal: {d} simple template(s)", .{workflow_templates.len});

    // DAG workflow presets
    utils.output.println("", .{});
    utils.output.printHeader("DAG Workflow Presets");

    for (dag_presets) |preset| {
        const wf = preset.def;

        // Compute layers for display
        const layers = wf.computeLayers(allocator) catch {
            utils.output.println("\n  {s}{s}{s} — {s}", .{
                utils.output.Color.cyan(), preset.name, utils.output.Color.reset(), wf.description,
            });
            utils.output.println("    (failed to compute layers)", .{});
            continue;
        };
        defer {
            for (layers) |layer| allocator.free(layer);
            allocator.free(layers);
        }

        utils.output.println("\n  {s}{s}{s} — {s}", .{
            utils.output.Color.cyan(), preset.name, utils.output.Color.reset(), wf.description,
        });
        utils.output.println("    Steps: {d} | Layers: {d}", .{ wf.steps.len, layers.len });

        // Show each step with deps, capabilities, and criticality
        for (wf.steps) |step| {
            const critical_marker: []const u8 = if (step.is_critical) " [critical]" else "";
            utils.output.println("    - {s}{s}", .{ step.id, critical_marker });
            utils.output.println("      {s}", .{step.description});

            // Dependencies
            if (step.depends_on.len > 0) {
                utils.output.print("      depends_on: ", .{});
                for (step.depends_on, 0..) |dep, di| {
                    if (di > 0) utils.output.print(", ", .{});
                    utils.output.print("{s}", .{dep});
                }
                utils.output.println("", .{});
            } else {
                utils.output.println("      depends_on: (none — root step)", .{});
            }

            // Capabilities
            if (step.required_capabilities.len > 0) {
                utils.output.print("      capabilities: ", .{});
                for (step.required_capabilities, 0..) |cap, ci| {
                    if (ci > 0) utils.output.print(", ", .{});
                    utils.output.print("{t}", .{cap});
                }
                utils.output.println("", .{});
            }
        }
    }

    utils.output.println("\nTotal: {d} DAG preset(s)", .{dag_presets.len});
    utils.output.println("\nUse 'abi multi-agent run-workflow <name>' to execute a DAG workflow", .{});
    utils.output.println("Use 'abi multi-agent run --workflow <name> --task \"...\"' for simple workflows", .{});
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
    utils.output.println("\nTo create a custom workflow, define it in your configuration:", .{});
    utils.output.println("", .{});
    utils.output.println("  {s}", .{workflow_path});
    utils.output.println("", .{});
    utils.output.println("Example workflow definition:", .{});
    utils.output.println("  {{", .{});
    utils.output.println("    \"name\": \"{s}\",", .{name});
    utils.output.println("    \"description\": \"Custom workflow\",", .{});
    utils.output.println("    \"agents\": [\"agent1\", \"agent2\"],", .{});
    utils.output.println("    \"mode\": \"sequential\"", .{});
    utils.output.println("  }}", .{});
    utils.output.println("", .{});

    utils.output.printSuccess("Workflow configuration guide shown", .{});
}

fn getPrimaryWorkflowsDir(allocator: std.mem.Allocator) ![]u8 {
    return app_paths.resolvePath(allocator, "workflows");
}

fn printRunHelp() void {
    utils.output.println("Usage: abi multi-agent run [--workflow <name>] --task \"...\"", .{});
    utils.output.println("       abi multi-agent run [--workflow <name>] \"...\"", .{});
}

fn printCreateHelp() void {
    utils.output.println("Usage: abi multi-agent create <name>", .{});
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
