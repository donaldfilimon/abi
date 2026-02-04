//! Multi‑Agent CLI command
//!
//! Provides an interface to the `MultiAgentCoordinator` API for running
//! multi-agent workflows.
//!
//! Subcommands:
//!   * `info`   - Show coordinator status and feature gating
//!   * `run`    - Execute a task using registered agents
//!   * `list`   - List available workflow templates
//!   * `create` - Create a new workflow from template

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

// libc import for environment access - required for Zig 0.16
const c = @cImport(@cInclude("stdlib.h"));

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

/// Entry point for the `multi-agent` command.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    // Show help if no sub‑command or help flag is present.
    if (!parser.hasMore() or parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    const sub = parser.next().?; // safe after hasMore check
    if (std.mem.eql(u8, sub, "info")) {
        try runInfo(allocator);
        return;
    }

    if (std.mem.eql(u8, sub, "run")) {
        try runWorkflow(allocator, &parser);
        return;
    }

    if (std.mem.eql(u8, sub, "list")) {
        try listWorkflows(allocator);
        return;
    }

    if (std.mem.eql(u8, sub, "create")) {
        try createWorkflow(allocator, &parser);
        return;
    }

    if (std.mem.eql(u8, sub, "status")) {
        try showStatus(allocator);
        return;
    }

    utils.output.printError("unknown subcommand: {s}", .{sub});
    printHelp(allocator);
}

fn runInfo(allocator: std.mem.Allocator) !void {
    // Initialise the framework to access runtime feature matrix.
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

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

    // Use the public API re‑exported via `abi.ai`.
    const Coordinator = abi.ai.MultiAgentCoordinator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    const count = coord.agents.items.len;
    utils.output.printKeyValueFmt("Registered Agents", "{d}", .{count});

    // Show available templates
    utils.output.printKeyValueFmt("Workflow Templates", "{d}", .{workflow_templates.len});

    utils.output.printSuccess("Coordinator ready.", .{});
}

fn runWorkflow(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    var workflow_name: ?[]const u8 = null;
    var task_description: ?[]const u8 = null;

    // Parse arguments
    while (parser.hasMore()) {
        if (parser.consumeOption(&[_][]const u8{ "--workflow", "-w" })) |val| {
            workflow_name = val;
        } else if (parser.consumeOption(&[_][]const u8{ "--task", "-t" })) |val| {
            task_description = val;
        } else {
            // Positional argument is the task
            if (task_description == null) {
                task_description = parser.next();
            } else {
                _ = parser.next();
            }
        }
    }

    if (task_description == null) {
        utils.output.printError("No task specified", .{});
        utils.output.printInfo("Usage: abi multi-agent run --task \"your task description\"", .{});
        utils.output.printInfo("       abi multi-agent run \"your task description\"", .{});
        return;
    }

    // Check AI feature
    if (!abi.ai.isEnabled()) {
        utils.output.printError("AI feature is disabled", .{});
        utils.output.printInfo("Rebuild with: zig build -Denable-ai=true", .{});
        return;
    }

    utils.output.printHeader("Multi-Agent Workflow");

    // Show workflow info
    if (workflow_name) |wf| {
        utils.output.printKeyValue("Workflow", wf);
        // Find template
        for (workflow_templates) |tmpl| {
            if (std.mem.eql(u8, tmpl.name, wf)) {
                utils.output.printInfo("Using template: {s}", .{tmpl.description});
                std.debug.print("Agents: ", .{});
                for (tmpl.agents, 0..) |agent, i| {
                    if (i > 0) std.debug.print(", ", .{});
                    std.debug.print("{s}", .{agent});
                }
                std.debug.print("\n", .{});
                break;
            }
        }
    } else {
        utils.output.printKeyValue("Workflow", "default (sequential)");
    }

    utils.output.printKeyValue("Task", task_description.?);

    std.debug.print("\n", .{});

    // Initialize coordinator
    const Coordinator = abi.ai.MultiAgentCoordinator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    // Run the task
    utils.output.printInfo("Executing workflow...", .{});

    if (coord.agents.items.len == 0) {
        utils.output.printWarning("No agents registered in coordinator", .{});
        utils.output.printInfo("The coordinator has no agents. Register agents to execute workflows.", .{});
        utils.output.printInfo("This is a demonstration - real workflows require agent registration.", .{});

        // Simulate workflow execution for demonstration
        std.debug.print("\n[Simulated Workflow Execution]\n", .{});
        std.debug.print("Step 1: Analyzing task...\n", .{});
        std.debug.print("Step 2: Planning execution...\n", .{});
        std.debug.print("Step 3: Executing agents...\n", .{});
        std.debug.print("Step 4: Aggregating results...\n", .{});
        std.debug.print("\n", .{});

        utils.output.printSuccess("Workflow simulation complete", .{});
    } else {
        const result = coord.runTask(task_description.?) catch |err| {
            utils.output.printError("Workflow execution failed: {t}", .{err});
            return;
        };
        defer allocator.free(result);

        std.debug.print("\n[Workflow Output]\n", .{});
        std.debug.print("{s}\n", .{result});

        utils.output.printSuccess("Workflow complete", .{});
    }
}

fn listWorkflows(allocator: std.mem.Allocator) !void {
    _ = allocator;
    utils.output.printHeader("Available Workflow Templates");

    std.debug.print("\n{s:<20} {s:<50}\n", .{ "NAME", "DESCRIPTION" });
    std.debug.print("{s}\n", .{"-" ** 70});

    for (workflow_templates) |tmpl| {
        std.debug.print("{s:<20} {s:<50}\n", .{ tmpl.name, tmpl.description });
    }

    std.debug.print("\nTotal: {d} workflow template(s)\n", .{workflow_templates.len});
    std.debug.print("\nUse 'abi multi-agent run --workflow <name> --task \"...\"' to execute\n", .{});
}

fn createWorkflow(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    const name = parser.next() orelse {
        utils.output.printError("No workflow name specified", .{});
        utils.output.printInfo("Usage: abi multi-agent create <name>", .{});
        return;
    };

    _ = allocator;

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

    // Show creation guidance
    std.debug.print("\nTo create a custom workflow, define it in your configuration:\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("  ~/.abi/workflows/{s}.json\n", .{name});
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

fn showStatus(allocator: std.mem.Allocator) !void {
    utils.output.printHeader("Multi-Agent Status");

    // Check AI feature
    const ai_enabled = abi.ai.isEnabled();
    utils.output.printKeyValue("AI Feature", if (ai_enabled) "enabled" else "disabled");

    if (!ai_enabled) {
        utils.output.printInfo("Multi-agent coordination requires AI feature", .{});
        utils.output.printInfo("Rebuild with: zig build -Denable-ai=true", .{});
        return;
    }

    // Initialize coordinator
    const Coordinator = abi.ai.MultiAgentCoordinator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    utils.output.printKeyValueFmt("Registered Agents", "{d}", .{coord.agents.items.len});
    utils.output.printKeyValueFmt("Workflow Templates", "{d}", .{workflow_templates.len});

    utils.output.printKeyValue("Status", "Ready");

    std.debug.print("\n", .{});
    utils.output.printInfo("Use 'abi multi-agent list' to see available workflows", .{});
    utils.output.printInfo("Use 'abi multi-agent run --task \"...\"' to execute a task", .{});
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
        .subcommand(.{ .name = "list", .description = "List available workflow templates" })
        .subcommand(.{ .name = "create", .description = "Create a new workflow configuration" })
        .subcommand(.{ .name = "status", .description = "Show current coordinator status" })
        .newline()
        .section("Run Options")
        .option(.{ .short = "-w", .long = "--workflow", .arg = "name", .description = "Workflow template to use" })
        .option(.{ .short = "-t", .long = "--task", .arg = "text", .description = "Task description to execute" })
        .newline()
        .section("Examples")
        .example("abi multi-agent info", "Show coordinator info")
        .example("abi multi-agent list", "List workflow templates")
        .example("abi multi-agent run --task \"Review this code\"", "Run with default workflow")
        .example("abi multi-agent run -w code-review -t \"Review PR #123\"", "Run with specific workflow")
        .newline()
        .section("Options")
        .option(utils.help.common_options.help);

    builder.print();
}
