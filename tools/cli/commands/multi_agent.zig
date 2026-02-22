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
const command_mod = @import("../command.zig");
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

// Wrapper functions for comptime children dispatch
fn wrapMaInfo(allocator: std.mem.Allocator, _: []const [:0]const u8) !void {
    try runInfo(allocator);
}
fn wrapMaRun(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try runWorkflow(allocator, &parser);
}
fn wrapMaList(allocator: std.mem.Allocator, _: []const [:0]const u8) !void {
    try listWorkflows(allocator);
}
fn wrapMaCreate(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try createWorkflow(allocator, &parser);
}
fn wrapMaStatus(allocator: std.mem.Allocator, _: []const [:0]const u8) !void {
    try showStatus(allocator);
}

pub const meta: command_mod.Meta = .{
    .name = "multi-agent",
    .description = "Run multi-agent workflows",
    .subcommands = &.{ "info", "run", "list", "create", "status" },
    .children = &.{
        .{ .name = "info", .description = "Show coordinator status and capabilities", .handler = .{ .basic = wrapMaInfo } },
        .{ .name = "run", .description = "Execute a task using the coordinator", .handler = .{ .basic = wrapMaRun } },
        .{ .name = "list", .description = "List available workflow templates", .handler = .{ .basic = wrapMaList } },
        .{ .name = "create", .description = "Create a new workflow configuration", .handler = .{ .basic = wrapMaCreate } },
        .{ .name = "status", .description = "Show current coordinator status", .handler = .{ .basic = wrapMaStatus } },
    },
};

const ma_subcommands = [_][]const u8{
    "info", "run", "list", "create", "status", "help",
};

/// Entry point for the `multi-agent` command.
pub fn run(_: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printHelp(std.heap.page_allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp(std.heap.page_allocator);
        return;
    }
    // Unknown subcommand
    utils.output.printError("unknown subcommand: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &ma_subcommands)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
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
    const Coordinator = abi.ai.multi_agent.Coordinator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    // Determine agent names from template or defaults
    const agent_names: []const []const u8 = blk: {
        if (workflow_name) |wf| {
            for (workflow_templates) |tmpl| {
                if (std.mem.eql(u8, tmpl.name, wf)) {
                    break :blk tmpl.agents;
                }
            }
        }
        break :blk &[_][]const u8{ "agent-1", "agent-2", "agent-3" };
    };

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
    const Coordinator = abi.ai.multi_agent.Coordinator;
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
