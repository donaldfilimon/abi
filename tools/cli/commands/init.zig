//! CLI command: abi init
//!
//! Scaffold a new ABI project from a template.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

pub const meta: command_mod.Meta = .{
    .name = "init",
    .description = "Create a new ABI project from a template",
    .subcommands = &.{"help"},
};

const Template = enum {
    default,
    @"llm-app",
    agent,
    training,

    fn description(self: Template) []const u8 {
        return switch (self) {
            .default => "Basic ABI framework project",
            .@"llm-app" => "LLM application with model loading",
            .agent => "AI agent with tool use and sessions",
            .training => "Training pipeline with Ralph integration",
        };
    }
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printHelp(allocator);
        return;
    }

    var template: Template = .default;
    var project_name: []const u8 = "my-abi-project";

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--template") or std.mem.eql(u8, arg, "-t")) {
            if (i < args.len) {
                const tmpl_name = std.mem.sliceTo(args[i], 0);
                i += 1;
                template = parseTemplate(tmpl_name) orelse {
                    utils.output.printError("Unknown template: {s}", .{tmpl_name});
                    utils.output.printInfo("Available: default, llm-app, agent, training", .{});
                    return;
                };
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--list")) {
            listTemplates();
            return;
        }

        // Positional: project name
        project_name = arg;
    }

    utils.output.printHeader("ABI Init");
    utils.output.printKeyValue("Project", project_name);
    utils.output.printKeyValue("Template", @tagName(template));
    utils.output.println("", .{});

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();
    const dir = std.Io.Dir.cwd();

    // Generate build.zig
    const build_content = generateBuildZig(template, project_name);
    dir.writeFile(io, .{ .sub_path = "build.zig", .data = build_content }) catch |err| {
        utils.output.printError("Could not write build.zig: {t}", .{err});
        return;
    };
    utils.output.printSuccess("Created build.zig", .{});

    // Generate src/main.zig
    dir.createDir(io, "src", .default_dir) catch {};
    const main_content = generateMainZig(template);
    dir.writeFile(io, .{ .sub_path = "src/main.zig", .data = main_content }) catch |err| {
        utils.output.printError("Could not write src/main.zig: {t}", .{err});
        return;
    };
    utils.output.printSuccess("Created src/main.zig", .{});

    // Generate .gitignore
    dir.writeFile(io, .{ .sub_path = ".gitignore", .data = gitignore_content }) catch |err| {
        utils.output.printError("Could not write .gitignore: {t}", .{err});
        return;
    };
    utils.output.printSuccess("Created .gitignore", .{});

    // Template-specific files
    if (template == .agent or template == .training) {
        dir.writeFile(io, .{ .sub_path = "ralph.yml", .data = ralph_yml_content }) catch |err| {
            utils.output.printError("Could not write ralph.yml: {t}", .{err});
            return;
        };
        utils.output.printSuccess("Created ralph.yml", .{});
    }

    utils.output.println("", .{});
    utils.output.printSuccess("Project '{s}' initialized!", .{project_name});
    utils.output.println("", .{});
    utils.output.printInfo("Next steps:", .{});
    utils.output.println("  1. zig build              # Build the project", .{});
    utils.output.println("  2. zig build run          # Run the application", .{});
    utils.output.println("  3. abi doctor             # Check your environment", .{});
}

fn parseTemplate(name: []const u8) ?Template {
    if (std.mem.eql(u8, name, "default")) return .default;
    if (std.mem.eql(u8, name, "llm-app")) return .@"llm-app";
    if (std.mem.eql(u8, name, "agent")) return .agent;
    if (std.mem.eql(u8, name, "training")) return .training;
    return null;
}

fn listTemplates() void {
    utils.output.printHeader("Available Templates");
    utils.output.println("", .{});
    utils.output.println("  {s: <12} {s}", .{ "default", "Basic ABI framework project" });
    utils.output.println("  {s: <12} {s}", .{ "llm-app", "LLM application with model loading" });
    utils.output.println("  {s: <12} {s}", .{ "agent", "AI agent with tool use and sessions" });
    utils.output.println("  {s: <12} {s}", .{ "training", "Training pipeline with Ralph integration" });
    utils.output.println("", .{});
    utils.output.printInfo("Use: abi init <name> --template <template>", .{});
}

fn generateBuildZig(template: Template, name: []const u8) []const u8 {
    _ = name;
    return switch (template) {
        .default => build_zig_default,
        .@"llm-app" => build_zig_llm,
        .agent => build_zig_agent,
        .training => build_zig_training,
    };
}

fn generateMainZig(template: Template) []const u8 {
    return switch (template) {
        .default => main_zig_default,
        .@"llm-app" => main_zig_llm,
        .agent => main_zig_agent,
        .training => main_zig_training,
    };
}

// ─── Template content ─────────────────────────────────────────────────────────

const build_zig_default =
    \\const std = @import("std");
    \\
    \\pub fn build(b: *std.Build) void {
    \\    const target = b.standardTargetOptions(.{});
    \\    const optimize = b.standardOptimizeOption(.{});
    \\
    \\    const exe = b.addExecutable(.{
    \\        .name = "app",
    \\        .root_source_file = b.path("src/main.zig"),
    \\        .target = target,
    \\        .optimize = optimize,
    \\    });
    \\    b.installArtifact(exe);
    \\
    \\    const run_cmd = b.addRunArtifact(exe);
    \\    const run_step = b.step("run", "Run the application");
    \\    run_step.dependOn(&run_cmd.step);
    \\}
    \\
;

const build_zig_llm = build_zig_default;
const build_zig_agent = build_zig_default;
const build_zig_training = build_zig_default;

const main_zig_default =
    \\const std = @import("std");
    \\
    \\pub fn main() !void {
    \\    std.debug.print("Hello from ABI!\n", .{});
    \\}
    \\
;

const main_zig_llm =
    \\const std = @import("std");
    \\
    \\pub fn main() !void {
    \\    std.debug.print("ABI LLM Application\n", .{});
    \\    std.debug.print("Set ABI_LLM_MODEL_PATH to your GGUF model file.\n", .{});
    \\    std.debug.print("Then use: abi llm run <model-path>\n", .{});
    \\}
    \\
;

const main_zig_agent =
    \\const std = @import("std");
    \\
    \\pub fn main() !void {
    \\    std.debug.print("ABI Agent Application\n", .{});
    \\    std.debug.print("Configure an AI provider (ABI_OPENAI_API_KEY or ABI_ANTHROPIC_API_KEY).\n", .{});
    \\    std.debug.print("Then use: abi agent --persona coder\n", .{});
    \\}
    \\
;

const main_zig_training =
    \\const std = @import("std");
    \\
    \\pub fn main() !void {
    \\    std.debug.print("ABI Training Pipeline\n", .{});
    \\    std.debug.print("Use: abi train self\n", .{});
    \\    std.debug.print("Or:  abi agent ralph --task \"Your task\"\n", .{});
    \\}
    \\
;

const gitignore_content =
    \\.zig-cache/
    \\zig-out/
    \\.abi/
    \\.ralph/state.json
    \\*.o
    \\*.a
    \\
;

const ralph_yml_content =
    \\# Ralph agent configuration
    \\name: my-project
    \\max_iterations: 10
    \\gates:
    \\  per_iteration: "zig build test --summary all"
    \\skills: []
    \\
;

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi init", "[name] [options]")
        .description("Create a new ABI project from a template.")
        .section("Options")
        .option(.{ .short = "-t", .long = "--template", .arg = "name", .description = "Template: default, llm-app, agent, training" })
        .option(.{ .long = "--list", .description = "List available templates" })
        .option(utils.help.common_options.help)
        .newline()
        .section("Templates")
        .text("  default      Basic ABI framework project\n")
        .text("  llm-app      LLM application with model loading\n")
        .text("  agent        AI agent with tool use and sessions\n")
        .text("  training     Training pipeline with Ralph integration\n")
        .newline()
        .section("Examples")
        .example("abi init my-project", "Create with default template")
        .example("abi init chatbot --template llm-app", "Create LLM app")
        .example("abi init --list", "List available templates");

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
