//! CLI command modules.
//!
//! Each command is implemented in its own module for maintainability.

const std = @import("std");
const framework = @import("../framework/mod.zig");
const catalog = @import("../tests/catalog.zig");

pub const db = @import("db.zig");
pub const agent = @import("agent.zig");
pub const bench = @import("bench/mod.zig");
pub const config = @import("config.zig");
pub const convert = @import("convert.zig");
pub const discord = @import("discord.zig");
pub const embed = @import("embed.zig");
pub const explore = @import("explore.zig");
pub const gpu = @import("gpu.zig");
pub const gpu_dashboard = @import("gpu_dashboard.zig");
pub const llm = @import("llm/mod.zig");
pub const model = @import("model.zig");
pub const network = @import("network.zig");
pub const simd = @import("simd.zig");
pub const system_info = @import("system_info.zig");
pub const task = @import("task.zig");
pub const tui = @import("tui/mod.zig");
pub const ui = @import("ui/mod.zig");
pub const train = @import("train/mod.zig");
pub const completions = @import("completions.zig");
pub const multi_agent = @import("multi_agent.zig");
pub const plugins = @import("plugins.zig");
pub const profile = @import("profile.zig");
pub const os_agent = @import("os_agent.zig");
pub const status = @import("status.zig");
pub const toolchain = @import("toolchain.zig");
pub const mcp = @import("mcp.zig");
pub const acp = @import("acp.zig");
pub const ralph = @import("ralph/mod.zig");
pub const gendocs = @import("gendocs.zig");

const llm_run = @import("llm/run.zig");
const llm_session = @import("llm/session.zig");
const llm_providers = @import("llm/providers.zig");
const llm_plugins = @import("llm/plugins.zig");
const llm_serve = @import("llm/serve.zig");

const ui_launch = @import("ui/launch.zig");
const ui_gpu = @import("ui/gpu.zig");
const ui_train = @import("ui/train.zig");
const ui_neural = @import("ui/neural.zig");

const CommandDescriptor = framework.types.CommandDescriptor;
const CommandHandler = framework.types.CommandHandler;
const CommandForward = framework.types.CommandForward;

const io_forward_launch = [_][:0]const u8{"launch"};
const io_forward_gpu = [_][:0]const u8{"gpu"};

fn command(comptime name: []const u8) catalog.CommandSpec {
    inline for (catalog.commands) |spec| {
        if (comptime std.mem.eql(u8, name, spec.name)) {
            return spec;
        }
    }
    @compileError("Unknown command catalog entry: " ++ name);
}

const db_meta = command("db");
const agent_meta = command("agent");
const bench_meta = command("bench");
const gpu_meta = command("gpu");
const gpu_dashboard_meta = command("gpu-dashboard");
const network_meta = command("network");
const system_info_meta = command("system-info");
const multi_agent_meta = command("multi-agent");
const explore_meta = command("explore");
const simd_meta = command("simd");
const config_meta = command("config");
const discord_meta = command("discord");
const llm_meta = command("llm");
const model_meta = command("model");
const embed_meta = command("embed");
const train_meta = command("train");
const convert_meta = command("convert");
const task_meta = command("task");
const tui_meta = command("tui");
const ui_meta = command("ui");
const plugins_meta = command("plugins");
const profile_meta = command("profile");
const completions_meta = command("completions");
const status_meta = command("status");
const toolchain_meta = command("toolchain");
const mcp_meta = command("mcp");
const acp_meta = command("acp");
const ralph_meta = command("ralph");
const gendocs_meta = command("gendocs");

fn basicHandler(comptime module: type) CommandHandler {
    return .{ .basic = module.run };
}

fn ioHandler(comptime module: type) CommandHandler {
    return .{ .io = module.run };
}

fn llmRunHandler(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    try llm_run.runRun(allocator, args);
}

fn llmSessionHandler(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    try llm_session.runSession(allocator, args);
}

fn llmServeHandler(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    try llm_serve.runServe(allocator, args);
}

fn llmProvidersHandler(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    try llm_providers.runProviders(allocator, args);
}

fn llmPluginsHandler(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    try llm_plugins.runPlugins(allocator, args);
}

fn uiLaunchHandler(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    try ui_launch.run(allocator, io, args);
}

fn uiGpuHandler(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    try ui_gpu.run(allocator, io, args);
}

fn uiTrainHandler(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    try ui_train.run(allocator, io, args);
}

fn uiNeuralHandler(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    try ui_neural.run(allocator, io, args);
}

const llm_children = [_]CommandDescriptor{
    .{ .name = "run", .description = "One-shot generation through provider router", .handler = .{ .basic = llmRunHandler } },
    .{ .name = "session", .description = "Interactive session through provider router", .handler = .{ .basic = llmSessionHandler } },
    .{ .name = "serve", .description = "Start streaming HTTP server", .handler = .{ .basic = llmServeHandler } },
    .{ .name = "providers", .description = "Show provider availability and routing order", .handler = .{ .basic = llmProvidersHandler } },
    .{ .name = "plugins", .description = "Manage HTTP/native provider plugins", .handler = .{ .basic = llmPluginsHandler } },
};

const ui_children = [_]CommandDescriptor{
    .{ .name = "launch", .description = "Open command launcher TUI", .handler = .{ .io = uiLaunchHandler } },
    .{ .name = "gpu", .description = "Open GPU dashboard TUI", .handler = .{ .io = uiGpuHandler } },
    .{ .name = "train", .description = "Open training monitor TUI", .handler = .{ .io = uiTrainHandler } },
    .{ .name = "neural", .description = "Render dynamic 3D neural network view", .handler = .{ .io = uiNeuralHandler } },
};

pub const descriptors = [_]CommandDescriptor{
    .{ .name = db_meta.name, .description = db_meta.description, .aliases = db_meta.aliases, .subcommands = db_meta.subcommands, .handler = basicHandler(db) },
    .{ .name = agent_meta.name, .description = agent_meta.description, .aliases = agent_meta.aliases, .subcommands = agent_meta.subcommands, .handler = basicHandler(agent) },
    .{ .name = bench_meta.name, .description = bench_meta.description, .aliases = bench_meta.aliases, .subcommands = bench_meta.subcommands, .handler = basicHandler(bench) },
    .{ .name = gpu_meta.name, .description = gpu_meta.description, .aliases = gpu_meta.aliases, .subcommands = gpu_meta.subcommands, .handler = basicHandler(gpu) },
    .{
        .name = gpu_dashboard_meta.name,
        .description = gpu_dashboard_meta.description,
        .aliases = gpu_dashboard_meta.aliases,
        .subcommands = gpu_dashboard_meta.subcommands,
        .handler = ioHandler(gpu_dashboard),
        .forward = CommandForward{
            .target = "ui",
            .prepend_args = &io_forward_gpu,
            .warning = "'abi gpu-dashboard' is deprecated; use 'abi ui gpu'.",
        },
    },
    .{ .name = network_meta.name, .description = network_meta.description, .aliases = network_meta.aliases, .subcommands = network_meta.subcommands, .handler = basicHandler(network) },
    .{ .name = system_info_meta.name, .description = system_info_meta.description, .aliases = system_info_meta.aliases, .subcommands = system_info_meta.subcommands, .handler = basicHandler(system_info) },
    .{ .name = multi_agent_meta.name, .description = multi_agent_meta.description, .aliases = multi_agent_meta.aliases, .subcommands = multi_agent_meta.subcommands, .handler = basicHandler(multi_agent) },
    .{ .name = explore_meta.name, .description = explore_meta.description, .aliases = explore_meta.aliases, .subcommands = explore_meta.subcommands, .handler = basicHandler(explore) },
    .{ .name = simd_meta.name, .description = simd_meta.description, .aliases = simd_meta.aliases, .subcommands = simd_meta.subcommands, .handler = basicHandler(simd) },
    .{ .name = config_meta.name, .description = config_meta.description, .aliases = config_meta.aliases, .subcommands = config_meta.subcommands, .handler = basicHandler(config) },
    .{ .name = discord_meta.name, .description = discord_meta.description, .aliases = discord_meta.aliases, .subcommands = discord_meta.subcommands, .handler = basicHandler(discord) },
    .{
        .name = llm_meta.name,
        .description = llm_meta.description,
        .aliases = llm_meta.aliases,
        .subcommands = llm_meta.subcommands,
        .children = &llm_children,
        .kind = .group,
        .handler = basicHandler(llm),
    },
    .{ .name = model_meta.name, .description = model_meta.description, .aliases = model_meta.aliases, .subcommands = model_meta.subcommands, .handler = basicHandler(model) },
    .{ .name = embed_meta.name, .description = embed_meta.description, .aliases = embed_meta.aliases, .subcommands = embed_meta.subcommands, .handler = basicHandler(embed) },
    .{ .name = train_meta.name, .description = train_meta.description, .aliases = train_meta.aliases, .subcommands = train_meta.subcommands, .handler = basicHandler(train) },
    .{ .name = convert_meta.name, .description = convert_meta.description, .aliases = convert_meta.aliases, .subcommands = convert_meta.subcommands, .handler = basicHandler(convert) },
    .{ .name = task_meta.name, .description = task_meta.description, .aliases = task_meta.aliases, .subcommands = task_meta.subcommands, .handler = basicHandler(task) },
    .{
        .name = tui_meta.name,
        .description = tui_meta.description,
        .aliases = tui_meta.aliases,
        .subcommands = tui_meta.subcommands,
        .handler = ioHandler(tui),
        .forward = CommandForward{
            .target = "ui",
            .prepend_args = &io_forward_launch,
            .warning = "'abi tui' is deprecated; use 'abi ui launch'.",
        },
    },
    .{
        .name = ui_meta.name,
        .description = ui_meta.description,
        .aliases = ui_meta.aliases,
        .subcommands = ui_meta.subcommands,
        .children = &ui_children,
        .kind = .group,
        .handler = ioHandler(ui),
    },
    .{ .name = plugins_meta.name, .description = plugins_meta.description, .aliases = plugins_meta.aliases, .subcommands = plugins_meta.subcommands, .handler = basicHandler(plugins) },
    .{ .name = profile_meta.name, .description = profile_meta.description, .aliases = profile_meta.aliases, .subcommands = profile_meta.subcommands, .handler = basicHandler(profile) },
    .{ .name = completions_meta.name, .description = completions_meta.description, .aliases = completions_meta.aliases, .subcommands = completions_meta.subcommands, .handler = basicHandler(completions) },
    .{ .name = status_meta.name, .description = status_meta.description, .aliases = status_meta.aliases, .subcommands = status_meta.subcommands, .handler = basicHandler(status) },
    .{ .name = toolchain_meta.name, .description = toolchain_meta.description, .aliases = toolchain_meta.aliases, .subcommands = toolchain_meta.subcommands, .handler = basicHandler(toolchain) },
    .{ .name = mcp_meta.name, .description = mcp_meta.description, .aliases = mcp_meta.aliases, .subcommands = mcp_meta.subcommands, .handler = basicHandler(mcp) },
    .{ .name = acp_meta.name, .description = acp_meta.description, .aliases = acp_meta.aliases, .subcommands = acp_meta.subcommands, .handler = basicHandler(acp) },
    .{ .name = ralph_meta.name, .description = ralph_meta.description, .aliases = ralph_meta.aliases, .subcommands = ralph_meta.subcommands, .handler = basicHandler(ralph) },
    .{ .name = gendocs_meta.name, .description = gendocs_meta.description, .aliases = gendocs_meta.aliases, .subcommands = gendocs_meta.subcommands, .handler = ioHandler(gendocs) },
};

pub fn findDescriptor(raw_name: []const u8) ?*const CommandDescriptor {
    for (descriptors) |*descriptor| {
        if (std.mem.eql(u8, raw_name, descriptor.name)) return descriptor;
        for (descriptor.aliases) |alias| {
            if (std.mem.eql(u8, raw_name, alias)) return descriptor;
        }
    }
    return null;
}
