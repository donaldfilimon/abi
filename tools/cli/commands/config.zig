//! Configuration management command.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;
const app_paths = abi.shared.app_paths;

// Use the shared config module for file-based configuration (legacy format)
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const shared_config = abi.shared.utils.config;

const OutputFormat = enum {
    human,
    json,
    zon,
};

fn wrapCfgInit(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try runInit(ctx, args);
}
fn wrapCfgShow(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try runShow(ctx, args);
}
fn wrapCfgValidate(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try runValidate(ctx, args);
}
fn wrapCfgEnv(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    _ = ctx;
    runEnv();
}
fn wrapCfgPath(ctx: *const context_mod.CommandContext, _: []const [:0]const u8) !void {
    try runPath(ctx);
}
fn wrapCfgSetup(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try runSetup(ctx, args);
}

pub const meta: command_mod.Meta = .{
    .name = "config",
    .description = "Configuration management (init, setup, show, validate, env)",
    .subcommands = &.{ "init", "setup", "show", "validate", "env", "path", "help" },
    .children = &.{
        .{ .name = "init", .description = "Generate a default configuration file", .handler = wrapCfgInit },
        .{ .name = "setup", .description = "Bootstrap user config in platform default location", .handler = wrapCfgSetup },
        .{ .name = "show", .description = "Display current configuration", .handler = wrapCfgShow },
        .{ .name = "validate", .description = "Validate a configuration file", .handler = wrapCfgValidate },
        .{ .name = "env", .description = "List environment variables", .handler = wrapCfgEnv },
        .{ .name = "path", .description = "Print platform default config path", .handler = wrapCfgPath },
    },
};

const config_subcommands = [_][]const u8{
    "init", "setup", "show", "validate", "env", "path", "help",
};

/// Run the config command with the provided arguments.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    _ = allocator;
    if (args.len == 0) {
        printHelp();
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp();
        return;
    }
    // Unknown subcommand
    utils.output.printError("Unknown config command: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &config_subcommands)) |suggestion| {
        utils.output.printInfo("Did you mean: {s}", .{suggestion});
    }
    utils.output.printInfo("Run 'abi config help' for usage.", .{});
}

fn runInit(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var output_path: []const u8 = "abi.zon";

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--output", "-o" })) {
            if (i < args.len) {
                output_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    // Create default configuration (ZON or JSON based on output path)
    const default_config = if (std.mem.endsWith(u8, output_path, ".json"))
        getDefaultConfigJson()
    else
        getDefaultConfigZon();

    // Create io backend for filesystem operations
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Write to file
    const file = std.Io.Dir.cwd().createFile(io, output_path, .{ .truncate = true }) catch |err| {
        utils.output.printError("creating config file '{s}': {t}", .{ output_path, err });
        return;
    };
    defer file.close(io);

    // Use writeStreamingAll for Zig 0.16 compatibility
    file.writeStreamingAll(io, default_config) catch |err| {
        utils.output.printError("writing config file: {t}", .{err});
        return;
    };

    std.debug.print("Created configuration file: {s}\n", .{output_path});
    std.debug.print("\nEdit this file to customize your ABI framework settings.\n", .{});
    std.debug.print("Run 'abi config validate {s}' to check your configuration.\n", .{output_path});
}

fn runShow(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var format: OutputFormat = .human;
    var config_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--format", "-f" })) {
            if (i < args.len) {
                const fmt = std.mem.sliceTo(args[i], 0);
                if (std.mem.eql(u8, fmt, "json")) {
                    format = .json;
                } else if (std.mem.eql(u8, fmt, "zon")) {
                    format = .zon;
                }
                i += 1;
            }
            continue;
        }

        if (config_path == null) {
            config_path = std.mem.sliceTo(arg, 0);
        }
    }

    if (config_path) |path| {
        var config = loadConfigFromFile(allocator, path) catch |err| {
            utils.output.printError("loading config file '{s}': {t}", .{ path, err });
            return;
        };
        defer config.deinit();
        return printByFormat(allocator, format, &config);
    } else {
        const resolved_path = app_paths.resolvePath(allocator, "config.json") catch |err| {
            if (err == error.NoHomeDirectory) {
                switch (format) {
                    .human => printDefaultConfigHuman(),
                    .json => std.debug.print("{s}\n", .{getDefaultConfigJson()}),
                    .zon => std.debug.print("{s}\n", .{getDefaultConfigZon()}),
                }
                return;
            }
            return err;
        };
        defer allocator.free(resolved_path);

        if (loadConfigFromFile(allocator, resolved_path)) |loaded_config| {
            var config = loaded_config;
            defer config.deinit();
            return printByFormat(allocator, format, &config);
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }

        // Show built-in default configuration only when no user config exists.
        switch (format) {
            .human => printDefaultConfigHuman(),
            .json => std.debug.print("{s}\n", .{getDefaultConfigJson()}),
            .zon => std.debug.print("{s}\n", .{getDefaultConfigZon()}),
        }
    }
}

fn runValidate(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len == 0) {
        std.debug.print("Usage: abi config validate <config-file>\n", .{});
        return;
    }

    if (utils.args.matchesAny(std.mem.sliceTo(args[0], 0), &[_][]const u8{ "--help", "-h", "help" })) {
        std.debug.print("Usage: abi config validate <config-file>\n", .{});
        return;
    }

    const path = std.mem.sliceTo(args[0], 0);

    // Try to load the configuration file using shared config loader (legacy format)
    const output = utils.output;
    var loader = shared_config.ConfigLoader.init(allocator);
    var config = loader.loadFromFile(path) catch |err| {
        output.printError("Failed to load '{s}'", .{path});
        output.println("  Reason: {t}", .{err});
        return error.ExecutionFailed;
    };
    defer config.deinit();

    // Validate the configuration
    config.validate() catch |err| {
        output.printError("Configuration validation failed", .{});
        output.println("  Reason: {t}", .{err});
        return error.ExecutionFailed;
    };

    output.printSuccess("Configuration file '{s}' is valid.", .{path});
    output.println("\nConfiguration summary:", .{});
    printConfigHuman(&config);
}

fn runEnv() void {
    const output = utils.output;
    output.printHeader("Environment Variables for ABI Framework");

    output.println("\n{s}Framework Settings:{s}", .{ output.Color.bold(), output.Color.reset() });
    output.printKeyValue("ABI_ENABLE_AI", "Enable AI features (true/false)");
    output.printKeyValue("ABI_ENABLE_GPU", "Enable GPU features (true/false)");
    output.printKeyValue("ABI_ENABLE_WEB", "Enable web features (true/false)");
    output.printKeyValue("ABI_ENABLE_DATABASE", "Enable database features (true/false)");
    output.printKeyValue("ABI_ENABLE_NETWORK", "Enable network features (true/false)");
    output.printKeyValue("ABI_WORKER_THREADS", "Number of worker threads (0=auto)");
    output.printKeyValue("ABI_LOG_LEVEL", "Log level (debug/info/warn/err)");

    output.println("\n{s}AI Connectors:{s}", .{ output.Color.bold(), output.Color.reset() });
    output.printKeyValue("ABI_OPENAI_API_KEY", "OpenAI API key");
    output.printKeyValue("OPENAI_API_KEY", "OpenAI API key (fallback)");
    output.printKeyValue("ABI_HF_API_TOKEN", "HuggingFace API token");
    output.printKeyValue("HF_API_TOKEN", "HuggingFace API token (fallback)");
    output.printKeyValue("ABI_OLLAMA_HOST", "Ollama host URL");
    output.printKeyValue("OLLAMA_HOST", "Ollama host URL (fallback)");

    output.println("\n{s}Database:{s}", .{ output.Color.bold(), output.Color.reset() });
    output.printKeyValue("ABI_DATABASE_NAME", "Database file name");

    output.println("\n{s}Network:{s}", .{ output.Color.bold(), output.Color.reset() });
    output.printKeyValue("ABI_CLUSTER_ID", "Cluster identifier");
    output.printKeyValue("ABI_NODE_ADDRESS", "Node address (host:port)");

    output.println("\n{s}Web:{s}", .{ output.Color.bold(), output.Color.reset() });
    output.printKeyValue("ABI_WEB_PORT", "Web server port");
    output.printKeyValue("ABI_WEB_CORS", "Enable CORS (true/false)");

    output.println("\n{s}GPU:{s}", .{ output.Color.bold(), output.Color.reset() });
    output.printKeyValue("ABI_GPU_BACKEND", "Preferred GPU backend");
}

fn printByFormat(
    allocator: std.mem.Allocator,
    format: OutputFormat,
    config: *const shared_config.Config,
) !void {
    switch (format) {
        .human => printConfigHuman(config),
        .json => try printConfigJson(allocator, config),
        .zon => std.debug.print("{s}\n", .{getDefaultConfigZon()}),
    }
}

fn loadConfigFromFile(allocator: std.mem.Allocator, path: []const u8) !shared_config.Config {
    var loader = shared_config.ConfigLoader.init(allocator);
    defer loader.deinit();
    return try loader.loadFromFile(path);
}

fn runPath(ctx: *const context_mod.CommandContext) !void {
    const allocator = ctx.allocator;
    const primary_dir = try app_paths.resolvePrimaryRoot(allocator);
    defer allocator.free(primary_dir);

    const config_path = try app_paths.resolvePath(allocator, "config.json");
    defer allocator.free(config_path);

    std.debug.print("Primary user config directory: {s}\n", .{primary_dir});
    std.debug.print("Primary user config file:      {s}\n", .{config_path});
}

fn runSetup(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var force = false;

    for (args) |arg_z| {
        const arg = std.mem.sliceTo(arg_z, 0);
        if (utils.args.matchesAny(arg, &.{ "--force", "-f" })) {
            force = true;
            continue;
        }
        if (utils.args.matchesAny(arg, &.{ "--help", "-h", "help" })) {
            printSetupHelp();
            return;
        }
        utils.output.printError("unknown setup option: {s}", .{arg});
        printSetupHelp();
        return;
    }

    const config_path = try app_paths.resolvePath(allocator, "config.json");
    defer allocator.free(config_path);
    const dir_path = std.fs.path.dirname(config_path) orelse ".";

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    std.Io.Dir.cwd().createDirPath(io, dir_path) catch |err| {
        utils.output.printError("creating config directory '{s}': {t}", .{ dir_path, err });
        return;
    };

    const existing = std.Io.Dir.cwd().openFile(io, config_path, .{}) catch |err| switch (err) {
        error.FileNotFound => null,
        else => {
            utils.output.printError("checking config file '{s}': {t}", .{ config_path, err });
            return;
        },
    };
    if (existing) |file| {
        file.close(io);
        if (!force) {
            std.debug.print("Config already exists: {s}\n", .{config_path});
            std.debug.print("Use 'abi config setup --force' to overwrite it.\n", .{});
            printStartupHints(config_path);
            return;
        }
    }

    const file = std.Io.Dir.cwd().createFile(io, config_path, .{ .truncate = true }) catch |err| {
        utils.output.printError("creating config file '{s}': {t}", .{ config_path, err });
        return;
    };
    defer file.close(io);

    file.writeStreamingAll(io, getDefaultConfigJson()) catch |err| {
        utils.output.printError("writing config file: {t}", .{err});
        return;
    };

    std.debug.print("User config ready: {s}\n", .{config_path});
    printStartupHints(config_path);
}

fn printSetupHelp() void {
    const help_text =
        "Usage: abi config setup [--force]\n\n" ++
        "Create the default user config file in the platform-specific location.\n\n" ++
        "Options:\n" ++
        "  -f, --force          Overwrite existing config file\n";
    std.debug.print("{s}", .{help_text});
}

fn printStartupHints(config_path: []const u8) void {
    std.debug.print("\nQuick start:\n", .{});
    std.debug.print("  abi launch                         Open command launcher\n", .{});
    std.debug.print("  abi config show {s}         Review config\n", .{config_path});
    std.debug.print("  abi llm discover                    Detect available LLM providers\n", .{});
    std.debug.print("  abi profile show                    Check active profile settings\n", .{});
}

fn printHelp() void {
    const help_text =
        "Usage: abi config <command> [options]\n\n" ++
        "Manage ABI framework configuration files.\n\n" ++
        "Commands:\n" ++
        "  init [options]       Generate a default configuration file\n" ++
        "  setup [--force]      Create user config in platform default location\n" ++
        "  show [file]          Display configuration (default or from file)\n" ++
        "  validate <file>      Validate a configuration file\n" ++
        "  env                  List environment variables\n" ++
        "  path                 Print primary config path + legacy fallback path\n" ++
        "  help                 Show this help message\n\n" ++
        "Init options:\n" ++
        "  -o, --output <path>  Output file path (default: abi.zon)\n\n" ++
        "Show options:\n" ++
        "  -f, --format <fmt>   Output format: human, json, zon (default: human)\n\n" ++
        "Examples:\n" ++
        "  abi config init                    Create default abi.zon\n" ++
        "  abi config init -o myconfig.json   Create JSON config file\n" ++
        "  abi config setup                   Bootstrap user config in platform location\n" ++
        "  abi config path                    Print user config path\n" ++
        "  abi config show                    Show default configuration\n" ++
        "  abi config show abi.json           Show file configuration\n" ++
        "  abi config show -f json            Show as JSON\n" ++
        "  abi config validate abi.json       Validate config file\n" ++
        "  abi config env                     List environment variables\n";
    std.debug.print("{s}", .{help_text});
}

fn printConfigHuman(config: *const shared_config.Config) void {
    std.debug.print("  Source: {t}\n", .{config.source});
    std.debug.print("\n  [Framework]\n", .{});
    std.debug.print("    enable_ai: {s}\n", .{utils.output.boolLabel(config.framework.enable_ai)});
    std.debug.print("    enable_gpu: {s}\n", .{utils.output.boolLabel(config.framework.enable_gpu)});
    std.debug.print("    enable_web: {s}\n", .{utils.output.boolLabel(config.framework.enable_web)});
    std.debug.print("    enable_database: {s}\n", .{utils.output.boolLabel(config.framework.enable_database)});
    std.debug.print("    enable_network: {s}\n", .{utils.output.boolLabel(config.framework.enable_network)});
    std.debug.print("    worker_threads: {d}\n", .{config.framework.worker_threads});
    std.debug.print("    log_level: {t}\n", .{config.framework.log_level});

    std.debug.print("\n  [Database]\n", .{});
    std.debug.print("    name: {s}\n", .{config.database.name});
    std.debug.print("    persistence_enabled: {s}\n", .{utils.output.boolLabel(config.database.persistence_enabled)});
    std.debug.print("    vector_search_enabled: {s}\n", .{utils.output.boolLabel(config.database.vector_search_enabled)});
    std.debug.print("    default_search_limit: {d}\n", .{config.database.default_search_limit});

    std.debug.print("\n  [AI]\n", .{});
    std.debug.print("    temperature: {d:.2}\n", .{config.ai.temperature});
    std.debug.print("    max_tokens: {d}\n", .{config.ai.max_tokens});
    std.debug.print("    streaming_enabled: {s}\n", .{utils.output.boolLabel(config.ai.streaming_enabled)});

    std.debug.print("\n  [Network]\n", .{});
    std.debug.print("    distributed_enabled: {s}\n", .{utils.output.boolLabel(config.network.distributed_enabled)});
    std.debug.print("    cluster_id: {s}\n", .{config.network.cluster_id});
    std.debug.print("    node_address: {s}\n", .{config.network.node_address});

    std.debug.print("\n  [Web]\n", .{});
    std.debug.print("    server_enabled: {s}\n", .{utils.output.boolLabel(config.web.server_enabled)});
    std.debug.print("    port: {d}\n", .{config.web.port});
    std.debug.print("    cors_enabled: {s}\n", .{utils.output.boolLabel(config.web.cors_enabled)});
}

fn printDefaultConfigHuman() void {
    std.debug.print("Default Configuration:\n", .{});
    std.debug.print("  Source: default\n", .{});
    std.debug.print("\n  [Framework]\n", .{});
    std.debug.print("    enable_ai: yes\n", .{});
    std.debug.print("    enable_gpu: yes\n", .{});
    std.debug.print("    enable_web: yes\n", .{});
    std.debug.print("    enable_database: yes\n", .{});
    std.debug.print("    enable_network: no\n", .{});
    std.debug.print("    worker_threads: 0 (auto-detect)\n", .{});
    std.debug.print("    log_level: info\n", .{});

    std.debug.print("\n  [Database]\n", .{});
    std.debug.print("    name: abi.db\n", .{});
    std.debug.print("    persistence_enabled: yes\n", .{});
    std.debug.print("    vector_search_enabled: yes\n", .{});
    std.debug.print("    default_search_limit: 10\n", .{});

    std.debug.print("\n  [AI]\n", .{});
    std.debug.print("    temperature: 0.70\n", .{});
    std.debug.print("    max_tokens: 2048\n", .{});
    std.debug.print("    streaming_enabled: yes\n", .{});

    std.debug.print("\n  [Network]\n", .{});
    std.debug.print("    distributed_enabled: no\n", .{});
    std.debug.print("    cluster_id: default\n", .{});
    std.debug.print("    node_address: 0.0.0.0:9000\n", .{});

    std.debug.print("\n  [Web]\n", .{});
    std.debug.print("    server_enabled: no\n", .{});
    std.debug.print("    port: 8080\n", .{});
    std.debug.print("    cors_enabled: yes\n", .{});
}

fn printConfigJson(allocator: std.mem.Allocator, config: *const shared_config.Config) !void {
    const SerializableConfig = struct {
        framework: shared_config.FrameworkConfig,
        database: shared_config.DatabaseConfig,
        gpu: shared_config.GpuConfig,
        ai: shared_config.AiConfig,
        network: shared_config.NetworkConfig,
        web: shared_config.WebConfig,
    };

    const payload: SerializableConfig = .{
        .framework = config.framework,
        .database = config.database,
        .gpu = config.gpu,
        .ai = config.ai,
        .network = config.network,
        .web = config.web,
    };

    var json_writer: std.Io.Writer.Allocating = .init(allocator);
    defer json_writer.deinit();
    try std.json.Stringify.value(payload, .{ .whitespace = .indent_2 }, &json_writer.writer);

    const json_data = try json_writer.toOwnedSlice();
    defer allocator.free(json_data);
    std.debug.print("{s}\n", .{json_data});
}

fn getDefaultConfigZon() []const u8 {
    return 
    \\// ABI Framework Configuration (ZON format)
    \\.{
    \\    .framework = .{
    \\        .enable_ai = true,
    \\        .enable_gpu = true,
    \\        .enable_web = true,
    \\        .enable_database = true,
    \\        .enable_network = false,
    \\        .enable_profiling = false,
    \\        .worker_threads = 0,
    \\        .log_level = .info,
    \\    },
    \\    .database = .{
    \\        .name = "abi.db",
    \\        .max_records = 0,
    \\        .persistence_enabled = true,
    \\        .persistence_path = "abi_data",
    \\        .vector_search_enabled = true,
    \\        .default_search_limit = 10,
    \\        .max_vector_dimension = 4096,
    \\    },
    \\    .gpu = .{
    \\        .enable_cuda = false,
    \\        .enable_vulkan = false,
    \\        .enable_metal = false,
    \\        .enable_webgpu = false,
    \\        .enable_opengl = false,
    \\        .enable_opengles = false,
    \\        .enable_webgl2 = false,
    \\        .preferred_backend = .auto,
    \\        .memory_pool_mb = 0,
    \\    },
    \\    .ai = .{
    \\        .default_model = "",
    \\        .max_tokens = 2048,
    \\        .temperature = 0.7,
    \\        .top_p = 0.9,
    \\        .streaming_enabled = true,
    \\        .timeout_ms = 60000,
    \\        .history_enabled = true,
    \\        .max_history = 100,
    \\    },
    \\    .network = .{
    \\        .distributed_enabled = false,
    \\        .cluster_id = "default",
    \\        .node_address = "0.0.0.0:9000",
    \\        .heartbeat_interval_ms = 5000,
    \\        .node_timeout_ms = 30000,
    \\        .max_nodes = 16,
    \\        .peer_discovery = false,
    \\    },
    \\    .web = .{
    \\        .server_enabled = false,
    \\        .port = 8080,
    \\        .max_connections = 256,
    \\        .request_timeout_ms = 30000,
    \\        .cors_enabled = true,
    \\        .cors_origins = "*",
    \\    },
    \\}
    \\
    ;
}

fn getDefaultConfigJson() []const u8 {
    return 
    \\{
    \\  "framework": {
    \\    "enable_ai": true,
    \\    "enable_gpu": true,
    \\    "enable_web": true,
    \\    "enable_database": true,
    \\    "enable_network": false,
    \\    "enable_profiling": false,
    \\    "worker_threads": 0,
    \\    "log_level": "info"
    \\  },
    \\  "database": {
    \\    "name": "abi.db",
    \\    "max_records": 0,
    \\    "persistence_enabled": true,
    \\    "persistence_path": "abi_data",
    \\    "vector_search_enabled": true,
    \\    "default_search_limit": 10,
    \\    "max_vector_dimension": 4096
    \\  },
    \\  "gpu": {
    \\    "enable_cuda": false,
    \\    "enable_vulkan": false,
    \\    "enable_metal": false,
    \\    "enable_webgpu": false,
    \\    "enable_opengl": false,
    \\    "enable_opengles": false,
    \\    "enable_webgl2": false,
    \\    "preferred_backend": "",
    \\    "memory_pool_mb": 0
    \\  },
    \\  "ai": {
    \\    "default_model": "",
    \\    "max_tokens": 2048,
    \\    "temperature": 0.7,
    \\    "top_p": 0.9,
    \\    "streaming_enabled": true,
    \\    "timeout_ms": 60000,
    \\    "history_enabled": true,
    \\    "max_history": 100
    \\  },
    \\  "network": {
    \\    "distributed_enabled": false,
    \\    "cluster_id": "default",
    \\    "node_address": "0.0.0.0:9000",
    \\    "heartbeat_interval_ms": 5000,
    \\    "node_timeout_ms": 30000,
    \\    "max_nodes": 16,
    \\    "peer_discovery": false
    \\  },
    \\  "web": {
    \\    "server_enabled": false,
    \\    "port": 8080,
    \\    "max_connections": 256,
    \\    "request_timeout_ms": 30000,
    \\    "cors_enabled": true,
    \\    "cors_origins": "*"
    \\  }
    \\}
    \\
    ;
}
