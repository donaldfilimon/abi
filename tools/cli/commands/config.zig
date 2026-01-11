//! Configuration management command.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

/// Run the config command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0 or utils.args.matchesAny(args[0], &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);

    if (std.mem.eql(u8, command, "init")) {
        try runInit(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "show")) {
        try runShow(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "validate")) {
        try runValidate(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "env")) {
        runEnv();
        return;
    }

    std.debug.print("Unknown config command: {s}\n", .{command});
    printHelp();
}

fn runInit(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var output_path: []const u8 = "abi.json";

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

    // Create default configuration
    const default_config = getDefaultConfigJson();

    // Create io backend for filesystem operations
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Write to file
    const file = std.Io.Dir.cwd().createFile(io, output_path, .{ .truncate = true }) catch |err| {
        std.debug.print("Error creating config file '{s}': {t}\n", .{ output_path, err });
        return;
    };
    defer file.close(io);

    var write_buf: [4096]u8 = undefined;
    var writer = file.writer(io, &write_buf);
    _ = writer.interface.write(default_config) catch |err| {
        std.debug.print("Error writing config file: {t}\n", .{err});
        return;
    };
    writer.flush() catch {};

    std.debug.print("Created configuration file: {s}\n", .{output_path});
    std.debug.print("\nEdit this file to customize your ABI framework settings.\n", .{});
    std.debug.print("Run 'abi config validate {s}' to check your configuration.\n", .{output_path});
}

fn runShow(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var format: enum { human, json } = .human;
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
        // Load and show from file
        var loader = abi.config.ConfigLoader.init(allocator);
        const config = loader.loadFromFile(path) catch |err| {
            std.debug.print("Error loading config file '{s}': {t}\n", .{ path, err });
            return;
        };
        defer @constCast(&config).deinit();

        switch (format) {
            .human => printConfigHuman(&config),
            .json => std.debug.print("{s}\n", .{getDefaultConfigJson()}),
        }
    } else {
        // Show default configuration
        switch (format) {
            .human => printDefaultConfigHuman(),
            .json => std.debug.print("{s}\n", .{getDefaultConfigJson()}),
        }
    }
}

fn runValidate(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        std.debug.print("Usage: abi config validate <config-file>\n", .{});
        return;
    }

    const path = std.mem.sliceTo(args[0], 0);

    // Try to load the configuration file
    var loader = abi.config.ConfigLoader.init(allocator);
    var config = loader.loadFromFile(path) catch |err| {
        std.debug.print("Error: Failed to load '{s}'\n", .{path});
        std.debug.print("  Reason: {t}\n", .{err});
        std.process.exit(1);
    };
    defer config.deinit();

    // Validate the configuration
    config.validate() catch |err| {
        std.debug.print("Error: Configuration validation failed\n", .{});
        std.debug.print("  Reason: {t}\n", .{err});
        std.process.exit(1);
    };

    std.debug.print("Configuration file '{s}' is valid.\n", .{path});
    std.debug.print("\nConfiguration summary:\n", .{});
    printConfigHuman(&config);
}

fn runEnv() void {
    std.debug.print("Environment Variables for ABI Framework\n", .{});
    std.debug.print("========================================\n\n", .{});

    std.debug.print("Framework Settings:\n", .{});
    std.debug.print("  ABI_ENABLE_AI          Enable AI features (true/false)\n", .{});
    std.debug.print("  ABI_ENABLE_GPU         Enable GPU features (true/false)\n", .{});
    std.debug.print("  ABI_ENABLE_WEB         Enable web features (true/false)\n", .{});
    std.debug.print("  ABI_ENABLE_DATABASE    Enable database features (true/false)\n", .{});
    std.debug.print("  ABI_ENABLE_NETWORK     Enable network features (true/false)\n", .{});
    std.debug.print("  ABI_WORKER_THREADS     Number of worker threads (0=auto)\n", .{});
    std.debug.print("  ABI_LOG_LEVEL          Log level (debug/info/warn/err)\n", .{});

    std.debug.print("\nAI Connectors:\n", .{});
    std.debug.print("  ABI_OPENAI_API_KEY     OpenAI API key\n", .{});
    std.debug.print("  OPENAI_API_KEY         OpenAI API key (fallback)\n", .{});
    std.debug.print("  ABI_HF_API_TOKEN       HuggingFace API token\n", .{});
    std.debug.print("  HF_API_TOKEN           HuggingFace API token (fallback)\n", .{});
    std.debug.print("  ABI_OLLAMA_HOST        Ollama host URL\n", .{});
    std.debug.print("  OLLAMA_HOST            Ollama host URL (fallback)\n", .{});

    std.debug.print("\nDatabase:\n", .{});
    std.debug.print("  ABI_DATABASE_NAME      Database file name\n", .{});

    std.debug.print("\nNetwork:\n", .{});
    std.debug.print("  ABI_CLUSTER_ID         Cluster identifier\n", .{});
    std.debug.print("  ABI_NODE_ADDRESS       Node address (host:port)\n", .{});

    std.debug.print("\nWeb:\n", .{});
    std.debug.print("  ABI_WEB_PORT           Web server port\n", .{});
    std.debug.print("  ABI_WEB_CORS           Enable CORS (true/false)\n", .{});

    std.debug.print("\nGPU:\n", .{});
    std.debug.print("  ABI_GPU_BACKEND        Preferred GPU backend\n", .{});
}

fn printHelp() void {
    const help_text =
        "Usage: abi config <command> [options]\n\n" ++
        "Manage ABI framework configuration files.\n\n" ++
        "Commands:\n" ++
        "  init [options]       Generate a default configuration file\n" ++
        "  show [file]          Display configuration (default or from file)\n" ++
        "  validate <file>      Validate a configuration file\n" ++
        "  env                  List environment variables\n" ++
        "  help                 Show this help message\n\n" ++
        "Init options:\n" ++
        "  -o, --output <path>  Output file path (default: abi.json)\n\n" ++
        "Show options:\n" ++
        "  -f, --format <fmt>   Output format: human, json (default: human)\n\n" ++
        "Examples:\n" ++
        "  abi config init                    Create default abi.json\n" ++
        "  abi config init -o myconfig.json   Create custom config file\n" ++
        "  abi config show                    Show default configuration\n" ++
        "  abi config show abi.json           Show file configuration\n" ++
        "  abi config show -f json            Show as JSON\n" ++
        "  abi config validate abi.json       Validate config file\n" ++
        "  abi config env                     List environment variables\n";
    std.debug.print("{s}", .{help_text});
}

fn printConfigHuman(config: *const abi.config.Config) void {
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
