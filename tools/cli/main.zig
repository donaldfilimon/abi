const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize the framework orchestration layer to coordinate feature toggles,
    // logging, and plugin discovery for the modernized runtime.
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer framework.deinit();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len <= 1) {
        try printHelp();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        try printHelp();
        return;
    }

    if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "--version") or std.mem.eql(u8, command, "-v")) {
        std.debug.print("ABI Framework v{s}\n", .{abi.version()});
        return;
    }

    if (std.mem.eql(u8, command, "db")) {
        try runDb(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "agent")) {
        try runAgent(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "system-info")) {
        try runSystemInfo(&framework);
        return;
    }

    std.debug.print("Unknown command: {s}\nUse 'help' for usage.\n", .{command});
    std.process.exit(1);
}

fn printHelp() !void {
    const tui = abi.plugins.tui;
    var app = tui.App.init(std.heap.page_allocator, "ABI Framework CLI");
    app.header();

    const help_text =
        \\Usage: abi <command> [options]
        \\
        \\Commands:
        \\  db <subcommand>   Database operations (add, query, knn, stats, server, etc.)
        \\  agent             Run AI agent interactions
        \\  system-info       Show system and GPU information
        \\  version           Show framework version
        \\  help              Show this help message
        \\
        \\Run 'abi db help' for database specific commands.
    ;
    std.debug.print("{s}\n", .{help_text});
    app.footer("Modernized ABI Runtime Engine");
}

fn runSystemInfo(framework: *abi.Framework) !void {
    const tui = abi.plugins.tui;
    const platform = abi.platform.platform;
    const gpu_detection = abi.gpu.hardware_detection;

    var app = tui.App.init(framework.allocator, "System Diagnostics");
    app.header();

    tui.info("Gathering platform metadata and hardware capabilities...");

    const info = platform.PlatformInfo.detect();

    var table = try tui.Table.init(framework.allocator, &.{ "Metric", "Value" });
    defer table.deinit();

    try table.addRow(&.{ "OS", @tagName(info.os) });
    try table.addRow(&.{ "Arch", @tagName(info.arch) });
    try table.addRow(&.{ "Zig Version", @import("builtin").zig_version_string });
    try table.addRow(&.{ "ABI Version", abi.version() });

    const threads_buf = try std.fmt.allocPrint(framework.allocator, "{d}", .{info.max_threads});
    defer framework.allocator.free(threads_buf);
    try table.addRow(&.{ "Recommended Threads", threads_buf });

    table.render();
    std.debug.print("\n", .{});

    tui.info("Framework Feature Matrix:");
    var feature_table = try tui.Table.init(framework.allocator, &.{ "Feature", "Status" });
    defer feature_table.deinit();

    // Iterate through current feature families using framework reflection
    inline for (std.enums.values(abi.Feature)) |tag| {
        const status = if (framework.isFeatureEnabled(tag)) "Enabled" else "Disabled";
        try feature_table.addRow(&.{ @tagName(tag), status });
    }
    feature_table.render();
    std.debug.print("\n", .{});

    tui.info("Hardware Acceleration Details:");
    var detector = gpu_detection.GPUDetector.init(framework.allocator);
    const result = try detector.detectGPUs();
    defer @constCast(&result).deinit();

    if (result.total_gpus > 0) {
        for (result.gpus) |gpu_info| {
            std.debug.print("  GPU: {s} [{s}]\n", .{ gpu_info.name, @tagName(gpu_info.performance_tier) });
            std.debug.print("    VRAM: {d} MB\n", .{gpu_info.memory_size / (1024 * 1024)});
            std.debug.print("    Available Backends: ", .{});
            for (gpu_info.available_backends, 0..) |backend, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{s}", .{backend.displayName()});
            }
            std.debug.print("\n", .{});
        }
    } else {
        tui.warning("No discrete compute units detected. Utilizing software fallback engine.");
    }
}

fn runDb(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    const cli_mod = abi.wdbx.cli;
    const tui = abi.plugins.tui;

    var options = cli_mod.Options{};
    var app = tui.App.init(allocator, "WDBX Database Controller");

    if (args.len == 0) {
        options.command = .help;
    } else {
        const cmd_str = args[0];
        if (cli_mod.Command.fromString(cmd_str)) |cmd| {
            options.command = cmd;
        } else {
            if (std.mem.eql(u8, cmd_str, "--help") or std.mem.eql(u8, cmd_str, "-h")) {
                options.command = .help;
            } else if (std.mem.eql(u8, cmd_str, "--version") or std.mem.eql(u8, cmd_str, "-v")) {
                options.command = .version;
            } else {
                tui.errorMsg("Invalid database command specified.");
                return;
            }
        }
    }

    // Parse options for the database CLI instance
    var i: usize = if (args.len > 0) 1 else 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, arg, "--db")) {
            if (i < args.len) {
                options.db_path = try allocator.dupe(u8, args[i]);
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--vector")) {
            if (i < args.len) {
                options.vector = try allocator.dupe(u8, args[i]);
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--k")) {
            if (i < args.len) {
                options.k = std.fmt.parseInt(usize, args[i], 10) catch 5;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--port")) {
            if (i < args.len) {
                options.port = std.fmt.parseInt(u16, args[i], 10) catch 8080;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--host")) {
            if (i < args.len) {
                options.host = try allocator.dupe(u8, args[i]);
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--http")) {
            options.server_type = try allocator.dupe(u8, "http");
        } else if (std.mem.eql(u8, arg, "--tcp")) {
            options.server_type = try allocator.dupe(u8, "tcp");
        } else if (std.mem.eql(u8, arg, "--verbose")) {
            options.verbose = true;
        } else if (std.mem.eql(u8, arg, "--quiet")) {
            options.quiet = true;
        } else if (std.mem.eql(u8, arg, "--debug")) {
            options.debug = true;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            options.profile = true;
        } else if (std.mem.eql(u8, arg, "--role")) {
            if (i < args.len) {
                options.role = try allocator.dupe(u8, args[i]);
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--format")) {
            if (i < args.len) {
                options.output_format = cli_mod.OutputFormat.fromString(args[i]) orelse .text;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--config")) {
            if (i < args.len) {
                options.config_file = try allocator.dupe(u8, args[i]);
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--log-level")) {
            if (i < args.len) {
                options.log_level = cli_mod.LogLevel.fromString(args[i]) orelse .info;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--max-connections")) {
            if (i < args.len) {
                options.max_connections = std.fmt.parseInt(usize, args[i], 10) catch 100;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--timeout")) {
            if (i < args.len) {
                options.timeout_ms = std.fmt.parseInt(u64, args[i], 10) catch 30000;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            if (i < args.len) {
                options.batch_size = std.fmt.parseInt(usize, args[i], 10) catch 100;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--compression")) {
            if (i < args.len) {
                options.compression_level = std.fmt.parseInt(u8, args[i], 10) catch 3;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--metrics")) {
            options.enable_metrics = true;
        } else if (std.mem.eql(u8, arg, "--metrics-port")) {
            if (i < args.len) {
                options.metrics_port = std.fmt.parseInt(u16, args[i], 10) catch 9090;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--trace")) {
            options.enable_tracing = true;
        } else if (std.mem.eql(u8, arg, "--trace-file")) {
            if (i < args.len) {
                options.trace_file = try allocator.dupe(u8, args[i]);
                i += 1;
            }
        }
    }

    if (options.command != .help and options.command != .version) {
        app.header();
    }

    var cli_instance = try cli_mod.WdbxCLI.init(allocator, options);
    defer cli_instance.deinit();

    try cli_instance.run();
}

fn runAgent(allocator: std.mem.Allocator, args: []const [:0]u8) !void {
    const agent_mod = abi.ai.agent;
    const tui = abi.plugins.tui;

    var name: []const u8 = "cli-agent";
    var msg: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;
        if (std.mem.eql(u8, arg, "--name")) {
            if (i < args.len) {
                name = args[i];
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--message") or std.mem.eql(u8, arg, "-m")) {
            if (i < args.len) {
                msg = args[i];
                i += 1;
            }
        }
    }

    var init_spinner = tui.Spinner.init(0, 0, "Initializing Agent Orchestrator...");
    init_spinner.render();

    var agent = try agent_mod.Agent.init(allocator, .{
        .name = name,
        .enable_history = true,
    });
    defer agent.deinit();

    std.debug.print("\x1b[1A\x1b[2K", .{});
    tui.success(try std.fmt.allocPrint(allocator, "Agent '{s}' synchronized with runtime.", .{agent.name()}));

    if (msg) |m| {
        var proc_spinner = tui.Spinner.init(0, 0, "Inferring response...");
        proc_spinner.render();

        const response = try agent.process(m, allocator);
        defer allocator.free(response);

        std.debug.print("\x1b[1A\x1b[2K", .{});
        std.debug.print("{s}User:{s} {s}\n", .{ tui.ansi.bold, tui.ansi.reset, m });
        std.debug.print("{s}Agent:{s} {s}\n", .{ tui.ansi.bright_cyan, tui.ansi.reset, response });
    } else {
        std.debug.print("\nInteractive Session (type 'exit' or 'quit' to terminate):\n", .{});
        const stdin = std.io.getStdIn().reader();
        var buf: [4096]u8 = undefined;

        while (true) {
            std.debug.print("{s}user>{s} ", .{ tui.ansi.bright_green, tui.ansi.reset });
            if (try stdin.readUntilDelimiterOrEof(&buf, '\n')) |line| {
                const trimmed = std.mem.trim(u8, line, "\r\n ");
                if (trimmed.len == 0) continue;
                if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) break;

                var thinking_spinner = tui.Spinner.init(0, 0, "Thinking...");
                thinking_spinner.render();

                const response = try agent.process(trimmed, allocator);
                defer allocator.free(response);

                std.debug.print("\x1b[1A\x1b[2K", .{});
                std.debug.print("{s}agent>{s} {s}\n", .{ tui.ansi.bright_cyan, tui.ansi.reset, response });
            } else {
                break;
            }
        }
    }
}
