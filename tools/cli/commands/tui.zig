//! Interactive TUI command launcher.
//!
//! Provides a terminal-based interface for selecting and running ABI CLI commands.

const std = @import("std");
const abi = @import("abi");
const tui = @import("../tui/mod.zig");
const utils = @import("../utils/mod.zig");

const agent = @import("agent.zig");
const bench = @import("bench.zig");
const config = @import("config.zig");
const db = @import("db.zig");
const discord = @import("discord.zig");
const embed = @import("embed.zig");
const explore = @import("explore.zig");
const gpu = @import("gpu.zig");
const llm = @import("llm.zig");
const network = @import("network.zig");
const simd = @import("simd.zig");
const system_info = @import("system_info.zig");
const train = @import("train.zig");

const Action = union(enum) {
    command: Command,
    version: void,
    help: void,
    quit: void,
};

const Command = enum {
    db,
    agent,
    bench,
    config,
    discord,
    embed,
    explore,
    gpu,
    llm,
    network,
    simd,
    system_info,
    train,
};

const MenuItem = struct {
    label: []const u8,
    description: []const u8,
    action: Action,
};

const empty_args = &[_][:0]const u8{};

/// Entry point for the TUI command.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp();
        return;
    }

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    try runInteractive(allocator, &framework);
}

fn runInteractive(allocator: std.mem.Allocator, framework: *abi.Framework) !void {
    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();

    terminal.enter() catch {
        utils.output.printError("TUI requires an interactive terminal.", .{});
        utils.output.printInfo("Run directly from a terminal (not through pipes).", .{});
        return;
    };
    defer terminal.exit() catch {};

    const items = menuItems();
    var selected: usize = 0;

    while (true) {
        // Clear and render menu
        try terminal.clear();
        try renderMenu(&terminal, items, selected);

        // Read key input
        const key = try terminal.readKey();

        switch (key.code) {
            .ctrl_c => break,
            .character => {
                if (key.char) |ch| {
                    switch (ch) {
                        'q' => break,
                        'j' => {
                            if (selected + 1 < items.len) selected += 1;
                        },
                        'k' => {
                            if (selected > 0) selected -= 1;
                        },
                        else => {},
                    }
                }
            },
            .up => {
                if (selected > 0) selected -= 1;
            },
            .down => {
                if (selected + 1 < items.len) selected += 1;
            },
            .enter => {
                const action = items[selected].action;
                if (action == .quit) break;

                try terminal.exit();
                try runAction(allocator, framework, action);

                std.debug.print("\nPress Enter to return to menu...", .{});
                _ = terminal.readKey() catch {};

                try terminal.enter();
            },
            else => {},
        }
    }
}

fn renderMenu(terminal: *tui.Terminal, items: []const MenuItem, selected: usize) !void {
    // Header
    try terminal.write("\x1b[1;36mABI Framework Launcher\x1b[0m\n");
    try terminal.write("\x1b[2m────────────────────────────────────────\x1b[0m\n\n");

    // Menu items
    for (items, 0..) |item, i| {
        if (i == selected) {
            try terminal.write("\x1b[32m > \x1b[1m");
        } else {
            try terminal.write("   ");
        }

        try terminal.write(item.label);
        try terminal.write("\x1b[0m");

        // Padding
        const padding = 15 - @min(item.label.len, 15);
        for (0..padding) |_| {
            try terminal.write(" ");
        }

        try terminal.write("\x1b[2m");
        try terminal.write(item.description);
        try terminal.write("\x1b[0m\n");
    }

    // Footer
    try terminal.write("\n\x1b[2m[Enter] Run  [j/k] Navigate  [q] Quit\x1b[0m\n");
}

fn runAction(allocator: std.mem.Allocator, framework: *abi.Framework, action: Action) !void {
    _ = framework;
    std.debug.print("\n", .{});

    switch (action) {
        .command => |cmd| try runCommand(allocator, cmd),
        .version => utils.output.printInfo("ABI Framework v{s}", .{abi.version()}),
        .help => printHelp(),
        .quit => {},
    }
}

fn runCommand(allocator: std.mem.Allocator, cmd: Command) !void {
    switch (cmd) {
        .db => try db.run(allocator, empty_args),
        .agent => try agent.run(allocator, empty_args),
        .bench => try bench.run(allocator, empty_args),
        .config => try config.run(allocator, empty_args),
        .discord => try discord.run(allocator, empty_args),
        .embed => try embed.run(allocator, empty_args),
        .explore => try explore.run(allocator, empty_args),
        .gpu => try gpu.run(allocator, empty_args),
        .llm => try llm.run(allocator, empty_args),
        .network => try network.run(allocator, empty_args),
        .simd => try simd.run(allocator, empty_args),
        .system_info => try system_info.run(allocator, empty_args),
        .train => try train.run(allocator, empty_args),
    }
}

fn menuItems() []const MenuItem {
    return &[_]MenuItem{
        .{ .label = "Database", .description = "Manage vector database", .action = .{ .command = .db } },
        .{ .label = "AI Agent", .description = "Interactive AI assistant", .action = .{ .command = .agent } },
        .{ .label = "LLM", .description = "Local LLM inference", .action = .{ .command = .llm } },
        .{ .label = "Training", .description = "Run training pipelines", .action = .{ .command = .train } },
        .{ .label = "GPU", .description = "GPU devices and backends", .action = .{ .command = .gpu } },
        .{ .label = "Network", .description = "Cluster management", .action = .{ .command = .network } },
        .{ .label = "Explore", .description = "Search the codebase", .action = .{ .command = .explore } },
        .{ .label = "Config", .description = "Configuration management", .action = .{ .command = .config } },
        .{ .label = "SIMD", .description = "SIMD performance demo", .action = .{ .command = .simd } },
        .{ .label = "Discord", .description = "Discord bot integration", .action = .{ .command = .discord } },
        .{ .label = "System Info", .description = "System and framework status", .action = .{ .command = .system_info } },
        .{ .label = "Benchmarks", .description = "Performance benchmarks", .action = .{ .command = .bench } },
        .{ .label = "Help", .description = "Show CLI usage", .action = .help },
        .{ .label = "Version", .description = "Show version", .action = .version },
        .{ .label = "Quit", .description = "Exit the launcher", .action = .quit },
    };
}

fn printHelp() void {
    std.debug.print(
        \\Usage: abi tui
        \\
        \\Launch an interactive terminal UI to select and run ABI commands.
        \\
        \\Controls:
        \\  Up/Down, j/k    Navigate the menu
        \\  Enter           Run the selected command
        \\  q, Ctrl+C       Exit the TUI launcher
        \\
    , .{});
}
