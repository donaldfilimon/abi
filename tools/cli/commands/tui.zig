//! Interactive TUI command launcher for the ABI CLI.
//!
//! Usage:
//!   abi tui
//!
//! Keys:
//!   Up/Down, j/k  Navigate
//!   Mouse         Click to select, click again to run, wheel scrolls
//!   Type          Filter commands (Esc clears, Backspace deletes)
//!   a             Run with arguments
//!   Enter         Run selected command
//!   q             Quit
//!   Ctrl+C        Quit

const std = @import("std");
const abi = @import("abi");
const tui = @import("../tui/mod.zig");

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
    version,
    help,
    quit,
};

const header_lines: usize = 4;
const footer_lines: usize = 2;
const color_title = "\x1b[1m\x1b[36m";
const color_dim = "\x1b[90m";
const color_reset = "\x1b[0m";

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

const empty_args: []const [:0]const u8 = &[_][:0]const u8{};

pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);
    return runWithFramework(allocator, args, &framework);
}

pub fn runWithFramework(
    allocator: std.mem.Allocator,
    args: []const [:0]const u8,
    framework: *abi.Framework,
) !void {
    if (args.len > 0) {
        const command = std.mem.sliceTo(args[0], 0);
        if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
            printHelp();
            return;
        }
    }

    const items = menuItems();
    const label_width = maxLabelWidth(items);

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();

    terminal.enter() catch |err| {
        switch (err) {
            error.ConsoleModeFailed, error.ConsoleUnavailable => {
                std.debug.print("TUI requires an interactive terminal.\n", .{});
                std.debug.print("Run directly from a terminal (not through build systems or pipes).\n", .{});
                return;
            },
            else => return err,
        }
    };
    terminal.resetInput();
    defer terminal.exit() catch {};

    var selected: usize = 0;
    var scroll: usize = 0;
    var query_buf: [64]u8 = undefined;
    var query_len: usize = 0;
    var last_click: ?usize = null;

    while (true) {
        const query = query_buf[0..query_len];
        const filtered_len = filteredCount(items, query);
        if (filtered_len == 0) {
            selected = 0;
        } else if (selected >= filtered_len) {
            selected = filtered_len - 1;
        }
        const visible = adjustScroll(&terminal, filtered_len, selected, &scroll);
        try renderMenu(&terminal, items, selected, scroll, visible, label_width, query, filtered_len);

        const event = terminal.readEvent() catch break;

        switch (event) {
            .mouse => |mouse| {
                if (filtered_len == 0) continue;
                if (mouse.button == .wheel_up) {
                    if (selected > 0) selected -= 1;
                    continue;
                }
                if (mouse.button == .wheel_down) {
                    if (selected + 1 < filtered_len) selected += 1;
                    continue;
                }
                if (mouse.button != .left) continue;
                if (mouse.row <= header_lines) continue;
                const row_index = @as(usize, mouse.row) - header_lines - 1;
                if (row_index >= visible) continue;
                const new_selected = scroll + row_index;
                if (new_selected >= filtered_len) continue;
                selected = new_selected;
                if (mouse.pressed) {
                    last_click = selected;
                    continue;
                }
                if (last_click != null and last_click.? == selected) {
                    if (itemIndexForFiltered(items, query, selected)) |item_index| {
                        const should_quit = try runAction(&terminal, allocator, framework, items[item_index].action);
                        if (should_quit) break;
                        terminal.resetInput();
                    }
                }
                continue;
            },
            .key => |key| {
                if (key.code == .ctrl_c) break;

                if (key.code == .up or (query_len == 0 and tui.events.isChar(key, 'k'))) {
                    if (filtered_len > 0 and selected > 0) selected -= 1;
                    continue;
                }

                if (key.code == .down or (query_len == 0 and tui.events.isChar(key, 'j'))) {
                    if (filtered_len > 0 and selected + 1 < filtered_len) selected += 1;
                    continue;
                }

                if (query_len == 0 and tui.events.isChar(key, 'q')) break;

                if (key.code == .enter) {
                    if (filtered_len == 0) continue;
                    if (itemIndexForFiltered(items, query, selected)) |item_index| {
                        const should_quit = try runAction(&terminal, allocator, framework, items[item_index].action);
                        if (should_quit) break;
                        terminal.resetInput();
                    }
                    continue;
                }

                if (tui.events.isChar(key, 'a')) {
                    if (filtered_len == 0) continue;
                    if (itemIndexForFiltered(items, query, selected)) |item_index| {
                        const should_quit = try runActionWithArgs(
                            &terminal,
                            allocator,
                            framework,
                            items[item_index],
                        );
                        if (should_quit) break;
                        terminal.resetInput();
                    }
                    continue;
                }

                if (key.code == .backspace) {
                    if (query_len > 0) {
                        query_len -= 1;
                        selected = 0;
                        scroll = 0;
                    }
                    continue;
                }

                if (key.code == .escape) {
                    if (query_len > 0) {
                        query_len = 0;
                        selected = 0;
                        scroll = 0;
                    }
                    continue;
                }

                if (key.code == .character and key.char != null) {
                    const ch = key.char.?;
                    if (ch >= 32 and ch <= 126 and query_len < query_buf.len) {
                        query_buf[query_len] = ch;
                        query_len += 1;
                        selected = 0;
                        scroll = 0;
                    }
                    continue;
                }
            },
        }
    }
}

fn menuItems() []const MenuItem {
    return &[_]MenuItem{
        .{ .label = "db", .description = "Database operations (add, query, stats, optimize)", .action = .{ .command = .db } },
        .{ .label = "agent", .description = "Interactive AI agent session", .action = .{ .command = .agent } },
        .{ .label = "bench", .description = "Benchmark suites (simd, memory, ai)", .action = .{ .command = .bench } },
        .{ .label = "config", .description = "Configuration management", .action = .{ .command = .config } },
        .{ .label = "discord", .description = "Discord bot operations", .action = .{ .command = .discord } },
        .{ .label = "embed", .description = "Generate embeddings from text", .action = .{ .command = .embed } },
        .{ .label = "explore", .description = "Search and explore the codebase", .action = .{ .command = .explore } },
        .{ .label = "gpu", .description = "GPU backends and device summary", .action = .{ .command = .gpu } },
        .{ .label = "llm", .description = "LLM inference (info, generate, chat)", .action = .{ .command = .llm } },
        .{ .label = "network", .description = "Network registry management", .action = .{ .command = .network } },
        .{ .label = "simd", .description = "SIMD performance demo", .action = .{ .command = .simd } },
        .{ .label = "system-info", .description = "System and framework summary", .action = .{ .command = .system_info } },
        .{ .label = "train", .description = "Training pipeline (run, resume, info)", .action = .{ .command = .train } },
        .{ .label = "version", .description = "Show framework version", .action = .version },
        .{ .label = "help", .description = "Show TUI usage and shortcuts", .action = .help },
        .{ .label = "quit", .description = "Exit the interactive menu", .action = .quit },
    };
}

fn maxLabelWidth(items: []const MenuItem) usize {
    var width: usize = 0;
    for (items) |item| {
        width = @max(width, item.label.len);
    }
    return width;
}

fn filteredCount(items: []const MenuItem, query: []const u8) usize {
    if (query.len == 0) return items.len;
    var count: usize = 0;
    for (items) |item| {
        if (matchesQuery(item, query)) count += 1;
    }
    return count;
}

fn itemIndexForFiltered(items: []const MenuItem, query: []const u8, filtered_index: usize) ?usize {
    if (query.len == 0) {
        return if (filtered_index < items.len) filtered_index else null;
    }
    var idx: usize = 0;
    for (items, 0..) |item, item_index| {
        if (!matchesQuery(item, query)) continue;
        if (idx == filtered_index) return item_index;
        idx += 1;
    }
    return null;
}

fn matchesQuery(item: MenuItem, query: []const u8) bool {
    return containsIgnoreCase(item.label, query) or containsIgnoreCase(item.description, query);
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var j: usize = 0;
        var matched = true;
        while (j < needle.len) : (j += 1) {
            const a = std.ascii.toLower(haystack[i + j]);
            const b = std.ascii.toLower(needle[j]);
            if (a != b) {
                matched = false;
                break;
            }
        }
        if (matched) return true;
    }
    return false;
}

fn adjustScroll(
    terminal: *tui.Terminal,
    item_count: usize,
    selected: usize,
    scroll: *usize,
) usize {
    const size = terminal.size();
    const rows = @as(usize, size.rows);
    const visible = if (rows > header_lines + footer_lines) rows - header_lines - footer_lines else 1;

    if (selected < scroll.*) {
        scroll.* = selected;
    } else if (selected >= scroll.* + visible) {
        scroll.* = selected - visible + 1;
    }

    if (item_count <= visible) {
        scroll.* = 0;
    } else if (scroll.* + visible > item_count) {
        scroll.* = item_count - visible;
    }

    return visible;
}

fn renderMenu(
    terminal: *tui.Terminal,
    items: []const MenuItem,
    selected: usize,
    scroll: usize,
    visible: usize,
    label_width: usize,
    query: []const u8,
    filtered_len: usize,
) !void {
    const size = terminal.size();
    const max_cols = @as(usize, size.cols);
    var render_storage: [16384]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&render_storage);
    const alloc = fba.allocator();
    var buffer: std.ArrayListUnmanaged(u8) = .empty;
    defer buffer.deinit(alloc);

    try buffer.appendSlice(alloc, "\x1b[2J\x1b[H");
    try buffer.appendSlice(alloc, color_title);
    try buffer.appendSlice(alloc, "ABI Interactive CLI");
    try buffer.appendSlice(alloc, color_reset);
    try buffer.append(alloc, '\n');
    var filter_buf: [160]u8 = undefined;
    const filter_line = if (query.len == 0)
        "Filter: (type to filter, Esc clears)"
    else
        std.fmt.bufPrint(
            &filter_buf,
            "Filter: {s} ({d}/{d})",
            .{ query, filtered_len, items.len },
        ) catch "Filter:";
    try buffer.appendSlice(alloc, color_dim);
    try buffer.appendSlice(alloc, filter_line);
    try buffer.appendSlice(alloc, color_reset);
    try buffer.append(alloc, '\n');
    try buffer.appendSlice(alloc, color_dim);
    try buffer.appendSlice(alloc, "Use Up/Down or j/k to navigate, Enter to run, q to quit.");
    try buffer.appendSlice(alloc, color_reset);
    try buffer.append(alloc, '\n');
    try appendLine(&buffer, alloc, "");

    var shown: usize = 0;
    var filtered_index: usize = 0;
    var selected_label: []const u8 = "";
    var selected_index: ?usize = null;

    for (items, 0..) |item, item_index| {
        if (query.len > 0 and !matchesQuery(item, query)) continue;
        if (filtered_index == selected) {
            selected_label = item.label;
            selected_index = item_index;
        }
        if (filtered_index >= scroll and shown < visible) {
            const is_selected = filtered_index == selected;
            if (is_selected) {
                try buffer.appendSlice(alloc, "\x1b[7m");
            }

            try buffer.appendSlice(alloc, if (is_selected) "> " else "  ");
            try buffer.appendSlice(alloc, item.label);
            const pad = if (label_width > item.label.len) label_width - item.label.len else 0;
            var pad_i: usize = 0;
            while (pad_i < pad) : (pad_i += 1) {
                try buffer.append(alloc, ' ');
            }
            try buffer.appendSlice(alloc, "  ");

            const fixed_len = 2 + label_width + 2;
            const desc_max = if (max_cols > fixed_len) max_cols - fixed_len else 0;
            const desc = if (desc_max > 0 and item.description.len > desc_max)
                item.description[0..desc_max]
            else
                item.description;
            if (desc.len > 0) {
                try buffer.appendSlice(alloc, desc);
            }

            if (is_selected) {
                try buffer.appendSlice(alloc, "\x1b[0m");
            }
            try buffer.append(alloc, '\n');
            shown += 1;
        }
        filtered_index += 1;
    }

    if (filtered_len == 0) {
        try appendLine(&buffer, alloc, "  No matches.");
    }

    try buffer.append(alloc, '\n');
    var footer_buf: [160]u8 = undefined;

    // Build pagination info
    const start_item = if (filtered_len > 0) scroll + 1 else 0;
    const end_item = if (filtered_len > 0) @min(scroll + visible, filtered_len) else 0;

    const footer = if (selected_index != null)
        std.fmt.bufPrint(&footer_buf, "Showing {d}-{d} of {d} | Selected: {s}", .{
            start_item,
            end_item,
            filtered_len,
            selected_label,
        }) catch "Selected:"
    else if (filtered_len == 0)
        std.fmt.bufPrint(&footer_buf, "Showing 0 of 0 | Selected: (none)", .{}) catch "Selected: (none)"
    else
        std.fmt.bufPrint(&footer_buf, "Showing {d}-{d} of {d} | Selected: (none)", .{
            start_item,
            end_item,
            filtered_len,
        }) catch "Selected: (none)";
    try appendLine(&buffer, alloc, footer);

    try terminal.write(buffer.items);
}

fn appendLine(buffer: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, line: []const u8) !void {
    if (line.len > 0) {
        try buffer.appendSlice(allocator, line);
    }
    try buffer.append(allocator, '\n');
}

fn runAction(
    terminal: *tui.Terminal,
    allocator: std.mem.Allocator,
    framework: *abi.Framework,
    action: Action,
) !bool {
    switch (action) {
        .quit => return true,
        else => {},
    }

    try terminal.exit();
    defer terminal.enter() catch {};
    terminal.resetInput();

    switch (action) {
        .command => |cmd| runCommand(allocator, framework, cmd) catch |err| {
            std.debug.print("Command failed: {t}\n", .{err});
        },
        .version => std.debug.print("ABI Framework v{s}\n", .{abi.version()}),
        .help => printHelp(),
        .quit => {},
    }

    waitForEnter(terminal);
    return false;
}

fn runActionWithArgs(
    terminal: *tui.Terminal,
    allocator: std.mem.Allocator,
    framework: *abi.Framework,
    item: MenuItem,
) !bool {
    switch (item.action) {
        .command => |cmd| {
            try terminal.exit();
            defer terminal.enter() catch {};
            terminal.resetInput();

            std.debug.print("Args for {s} (empty for none): ", .{item.label});
            const line = try readLineOwned(terminal, allocator);
            defer if (line) |owned| allocator.free(owned);

            var parsed = try parseArgsLine(allocator, line orelse "");
            defer parsed.deinit();

            runCommandWithArgs(allocator, framework, cmd, parsed.args.items) catch |err| {
                std.debug.print("Command failed: {t}\n", .{err});
            };

            waitForEnter(terminal);
            return false;
        },
        else => return runAction(terminal, allocator, framework, item.action),
    }
}

fn runCommand(allocator: std.mem.Allocator, framework: *abi.Framework, cmd: Command) !void {
    try runCommandWithArgs(allocator, framework, cmd, empty_args);
}

fn runCommandWithArgs(
    allocator: std.mem.Allocator,
    framework: *abi.Framework,
    cmd: Command,
    args: []const [:0]const u8,
) !void {
    switch (cmd) {
        .db => try db.run(allocator, args),
        .agent => try agent.run(allocator, args),
        .bench => try bench.run(allocator, args),
        .config => try config.run(allocator, args),
        .discord => try discord.run(allocator, args),
        .embed => try embed.run(allocator, args),
        .explore => try explore.run(allocator, args),
        .gpu => try gpu.run(allocator, args),
        .llm => try llm.run(allocator, args),
        .network => try network.run(allocator, args),
        .simd => try simd.run(allocator),
        .system_info => try system_info.run(allocator, framework),
        .train => try train.run(allocator, args),
    }
}

const ParsedArgs = struct {
    allocator: std.mem.Allocator,
    args: std.ArrayListUnmanaged([:0]const u8) = .empty,

    pub fn deinit(self: *ParsedArgs) void {
        for (self.args.items) |arg| {
            self.allocator.free(arg);
        }
        self.args.deinit(self.allocator);
    }
};

fn parseArgsLine(allocator: std.mem.Allocator, line: []const u8) !ParsedArgs {
    var parsed = ParsedArgs{ .allocator = allocator };
    errdefer parsed.deinit();

    var token: std.ArrayListUnmanaged(u8) = .empty;
    defer token.deinit(allocator);

    var in_single = false;
    var in_double = false;
    var escaped = false;

    for (line) |ch| {
        if (escaped) {
            try token.append(allocator, ch);
            escaped = false;
            continue;
        }

        if (!in_single and ch == '\\') {
            escaped = true;
            continue;
        }

        if (!in_double and ch == '\'') {
            in_single = !in_single;
            continue;
        }

        if (!in_single and ch == '"') {
            in_double = !in_double;
            continue;
        }

        if (!in_single and !in_double and std.ascii.isWhitespace(ch)) {
            if (token.items.len > 0) {
                try appendToken(&parsed, allocator, &token);
            }
            continue;
        }

        try token.append(allocator, ch);
    }

    if (token.items.len > 0) {
        try appendToken(&parsed, allocator, &token);
    }

    return parsed;
}

fn appendToken(parsed: *ParsedArgs, allocator: std.mem.Allocator, token: *std.ArrayListUnmanaged(u8)) !void {
    const owned = try allocator.dupeZ(u8, token.items);
    try parsed.args.append(allocator, owned);
    token.clearRetainingCapacity();
}

fn readLineOwned(terminal: *tui.Terminal, allocator: std.mem.Allocator) !?[]u8 {
    const io = terminal.io_backend.io();
    var buffer: [2048]u8 = undefined;
    var reader = terminal.stdin_file.reader(io, &buffer);
    const line_opt = reader.interface.takeDelimiter('\n') catch return null;
    const line = line_opt orelse return null;
    const trimmed = std.mem.trim(u8, line, " \t\r\n");
    if (trimmed.len == 0) return null;
    const duped = try allocator.dupe(u8, trimmed);
    return duped;
}

fn waitForEnter(terminal: *tui.Terminal) void {
    std.debug.print("\nPress Enter to return to the menu...", .{});
    const io = terminal.io_backend.io();
    var buffer: [128]u8 = undefined;
    var reader = terminal.stdin_file.reader(io, &buffer);
    _ = reader.interface.takeDelimiter('\n') catch {};
    std.debug.print("\n", .{});
}

fn printHelp() void {
    const help_text =
        \\ABI Interactive TUI
        \\
        \\Usage:
        \\  abi tui
        \\
        \\Keys:
        \\  Up/Down, j/k  Navigate menu
        \\  Mouse         Click to select, click again to run, wheel scrolls
        \\  Type          Filter commands (Esc clears, Backspace deletes)
        \\  a             Run selected command with arguments
        \\  Enter         Run selected command
        \\  q             Quit
        \\  Ctrl+C        Quit
        \\
        \\Note: Commands run in normal terminal mode. Press Enter to return to the menu.
    ;

    std.debug.print("{s}\n", .{help_text});
}
