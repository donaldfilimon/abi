//! Interactive TUI command launcher.
//!
//! Provides a terminal-based interface for selecting and running ABI CLI commands.
//! Features: categories, search/filter, quick launch (1-9), mouse support, status bar.

const std = @import("std");
const builtin = @import("builtin");
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
const task = @import("task.zig");

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

const Category = enum {
    ai,
    data,
    system,
    tools,
    meta,

    fn icon(self: Category) []const u8 {
        return switch (self) {
            .ai => "🤖",
            .data => "💾",
            .system => "⚙️",
            .tools => "🔧",
            .meta => "📋",
        };
    }

    fn name(self: Category) []const u8 {
        return switch (self) {
            .ai => "AI & ML",
            .data => "Data",
            .system => "System",
            .tools => "Tools",
            .meta => "Meta",
        };
    }

    fn color(self: Category) []const u8 {
        return switch (self) {
            .ai => "\x1b[35m", // Magenta
            .data => "\x1b[34m", // Blue
            .system => "\x1b[33m", // Yellow
            .tools => "\x1b[32m", // Green
            .meta => "\x1b[36m", // Cyan
        };
    }
};

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
    task,
};

const MenuItem = struct {
    label: []const u8,
    description: []const u8,
    action: Action,
    category: Category,
    shortcut: ?u8 = null, // Quick launch key (1-9)
};

const TuiState = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    framework: *abi.Framework,
    items: []const MenuItem,
    filtered_indices: std.ArrayListUnmanaged(usize),
    selected: usize,
    scroll_offset: usize,
    search_mode: bool,
    search_buffer: [64]u8,
    search_len: usize,
    visible_rows: usize,
    term_size: tui.TerminalSize,

    fn init(allocator: std.mem.Allocator, terminal: *tui.Terminal, framework: *abi.Framework) !TuiState {
        var state = TuiState{
            .allocator = allocator,
            .terminal = terminal,
            .framework = framework,
            .items = menuItems(),
            .filtered_indices = .empty,
            .selected = 0,
            .scroll_offset = 0,
            .search_mode = false,
            .search_buffer = undefined,
            .search_len = 0,
            .visible_rows = 10,
            .term_size = terminal.size(),
        };
        // Initialize with all items
        try state.resetFilter();
        return state;
    }

    fn deinit(self: *TuiState) void {
        self.filtered_indices.deinit(self.allocator);
    }

    fn resetFilter(self: *TuiState) !void {
        self.filtered_indices.clearRetainingCapacity();
        for (self.items, 0..) |_, i| {
            try self.filtered_indices.append(self.allocator, i);
        }
        self.selected = 0;
        self.scroll_offset = 0;
    }

    fn applyFilter(self: *TuiState) !void {
        self.filtered_indices.clearRetainingCapacity();
        const query = self.search_buffer[0..self.search_len];

        for (self.items, 0..) |item, i| {
            if (query.len == 0 or containsIgnoreCase(item.label, query) or containsIgnoreCase(item.description, query)) {
                try self.filtered_indices.append(self.allocator, i);
            }
        }

        if (self.selected >= self.filtered_indices.items.len) {
            self.selected = if (self.filtered_indices.items.len > 0) self.filtered_indices.items.len - 1 else 0;
        }
        self.scroll_offset = 0;
    }

    fn selectedItem(self: *const TuiState) ?*const MenuItem {
        if (self.filtered_indices.items.len == 0) return null;
        const idx = self.filtered_indices.items[self.selected];
        return &self.items[idx];
    }

    fn moveUp(self: *TuiState) void {
        if (self.selected > 0) {
            self.selected -= 1;
            if (self.selected < self.scroll_offset) {
                self.scroll_offset = self.selected;
            }
        }
    }

    fn moveDown(self: *TuiState) void {
        if (self.selected + 1 < self.filtered_indices.items.len) {
            self.selected += 1;
            if (self.selected >= self.scroll_offset + self.visible_rows) {
                self.scroll_offset = self.selected - self.visible_rows + 1;
            }
        }
    }

    fn pageUp(self: *TuiState) void {
        if (self.selected >= self.visible_rows) {
            self.selected -= self.visible_rows;
        } else {
            self.selected = 0;
        }
        if (self.selected < self.scroll_offset) {
            self.scroll_offset = self.selected;
        }
    }

    fn pageDown(self: *TuiState) void {
        const max = self.filtered_indices.items.len;
        if (max == 0) return;
        if (self.selected + self.visible_rows < max) {
            self.selected += self.visible_rows;
        } else {
            self.selected = max - 1;
        }
        if (self.selected >= self.scroll_offset + self.visible_rows) {
            self.scroll_offset = self.selected - self.visible_rows + 1;
        }
    }

    fn goHome(self: *TuiState) void {
        self.selected = 0;
        self.scroll_offset = 0;
    }

    fn goEnd(self: *TuiState) void {
        if (self.filtered_indices.items.len > 0) {
            self.selected = self.filtered_indices.items.len - 1;
            if (self.selected >= self.visible_rows) {
                self.scroll_offset = self.selected - self.visible_rows + 1;
            }
        }
    }

    fn handleMouseClick(self: *TuiState, row: u16) bool {
        // Account for header (4 rows: title, border, blank, category header)
        const header_rows: u16 = 4;
        if (row < header_rows) return false;

        const menu_row = row - header_rows;
        const clicked_idx = self.scroll_offset + @as(usize, menu_row);

        if (clicked_idx < self.filtered_indices.items.len) {
            self.selected = clicked_idx;
            return true; // Indicate selection changed, could trigger action
        }
        return false;
    }
};

const empty_args = &[_][:0]const u8{};

// ═══════════════════════════════════════════════════════════════════════════
// Box Drawing Characters
// ═══════════════════════════════════════════════════════════════════════════

const box = struct {
    const tl = "╭"; // Top-left
    const tr = "╮"; // Top-right
    const bl = "╰"; // Bottom-left
    const br = "╯"; // Bottom-right
    const h = "─"; // Horizontal
    const v = "│"; // Vertical
    const lsep = "├"; // Left separator
    const rsep = "┤"; // Right separator
};

// ═══════════════════════════════════════════════════════════════════════════
// Colors
// ═══════════════════════════════════════════════════════════════════════════

const colors = struct {
    const reset = "\x1b[0m";
    const bold = "\x1b[1m";
    const dim = "\x1b[2m";
    const italic = "\x1b[3m";
    const underline = "\x1b[4m";

    const black = "\x1b[30m";
    const red = "\x1b[31m";
    const green = "\x1b[32m";
    const yellow = "\x1b[33m";
    const blue = "\x1b[34m";
    const magenta = "\x1b[35m";
    const cyan = "\x1b[36m";
    const white = "\x1b[37m";

    const bg_black = "\x1b[40m";
    const bg_blue = "\x1b[44m";
    const bg_cyan = "\x1b[46m";

    const bright_black = "\x1b[90m";
    const bright_white = "\x1b[97m";
};

// ═══════════════════════════════════════════════════════════════════════════
// Entry Point
// ═══════════════════════════════════════════════════════════════════════════

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

    var state = try TuiState.init(allocator, &terminal, framework);
    defer state.deinit();

    // Calculate visible rows based on terminal size
    state.term_size = terminal.size();
    state.visible_rows = @max(5, state.term_size.rows -| 10);

    while (true) {
        // Refresh terminal size
        state.term_size = terminal.size();
        state.visible_rows = @max(5, state.term_size.rows -| 10);

        // Clear and render
        try terminal.clear();
        try renderFrame(&state);

        // Read input
        const event = try terminal.readEvent();

        switch (event) {
            .key => |key| {
                if (try handleKeyEvent(&state, key)) break;
            },
            .mouse => |mouse| {
                if (mouse.pressed and mouse.button == .left) {
                    if (state.handleMouseClick(mouse.row)) {
                        // Double-click could run the command
                    }
                } else if (mouse.button == .wheel_up) {
                    state.moveUp();
                } else if (mouse.button == .wheel_down) {
                    state.moveDown();
                }
            },
        }
    }
}

fn handleKeyEvent(state: *TuiState, key: tui.Key) !bool {
    if (state.search_mode) {
        return try handleSearchKey(state, key);
    }

    switch (key.code) {
        .ctrl_c => return true,
        .escape => {
            if (state.search_len > 0) {
                state.search_len = 0;
                try state.resetFilter();
            }
        },
        .character => {
            if (key.char) |ch| {
                switch (ch) {
                    'q' => return true,
                    'j' => state.moveDown(),
                    'k' => state.moveUp(),
                    '/' => {
                        state.search_mode = true;
                    },
                    'g' => state.goHome(),
                    'G' => state.goEnd(),
                    '1'...'9' => {
                        // Quick launch
                        const num = ch - '0';
                        if (findByShortcut(state.items, num)) |idx| {
                            try state.terminal.exit();
                            try runAction(state.allocator, state.framework, state.items[idx].action);
                            std.debug.print("\n{s}Press Enter to return to menu...{s}", .{ colors.dim, colors.reset });
                            _ = state.terminal.readKey() catch {};
                            try state.terminal.enter();
                        }
                    },
                    else => {},
                }
            }
        },
        .up => state.moveUp(),
        .down => state.moveDown(),
        .page_up => state.pageUp(),
        .page_down => state.pageDown(),
        .home => state.goHome(),
        .end => state.goEnd(),
        .enter => {
            if (state.selectedItem()) |item| {
                if (item.action == .quit) return true;

                try state.terminal.exit();
                try runAction(state.allocator, state.framework, item.action);

                std.debug.print("\n{s}Press Enter to return to menu...{s}", .{ colors.dim, colors.reset });
                _ = state.terminal.readKey() catch {};

                try state.terminal.enter();
            }
        },
        else => {},
    }
    return false;
}

fn handleSearchKey(state: *TuiState, key: tui.Key) !bool {
    switch (key.code) {
        .ctrl_c => return true,
        .escape => {
            state.search_mode = false;
        },
        .enter => {
            state.search_mode = false;
            // Execute selected if any
            if (state.selectedItem()) |item| {
                if (item.action == .quit) return true;

                try state.terminal.exit();
                try runAction(state.allocator, state.framework, item.action);

                std.debug.print("\n{s}Press Enter to return to menu...{s}", .{ colors.dim, colors.reset });
                _ = state.terminal.readKey() catch {};

                try state.terminal.enter();
            }
        },
        .backspace => {
            if (state.search_len > 0) {
                state.search_len -= 1;
                try state.applyFilter();
            }
        },
        .character => {
            if (key.char) |ch| {
                if (state.search_len < state.search_buffer.len) {
                    state.search_buffer[state.search_len] = ch;
                    state.search_len += 1;
                    try state.applyFilter();
                }
            }
        },
        .up => state.moveUp(),
        .down => state.moveDown(),
        else => {},
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Rendering
// ═══════════════════════════════════════════════════════════════════════════

fn renderFrame(state: *TuiState) !void {
    const term = state.terminal;
    const width: usize = @min(80, state.term_size.cols);

    // Title bar
    try renderTitleBar(term, width);

    // Search bar (if active or has content)
    if (state.search_mode or state.search_len > 0) {
        try renderSearchBar(term, state, width);
    }

    // Menu items
    try renderMenuItems(term, state, width);

    // Status bar
    try renderStatusBar(term, state, width);

    // Help bar
    try renderHelpBar(term, state, width);
}

fn renderTitleBar(term: *tui.Terminal, width: usize) !void {
    // Top border
    try term.write(colors.cyan);
    try term.write(box.tl);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.tr);
    try term.write(colors.reset);
    try term.write("\n");

    // Title
    try term.write(colors.cyan);
    try term.write(box.v);
    try term.write(colors.reset);

    const title = " ABI Framework ";
    const version_str = abi.version();
    const title_len = title.len + version_str.len + 3; // " vX.X.X"
    const left_pad = (width - 2 - title_len) / 2;
    const right_pad = width - 2 - title_len - left_pad;

    try writeRepeat(term, " ", left_pad);
    try term.write(colors.bold);
    try term.write(colors.cyan);
    try term.write(title);
    try term.write(colors.dim);
    try term.write("v");
    try term.write(version_str);
    try term.write(colors.reset);
    try writeRepeat(term, " ", right_pad);

    try term.write(colors.cyan);
    try term.write(box.v);
    try term.write(colors.reset);
    try term.write("\n");

    // Separator
    try term.write(colors.cyan);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.rsep);
    try term.write(colors.reset);
    try term.write("\n");
}

fn renderSearchBar(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    try term.write(colors.cyan);
    try term.write(box.v);
    try term.write(colors.reset);

    try term.write(" ");
    if (state.search_mode) {
        try term.write(colors.yellow);
    } else {
        try term.write(colors.dim);
    }
    try term.write("/");
    try term.write(colors.reset);
    try term.write(" ");

    const query = state.search_buffer[0..state.search_len];
    try term.write(query);

    if (state.search_mode) {
        try term.write(colors.yellow);
        try term.write("_");
        try term.write(colors.reset);
    }

    const used = 4 + query.len + @as(usize, if (state.search_mode) 1 else 0);
    if (used < width - 1) {
        try writeRepeat(term, " ", width - 1 - used);
    }

    try term.write(colors.cyan);
    try term.write(box.v);
    try term.write(colors.reset);
    try term.write("\n");

    // Separator
    try term.write(colors.cyan);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.rsep);
    try term.write(colors.reset);
    try term.write("\n");
}

fn renderMenuItems(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const items = state.items;
    const indices = state.filtered_indices.items;
    const visible = state.visible_rows;
    const start = state.scroll_offset;
    const end = @min(start + visible, indices.len);

    // Show scroll indicator if needed
    if (start > 0) {
        try term.write(colors.cyan);
        try term.write(box.v);
        try term.write(colors.reset);
        try term.write(colors.dim);
        try term.write("   ▲ more above");
        try writeRepeat(term, " ", width - 18);
        try term.write(colors.reset);
        try term.write(colors.cyan);
        try term.write(box.v);
        try term.write(colors.reset);
        try term.write("\n");
    }

    for (start..end) |i| {
        const idx = indices[i];
        const item = items[idx];
        const is_selected = i == state.selected;

        try term.write(colors.cyan);
        try term.write(box.v);
        try term.write(colors.reset);

        if (is_selected) {
            try term.write(colors.bg_blue);
            try term.write(colors.bright_white);
            try term.write(" ▸ ");
        } else {
            try term.write("   ");
        }

        // Shortcut number
        if (item.shortcut) |num| {
            try term.write(colors.dim);
            var buf: [2]u8 = undefined;
            buf[0] = '0' + num;
            buf[1] = 0;
            try term.write(buf[0..1]);
            try term.write(colors.reset);
            if (is_selected) {
                try term.write(colors.bg_blue);
                try term.write(colors.bright_white);
            }
            try term.write(" ");
        } else {
            try term.write("  ");
        }

        // Category color
        try term.write(item.category.color());
        if (is_selected) try term.write(colors.bold);
        try term.write(item.label);
        try term.write(colors.reset);
        if (is_selected) {
            try term.write(colors.bg_blue);
            try term.write(colors.bright_white);
        }

        // Padding
        const label_len = item.label.len;
        const padding = 16 -| @min(label_len, 16);
        try writeRepeat(term, " ", padding);

        // Description
        try term.write(colors.dim);
        if (is_selected) try term.write(colors.bright_white);
        const desc_max = width -| 24;
        const desc_len = @min(item.description.len, desc_max);
        try term.write(item.description[0..desc_len]);
        try term.write(colors.reset);

        // Fill rest
        const used = 6 + label_len + padding + desc_len;
        if (used < width - 1) {
            try writeRepeat(term, " ", width - 1 - used);
        }

        try term.write(colors.cyan);
        try term.write(box.v);
        try term.write(colors.reset);
        try term.write("\n");
    }

    // Show scroll indicator if needed
    if (end < indices.len) {
        try term.write(colors.cyan);
        try term.write(box.v);
        try term.write(colors.reset);
        try term.write(colors.dim);
        try term.write("   ▼ more below");
        try writeRepeat(term, " ", width - 18);
        try term.write(colors.reset);
        try term.write(colors.cyan);
        try term.write(box.v);
        try term.write(colors.reset);
        try term.write("\n");
    }

    // Fill empty rows if menu is shorter than visible area
    const rendered = end - start + @as(usize, if (start > 0) 1 else 0) + @as(usize, if (end < indices.len) 1 else 0);
    if (rendered < visible) {
        for (0..(visible - rendered)) |_| {
            try term.write(colors.cyan);
            try term.write(box.v);
            try term.write(colors.reset);
            try writeRepeat(term, " ", width - 2);
            try term.write(colors.cyan);
            try term.write(box.v);
            try term.write(colors.reset);
            try term.write("\n");
        }
    }
}

fn renderStatusBar(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    // Separator
    try term.write(colors.cyan);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.rsep);
    try term.write(colors.reset);
    try term.write("\n");

    // Status content
    try term.write(colors.cyan);
    try term.write(box.v);
    try term.write(colors.reset);
    try term.write(" ");

    // OS info
    const os_name = switch (builtin.os.tag) {
        .windows => "Windows",
        .linux => "Linux",
        .macos => "macOS",
        else => "Unknown",
    };
    try term.write(colors.dim);
    try term.write(os_name);

    // Item count
    var count_buf: [32]u8 = undefined;
    const count_str = std.fmt.bufPrint(&count_buf, " │ {d}/{d} items", .{
        state.filtered_indices.items.len,
        state.items.len,
    }) catch "?";
    try term.write(count_str);
    try term.write(colors.reset);

    const used = 2 + os_name.len + count_str.len;
    if (used < width - 1) {
        try writeRepeat(term, " ", width - 1 - used);
    }

    try term.write(colors.cyan);
    try term.write(box.v);
    try term.write(colors.reset);
    try term.write("\n");
}

fn renderHelpBar(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    // Bottom border with help
    try term.write(colors.cyan);
    try term.write(box.bl);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.br);
    try term.write(colors.reset);
    try term.write("\n");

    // Help text
    try term.write(" ");
    if (state.search_mode) {
        try term.write(colors.dim);
        try term.write("Type to filter │ ");
        try term.write(colors.reset);
        try term.write(colors.yellow);
        try term.write("Enter");
        try term.write(colors.reset);
        try term.write(colors.dim);
        try term.write(" Run │ ");
        try term.write(colors.reset);
        try term.write(colors.yellow);
        try term.write("Esc");
        try term.write(colors.reset);
        try term.write(colors.dim);
        try term.write(" Cancel");
        try term.write(colors.reset);
    } else {
        try term.write(colors.yellow);
        try term.write("Enter");
        try term.write(colors.reset);
        try term.write(colors.dim);
        try term.write(" Run │ ");
        try term.write(colors.reset);
        try term.write(colors.yellow);
        try term.write("j/k");
        try term.write(colors.reset);
        try term.write(colors.dim);
        try term.write(" Nav │ ");
        try term.write(colors.reset);
        try term.write(colors.yellow);
        try term.write("/");
        try term.write(colors.reset);
        try term.write(colors.dim);
        try term.write(" Search │ ");
        try term.write(colors.reset);
        try term.write(colors.yellow);
        try term.write("1-9");
        try term.write(colors.reset);
        try term.write(colors.dim);
        try term.write(" Quick │ ");
        try term.write(colors.reset);
        try term.write(colors.yellow);
        try term.write("q");
        try term.write(colors.reset);
        try term.write(colors.dim);
        try term.write(" Quit");
        try term.write(colors.reset);
    }
    try term.write("\n");
}

fn writeRepeat(term: *tui.Terminal, char: []const u8, count: usize) !void {
    for (0..count) |_| {
        try term.write(char);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Actions
// ═══════════════════════════════════════════════════════════════════════════

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
        .task => try task.run(allocator, empty_args),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Menu Items
// ═══════════════════════════════════════════════════════════════════════════

fn menuItems() []const MenuItem {
    return &[_]MenuItem{
        // AI & ML (shortcuts 1-3)
        .{ .label = "AI Agent", .description = "Interactive AI assistant", .action = .{ .command = .agent }, .category = .ai, .shortcut = 1 },
        .{ .label = "LLM", .description = "Local LLM inference", .action = .{ .command = .llm }, .category = .ai, .shortcut = 2 },
        .{ .label = "Training", .description = "Run training pipelines", .action = .{ .command = .train }, .category = .ai, .shortcut = 3 },
        .{ .label = "Embeddings", .description = "Generate embeddings", .action = .{ .command = .embed }, .category = .ai },

        // Data (shortcuts 4-5)
        .{ .label = "Database", .description = "Manage vector database", .action = .{ .command = .db }, .category = .data, .shortcut = 4 },
        .{ .label = "Explore", .description = "Search the codebase", .action = .{ .command = .explore }, .category = .data, .shortcut = 5 },

        // System (shortcuts 6-7)
        .{ .label = "GPU", .description = "GPU devices and backends", .action = .{ .command = .gpu }, .category = .system, .shortcut = 6 },
        .{ .label = "Network", .description = "Cluster management", .action = .{ .command = .network }, .category = .system, .shortcut = 7 },
        .{ .label = "System Info", .description = "System and framework status", .action = .{ .command = .system_info }, .category = .system },

        // Tools (shortcuts 8-9)
        .{ .label = "Benchmarks", .description = "Performance benchmarks", .action = .{ .command = .bench }, .category = .tools, .shortcut = 8 },
        .{ .label = "SIMD", .description = "SIMD performance demo", .action = .{ .command = .simd }, .category = .tools, .shortcut = 9 },
        .{ .label = "Config", .description = "Configuration management", .action = .{ .command = .config }, .category = .tools },
        .{ .label = "Tasks", .description = "Task management", .action = .{ .command = .task }, .category = .tools },
        .{ .label = "Discord", .description = "Discord bot integration", .action = .{ .command = .discord }, .category = .tools },

        // Meta
        .{ .label = "Help", .description = "Show CLI usage", .action = .help, .category = .meta },
        .{ .label = "Version", .description = "Show version", .action = .version, .category = .meta },
        .{ .label = "Quit", .description = "Exit the launcher", .action = .quit, .category = .meta },
    };
}

fn findByShortcut(items: []const MenuItem, num: u8) ?usize {
    for (items, 0..) |item, i| {
        if (item.shortcut) |s| {
            if (s == num) return i;
        }
    }
    return null;
}

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;

    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var match = true;
        for (needle, 0..) |nc, j| {
            const hc = haystack[i + j];
            if (toLower(hc) != toLower(nc)) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

fn toLower(c: u8) u8 {
    if (c >= 'A' and c <= 'Z') return c + 32;
    return c;
}

fn printHelp() void {
    std.debug.print(
        \\{s}Usage:{s} abi tui
        \\
        \\Launch an interactive terminal UI to select and run ABI commands.
        \\
        \\{s}Navigation:{s}
        \\  Up/Down, j/k    Navigate the menu
        \\  PgUp/PgDn       Page navigation
        \\  Home/End, g/G   Jump to first/last
        \\  Mouse wheel     Scroll the menu
        \\  Mouse click     Select item
        \\
        \\{s}Actions:{s}
        \\  Enter           Run the selected command
        \\  1-9             Quick launch (numbered items)
        \\  /               Search/filter commands
        \\  Esc             Clear search
        \\  q, Ctrl+C       Exit the TUI launcher
        \\
    , .{
        colors.bold,
        colors.reset,
        colors.bold,
        colors.reset,
        colors.bold,
        colors.reset,
    });
}
