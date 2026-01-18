//! TUI Widget Components
//!
//! Provides reusable UI components for the TUI:
//! - Progress indicators (spinner, bar)
//! - Dialog boxes (confirm, info, preview)
//! - Notification toasts

const std = @import("std");
const terminal = @import("terminal.zig");

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

pub const colors = struct {
    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";
    pub const italic = "\x1b[3m";
    pub const underline = "\x1b[4m";
    pub const blink = "\x1b[5m";

    pub const black = "\x1b[30m";
    pub const red = "\x1b[31m";
    pub const green = "\x1b[32m";
    pub const yellow = "\x1b[33m";
    pub const blue = "\x1b[34m";
    pub const magenta = "\x1b[35m";
    pub const cyan = "\x1b[36m";
    pub const white = "\x1b[37m";

    pub const bg_black = "\x1b[40m";
    pub const bg_red = "\x1b[41m";
    pub const bg_green = "\x1b[42m";
    pub const bg_yellow = "\x1b[43m";
    pub const bg_blue = "\x1b[44m";
    pub const bg_magenta = "\x1b[45m";
    pub const bg_cyan = "\x1b[46m";
    pub const bg_white = "\x1b[47m";

    pub const bright_black = "\x1b[90m";
    pub const bright_red = "\x1b[91m";
    pub const bright_green = "\x1b[92m";
    pub const bright_yellow = "\x1b[93m";
    pub const bright_blue = "\x1b[94m";
    pub const bright_magenta = "\x1b[95m";
    pub const bright_cyan = "\x1b[96m";
    pub const bright_white = "\x1b[97m";
};

pub const box = struct {
    pub const tl = "╭";
    pub const tr = "╮";
    pub const bl = "╰";
    pub const br = "╯";
    pub const h = "─";
    pub const v = "│";
    pub const lsep = "├";
    pub const rsep = "┤";

    // Double line variants
    pub const dtl = "╔";
    pub const dtr = "╗";
    pub const dbl = "╚";
    pub const dbr = "╝";
    pub const dh = "═";
    pub const dv = "║";
};

// ═══════════════════════════════════════════════════════════════════════════
// Progress Indicator
// ═══════════════════════════════════════════════════════════════════════════

/// Animated spinner styles
pub const SpinnerStyle = enum {
    dots,
    line,
    arrow,
    circle,
    braille,

    pub fn frames(self: SpinnerStyle) []const []const u8 {
        return switch (self) {
            .dots => &[_][]const u8{ "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" },
            .line => &[_][]const u8{ "|", "/", "-", "\\" },
            .arrow => &[_][]const u8{ "←", "↖", "↑", "↗", "→", "↘", "↓", "↙" },
            .circle => &[_][]const u8{ "◐", "◓", "◑", "◒" },
            .braille => &[_][]const u8{ "⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷" },
        };
    }
};

pub const ProgressIndicator = struct {
    term: *terminal.Terminal,
    message: []const u8,
    style: SpinnerStyle,
    frame: usize,
    active: bool,
    start_time: i64,

    pub fn init(term: *terminal.Terminal, message: []const u8, style: SpinnerStyle) ProgressIndicator {
        return .{
            .term = term,
            .message = message,
            .style = style,
            .frame = 0,
            .active = false,
            .start_time = std.time.milliTimestamp(),
        };
    }

    pub fn start(self: *ProgressIndicator) !void {
        self.active = true;
        self.start_time = std.time.milliTimestamp();
        try self.render();
    }

    pub fn tick(self: *ProgressIndicator) !void {
        if (!self.active) return;
        self.frame = (self.frame + 1) % self.style.frames().len;
        try self.render();
    }

    pub fn stop(self: *ProgressIndicator, final_message: ?[]const u8) !void {
        self.active = false;
        try self.term.write("\r\x1b[K"); // Clear line

        if (final_message) |msg| {
            try self.term.write(colors.green);
            try self.term.write("✓ ");
            try self.term.write(colors.reset);
            try self.term.write(msg);
            try self.term.write("\n");
        }
    }

    pub fn fail(self: *ProgressIndicator, error_message: ?[]const u8) !void {
        self.active = false;
        try self.term.write("\r\x1b[K"); // Clear line

        if (error_message) |msg| {
            try self.term.write(colors.red);
            try self.term.write("✗ ");
            try self.term.write(colors.reset);
            try self.term.write(msg);
            try self.term.write("\n");
        }
    }

    fn render(self: *ProgressIndicator) !void {
        const frames = self.style.frames();
        const current_frame = frames[self.frame];

        try self.term.write("\r");
        try self.term.write(colors.cyan);
        try self.term.write(current_frame);
        try self.term.write(" ");
        try self.term.write(colors.reset);
        try self.term.write(self.message);

        // Show elapsed time
        const elapsed = std.time.milliTimestamp() - self.start_time;
        const elapsed_secs = @as(f64, @floatFromInt(elapsed)) / 1000.0;
        var buf: [32]u8 = undefined;
        const time_str = std.fmt.bufPrint(&buf, " ({d:.1}s)", .{elapsed_secs}) catch "";
        try self.term.write(colors.dim);
        try self.term.write(time_str);
        try self.term.write(colors.reset);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Progress Bar
// ═══════════════════════════════════════════════════════════════════════════

pub const ProgressBar = struct {
    term: *terminal.Terminal,
    total: usize,
    current: usize,
    width: usize,
    label: []const u8,
    show_percentage: bool,
    show_count: bool,

    pub fn init(term: *terminal.Terminal, total: usize, label: []const u8) ProgressBar {
        return .{
            .term = term,
            .total = total,
            .current = 0,
            .width = 40,
            .label = label,
            .show_percentage = true,
            .show_count = true,
        };
    }

    pub fn setWidth(self: *ProgressBar, width: usize) *ProgressBar {
        self.width = width;
        return self;
    }

    pub fn update(self: *ProgressBar, current: usize) !void {
        self.current = @min(current, self.total);
        try self.render();
    }

    pub fn increment(self: *ProgressBar) !void {
        try self.update(self.current + 1);
    }

    pub fn complete(self: *ProgressBar) !void {
        try self.update(self.total);
        try self.term.write("\n");
    }

    fn render(self: *ProgressBar) !void {
        const progress: f64 = if (self.total == 0)
            1.0
        else
            @as(f64, @floatFromInt(self.current)) / @as(f64, @floatFromInt(self.total));

        const filled = @as(usize, @intFromFloat(progress * @as(f64, @floatFromInt(self.width))));

        try self.term.write("\r");

        // Label
        if (self.label.len > 0) {
            try self.term.write(self.label);
            try self.term.write(" ");
        }

        // Bar
        try self.term.write(colors.dim);
        try self.term.write("[");
        try self.term.write(colors.reset);
        try self.term.write(colors.green);

        for (0..self.width) |i| {
            if (i < filled) {
                try self.term.write("█");
            } else if (i == filled) {
                try self.term.write(colors.yellow);
                try self.term.write("▓");
                try self.term.write(colors.dim);
            } else {
                try self.term.write("░");
            }
        }

        try self.term.write(colors.reset);
        try self.term.write(colors.dim);
        try self.term.write("]");
        try self.term.write(colors.reset);

        // Percentage
        if (self.show_percentage) {
            var buf: [8]u8 = undefined;
            const pct = @as(u8, @intFromFloat(progress * 100));
            const pct_str = std.fmt.bufPrint(&buf, " {d:>3}%", .{pct}) catch "";
            try self.term.write(pct_str);
        }

        // Count
        if (self.show_count) {
            var buf: [32]u8 = undefined;
            const count_str = std.fmt.bufPrint(&buf, " ({d}/{d})", .{ self.current, self.total }) catch "";
            try self.term.write(colors.dim);
            try self.term.write(count_str);
            try self.term.write(colors.reset);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Dialog Box
// ═══════════════════════════════════════════════════════════════════════════

pub const DialogResult = enum {
    confirm,
    cancel,
    option1,
    option2,
};

pub const Dialog = struct {
    term: *terminal.Terminal,
    title: []const u8,
    message: []const u8,
    width: usize,
    style: DialogStyle,

    pub const DialogStyle = enum {
        info,
        warning,
        @"error",
        confirm,
    };

    pub fn init(term: *terminal.Terminal, title: []const u8, message: []const u8) Dialog {
        return .{
            .term = term,
            .title = title,
            .message = message,
            .width = 50,
            .style = .info,
        };
    }

    pub fn setStyle(self: *Dialog, style: DialogStyle) *Dialog {
        self.style = style;
        return self;
    }

    pub fn setWidth(self: *Dialog, width: usize) *Dialog {
        self.width = width;
        return self;
    }

    pub fn show(self: *Dialog) !void {
        const style_color = switch (self.style) {
            .info => colors.cyan,
            .warning => colors.yellow,
            .@"error" => colors.red,
            .confirm => colors.green,
        };

        const icon = switch (self.style) {
            .info => "ℹ",
            .warning => "⚠",
            .@"error" => "✗",
            .confirm => "?",
        };

        // Top border
        try self.term.write("\n");
        try self.term.write(style_color);
        try self.term.write(box.tl);
        try writeRepeat(self.term, box.h, self.width - 2);
        try self.term.write(box.tr);
        try self.term.write(colors.reset);
        try self.term.write("\n");

        // Title
        try self.term.write(style_color);
        try self.term.write(box.v);
        try self.term.write(" ");
        try self.term.write(icon);
        try self.term.write(" ");
        try self.term.write(colors.bold);
        try self.term.write(self.title);
        try self.term.write(colors.reset);

        const title_len = self.title.len + 4;
        if (title_len < self.width - 2) {
            try writeRepeat(self.term, " ", self.width - 2 - title_len);
        }
        try self.term.write(style_color);
        try self.term.write(box.v);
        try self.term.write(colors.reset);
        try self.term.write("\n");

        // Separator
        try self.term.write(style_color);
        try self.term.write(box.lsep);
        try writeRepeat(self.term, box.h, self.width - 2);
        try self.term.write(box.rsep);
        try self.term.write(colors.reset);
        try self.term.write("\n");

        // Message (word-wrapped)
        try self.renderWrappedText(self.message, style_color);

        // Bottom border
        try self.term.write(style_color);
        try self.term.write(box.bl);
        try writeRepeat(self.term, box.h, self.width - 2);
        try self.term.write(box.br);
        try self.term.write(colors.reset);
        try self.term.write("\n");
    }

    pub fn confirm(self: *Dialog, yes_label: []const u8, no_label: []const u8) !DialogResult {
        self.style = .confirm;
        try self.show();

        // Show options
        try self.term.write("\n ");
        try self.term.write(colors.green);
        try self.term.write("[Y] ");
        try self.term.write(yes_label);
        try self.term.write(colors.reset);
        try self.term.write("  ");
        try self.term.write(colors.red);
        try self.term.write("[N] ");
        try self.term.write(no_label);
        try self.term.write(colors.reset);
        try self.term.write(" ");

        // Read response
        while (true) {
            const key = try self.term.readKey();
            switch (key.code) {
                .character => {
                    if (key.char) |ch| {
                        switch (ch) {
                            'y', 'Y' => {
                                try self.term.write(colors.green);
                                try self.term.write("Yes\n");
                                try self.term.write(colors.reset);
                                return .confirm;
                            },
                            'n', 'N' => {
                                try self.term.write(colors.red);
                                try self.term.write("No\n");
                                try self.term.write(colors.reset);
                                return .cancel;
                            },
                            else => {},
                        }
                    }
                },
                .escape, .ctrl_c => {
                    try self.term.write(colors.dim);
                    try self.term.write("Cancelled\n");
                    try self.term.write(colors.reset);
                    return .cancel;
                },
                .enter => {
                    // Default to cancel
                    try self.term.write(colors.dim);
                    try self.term.write("Cancelled\n");
                    try self.term.write(colors.reset);
                    return .cancel;
                },
                else => {},
            }
        }
    }

    fn renderWrappedText(self: *Dialog, text: []const u8, border_color: []const u8) !void {
        const max_line_width = self.width - 4;
        var start: usize = 0;

        while (start < text.len) {
            var end = start + max_line_width;
            if (end >= text.len) {
                end = text.len;
            } else {
                // Find word boundary
                while (end > start and text[end] != ' ' and text[end] != '\n') {
                    end -= 1;
                }
                if (end == start) {
                    end = start + max_line_width;
                }
            }

            // Handle explicit newlines
            var newline_pos: ?usize = null;
            for (start..end) |i| {
                if (text[i] == '\n') {
                    newline_pos = i;
                    break;
                }
            }
            if (newline_pos) |nl| {
                end = nl;
            }

            try self.term.write(border_color);
            try self.term.write(box.v);
            try self.term.write(colors.reset);
            try self.term.write(" ");
            try self.term.write(text[start..end]);

            const line_len = end - start + 2;
            if (line_len < self.width - 1) {
                try writeRepeat(self.term, " ", self.width - 1 - line_len);
            }

            try self.term.write(border_color);
            try self.term.write(box.v);
            try self.term.write(colors.reset);
            try self.term.write("\n");

            start = end;
            if (start < text.len and (text[start] == ' ' or text[start] == '\n')) {
                start += 1;
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Command Preview Panel
// ═══════════════════════════════════════════════════════════════════════════

pub const CommandPreview = struct {
    term: *terminal.Terminal,
    name: []const u8,
    description: []const u8,
    usage: []const u8,
    examples: []const []const u8,
    related: []const []const u8,

    pub fn init(term: *terminal.Terminal, name: []const u8) CommandPreview {
        return .{
            .term = term,
            .name = name,
            .description = "",
            .usage = "",
            .examples = &[_][]const u8{},
            .related = &[_][]const u8{},
        };
    }

    pub fn setDescription(self: *CommandPreview, desc: []const u8) *CommandPreview {
        self.description = desc;
        return self;
    }

    pub fn setUsage(self: *CommandPreview, usage: []const u8) *CommandPreview {
        self.usage = usage;
        return self;
    }

    pub fn setExamples(self: *CommandPreview, examples: []const []const u8) *CommandPreview {
        self.examples = examples;
        return self;
    }

    pub fn setRelated(self: *CommandPreview, related: []const []const u8) *CommandPreview {
        self.related = related;
        return self;
    }

    pub fn render(self: *CommandPreview, width: usize) !void {
        // Header
        try self.term.write("\n");
        try self.term.write(colors.cyan);
        try self.term.write(box.dtl);
        try writeRepeat(self.term, box.dh, width - 2);
        try self.term.write(box.dtr);
        try self.term.write(colors.reset);
        try self.term.write("\n");

        // Title
        try self.term.write(colors.cyan);
        try self.term.write(box.dv);
        try self.term.write(colors.reset);
        try self.term.write(" ");
        try self.term.write(colors.bold);
        try self.term.write(colors.bright_white);
        try self.term.write(self.name);
        try self.term.write(colors.reset);

        const title_len = self.name.len + 2;
        if (title_len < width - 1) {
            try writeRepeat(self.term, " ", width - 1 - title_len);
        }
        try self.term.write(colors.cyan);
        try self.term.write(box.dv);
        try self.term.write(colors.reset);
        try self.term.write("\n");

        // Description
        if (self.description.len > 0) {
            try self.term.write(colors.cyan);
            try self.term.write(box.dv);
            try self.term.write(colors.reset);
            try self.term.write(" ");
            try self.term.write(colors.dim);
            try self.term.write(self.description);
            try self.term.write(colors.reset);

            const desc_len = self.description.len + 2;
            if (desc_len < width - 1) {
                try writeRepeat(self.term, " ", width - 1 - desc_len);
            }
            try self.term.write(colors.cyan);
            try self.term.write(box.dv);
            try self.term.write(colors.reset);
            try self.term.write("\n");
        }

        // Usage section
        if (self.usage.len > 0) {
            try self.renderSection("Usage", width);
            try self.renderLine(self.usage, width);
        }

        // Examples section
        if (self.examples.len > 0) {
            try self.renderSection("Examples", width);
            for (self.examples) |example| {
                try self.term.write(colors.cyan);
                try self.term.write(box.dv);
                try self.term.write(colors.reset);
                try self.term.write("   ");
                try self.term.write(colors.green);
                try self.term.write("$ ");
                try self.term.write(colors.reset);
                try self.term.write(example);

                const ex_len = example.len + 5;
                if (ex_len < width - 1) {
                    try writeRepeat(self.term, " ", width - 1 - ex_len);
                }
                try self.term.write(colors.cyan);
                try self.term.write(box.dv);
                try self.term.write(colors.reset);
                try self.term.write("\n");
            }
        }

        // Related commands
        if (self.related.len > 0) {
            try self.renderSection("Related", width);
            try self.term.write(colors.cyan);
            try self.term.write(box.dv);
            try self.term.write(colors.reset);
            try self.term.write("   ");

            var total_len: usize = 3;
            for (self.related, 0..) |rel, i| {
                if (i > 0) {
                    try self.term.write(", ");
                    total_len += 2;
                }
                try self.term.write(colors.yellow);
                try self.term.write(rel);
                try self.term.write(colors.reset);
                total_len += rel.len;
            }

            if (total_len < width - 1) {
                try writeRepeat(self.term, " ", width - 1 - total_len);
            }
            try self.term.write(colors.cyan);
            try self.term.write(box.dv);
            try self.term.write(colors.reset);
            try self.term.write("\n");
        }

        // Footer
        try self.term.write(colors.cyan);
        try self.term.write(box.dbl);
        try writeRepeat(self.term, box.dh, width - 2);
        try self.term.write(box.dbr);
        try self.term.write(colors.reset);
        try self.term.write("\n");

        // Help text
        try self.term.write(colors.dim);
        try self.term.write(" Press ");
        try self.term.write(colors.reset);
        try self.term.write(colors.yellow);
        try self.term.write("Enter");
        try self.term.write(colors.reset);
        try self.term.write(colors.dim);
        try self.term.write(" to run, ");
        try self.term.write(colors.reset);
        try self.term.write(colors.yellow);
        try self.term.write("Esc");
        try self.term.write(colors.reset);
        try self.term.write(colors.dim);
        try self.term.write(" to go back\n");
        try self.term.write(colors.reset);
    }

    fn renderSection(self: *CommandPreview, title: []const u8, width: usize) !void {
        try self.term.write(colors.cyan);
        try self.term.write(box.dv);
        try self.term.write(colors.reset);
        try writeRepeat(self.term, " ", width - 2);
        try self.term.write(colors.cyan);
        try self.term.write(box.dv);
        try self.term.write(colors.reset);
        try self.term.write("\n");

        try self.term.write(colors.cyan);
        try self.term.write(box.dv);
        try self.term.write(colors.reset);
        try self.term.write(" ");
        try self.term.write(colors.bold);
        try self.term.write(colors.cyan);
        try self.term.write(title);
        try self.term.write(":");
        try self.term.write(colors.reset);

        const sect_len = title.len + 3;
        if (sect_len < width - 1) {
            try writeRepeat(self.term, " ", width - 1 - sect_len);
        }
        try self.term.write(colors.cyan);
        try self.term.write(box.dv);
        try self.term.write(colors.reset);
        try self.term.write("\n");
    }

    fn renderLine(self: *CommandPreview, text: []const u8, width: usize) !void {
        try self.term.write(colors.cyan);
        try self.term.write(box.dv);
        try self.term.write(colors.reset);
        try self.term.write("   ");
        try self.term.write(text);

        const line_len = text.len + 3;
        if (line_len < width - 1) {
            try writeRepeat(self.term, " ", width - 1 - line_len);
        }
        try self.term.write(colors.cyan);
        try self.term.write(box.dv);
        try self.term.write(colors.reset);
        try self.term.write("\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Notification Toast
// ═══════════════════════════════════════════════════════════════════════════

pub const Toast = struct {
    term: *terminal.Terminal,

    pub const Level = enum {
        success,
        info,
        warning,
        @"error",
    };

    pub fn init(term: *terminal.Terminal) Toast {
        return .{ .term = term };
    }

    pub fn show(self: *Toast, level: Level, message: []const u8) !void {
        const prefix = switch (level) {
            .success => colors.green ++ "✓ ",
            .info => colors.cyan ++ "ℹ ",
            .warning => colors.yellow ++ "⚠ ",
            .@"error" => colors.red ++ "✗ ",
        };

        try self.term.write(prefix);
        try self.term.write(colors.reset);
        try self.term.write(message);
        try self.term.write("\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn writeRepeat(term: *terminal.Terminal, char: []const u8, count: usize) !void {
    for (0..count) |_| {
        try term.write(char);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "SpinnerStyle frames" {
    const dots = SpinnerStyle.dots.frames();
    try std.testing.expect(dots.len == 10);

    const line = SpinnerStyle.line.frames();
    try std.testing.expect(line.len == 4);
}
