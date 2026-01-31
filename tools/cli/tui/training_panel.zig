//! Training Progress Panel for TUI
//!
//! Displays training metrics: loss curves, learning rate,
//! GPU/memory usage, and checkpoint status.
//!
//! Usage:
//!   zig build run -- train monitor [run-id]
//!
//! Keyboard:
//!   r - Refresh
//!   h - Toggle history mode
//!   q - Quit
//!   ←/→ - Switch runs (history mode)

const std = @import("std");
const terminal = @import("terminal.zig");
const events = @import("events.zig");
const themes = @import("themes.zig");
const widgets = @import("widgets.zig");
const metrics = @import("training_metrics.zig");
const cli_io = @import("../utils/io_backend.zig");

// ===============================================================================
// Types
// ===============================================================================

/// Panel display mode
pub const Mode = enum {
    live,
    history,

    pub fn label(self: Mode) []const u8 {
        return switch (self) {
            .live => "[Live]",
            .history => "[History]",
        };
    }
};

/// Panel configuration
pub const PanelConfig = struct {
    log_dir: []const u8 = "logs",
    run_id: ?[]const u8 = null,
    refresh_ms: u64 = 500,
};

// ===============================================================================
// Training Panel
// ===============================================================================

pub const TrainingPanel = struct {
    allocator: std.mem.Allocator,
    theme: *const themes.Theme,
    config: PanelConfig,
    mode: Mode,
    training_metrics: metrics.TrainingMetrics,
    metrics_path: []const u8,
    owns_metrics_path: bool,

    // Display state
    width: usize,
    last_refresh: i64,
    last_file_pos: u64,
    running: bool,

    // Help overlay state
    show_help: bool,

    // Run switching state (for history mode)
    available_runs: std.ArrayListUnmanaged([]const u8),
    current_run_index: usize,

    pub fn init(allocator: std.mem.Allocator, theme: *const themes.Theme, config: PanelConfig) TrainingPanel {
        const path = std.fmt.allocPrint(allocator, "{s}/metrics.jsonl", .{config.log_dir}) catch config.log_dir;
        const owns = path.ptr != config.log_dir.ptr;
        return .{
            .allocator = allocator,
            .theme = theme,
            .config = config,
            .mode = .live,
            .training_metrics = metrics.TrainingMetrics{},
            .width = 72,
            .last_refresh = 0,
            .last_file_pos = 0,
            .running = true,
            .metrics_path = path,
            .owns_metrics_path = owns,
            .show_help = false,
            .available_runs = .empty,
            .current_run_index = 0,
        };
    }

    pub fn deinit(self: *TrainingPanel) void {
        if (self.owns_metrics_path) {
            self.allocator.free(self.metrics_path);
        }
        // Free run names
        for (self.available_runs.items) |run_name| {
            self.allocator.free(run_name);
        }
        self.available_runs.deinit(self.allocator);
    }

    /// Render the full panel to a writer
    pub fn render(self: *TrainingPanel, writer: anytype) !void {
        if (self.show_help) {
            try self.renderHelpOverlay(writer);
            return;
        }

        try self.renderHeader(writer);
        try self.renderDivider(writer, .top);

        // Top row: Loss | Learning Rate
        try self.renderLossPanel(writer);
        try self.renderDivider(writer, .middle);

        // Bottom row: Resources | Checkpoints
        try self.renderResourcesPanel(writer);
        try self.renderDivider(writer, .bottom);

        try self.renderFooter(writer);
    }

    /// Render the help overlay
    fn renderHelpOverlay(self: *TrainingPanel, writer: anytype) !void {
        const help_width = 50;

        // Top border
        try writer.print("{s}╭", .{self.theme.primary});
        try self.writeRepeat(writer, "─", help_width - 2);
        try writer.print("╮{s}\n", .{self.theme.reset});

        // Title
        try writer.print("{s}│{s}  {s}Training Panel Help{s}", .{
            self.theme.primary,
            self.theme.reset,
            self.theme.bold,
            self.theme.reset,
        });
        try self.writeRepeat(writer, " ", help_width - 23);
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });

        // Divider
        try writer.print("{s}├", .{self.theme.primary});
        try self.writeRepeat(writer, "─", help_width - 2);
        try writer.print("┤{s}\n", .{self.theme.reset});

        // Help entries
        const help_entries = [_]struct { key: []const u8, desc: []const u8 }{
            .{ .key = "r", .desc = "Refresh metrics display" },
            .{ .key = "h", .desc = "Toggle history/live mode" },
            .{ .key = "q / Esc", .desc = "Quit the panel" },
            .{ .key = "← / →", .desc = "Switch runs (history mode)" },
            .{ .key = "?", .desc = "Toggle this help overlay" },
            .{ .key = "Ctrl+C", .desc = "Force quit" },
        };

        for (help_entries) |entry| {
            try writer.print("{s}│{s}  {s}{s: <12}{s} {s}", .{
                self.theme.primary,
                self.theme.reset,
                self.theme.bold,
                entry.key,
                self.theme.reset,
                entry.desc,
            });
            const content_len = 14 + entry.desc.len;
            if (content_len < help_width - 2) {
                try self.writeRepeat(writer, " ", help_width - 2 - content_len);
            }
            try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });
        }

        // Empty line
        try writer.print("{s}│{s}", .{ self.theme.primary, self.theme.reset });
        try self.writeRepeat(writer, " ", help_width - 2);
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });

        // Mode info
        try writer.print("{s}│{s}  {s}Current Mode:{s} {s}", .{
            self.theme.primary,
            self.theme.reset,
            self.theme.bold,
            self.theme.reset,
            self.mode.label(),
        });
        const mode_len = 16 + self.mode.label().len;
        if (mode_len < help_width - 2) {
            try self.writeRepeat(writer, " ", help_width - 2 - mode_len);
        }
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });

        // Run count
        const run_count = self.available_runs.items.len;
        if (run_count > 0) {
            try writer.print("{s}│{s}  {s}Available Runs:{s} {d}", .{
                self.theme.primary,
                self.theme.reset,
                self.theme.bold,
                self.theme.reset,
                run_count,
            });
            try self.writeRepeat(writer, " ", help_width - 22);
            try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });
        }

        // Bottom border
        try writer.print("{s}├", .{self.theme.primary});
        try self.writeRepeat(writer, "─", help_width - 2);
        try writer.print("┤{s}\n", .{self.theme.reset});

        // Footer
        try writer.print("{s}│{s}  Press any key to close", .{
            self.theme.primary,
            self.theme.reset,
        });
        try self.writeRepeat(writer, " ", help_width - 27);
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });

        try writer.print("{s}╰", .{self.theme.primary});
        try self.writeRepeat(writer, "─", help_width - 2);
        try writer.print("╯{s}\n", .{self.theme.reset});
    }

    // =========================================================================
    // Header
    // =========================================================================

    fn renderHeader(self: *TrainingPanel, writer: anytype) !void {
        const run_id = self.config.run_id orelse "current";
        const elapsed = self.training_metrics.elapsedSeconds();
        const hours = elapsed / 3600;
        const mins = (elapsed % 3600) / 60;
        const secs = elapsed % 60;

        try writer.print("{s}╭", .{self.theme.primary});
        try self.writeRepeat(writer, "─", self.width - 2);
        try writer.print("╮{s}\n", .{self.theme.reset});

        try writer.print("{s}│{s} Training Monitor: {s}{s}{s}  {s}  ⏱ {d:0>2}:{d:0>2}:{d:0>2}", .{
            self.theme.primary,
            self.theme.reset,
            self.theme.bold,
            run_id,
            self.theme.reset,
            self.mode.label(),
            hours,
            mins,
            secs,
        });

        // Pad to width
        const content_len = 30 + run_id.len + 8 + 8; // approximate
        if (content_len < self.width - 2) {
            try self.writeRepeat(writer, " ", self.width - 2 - content_len);
        }
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });
    }

    // =========================================================================
    // Loss Panel
    // =========================================================================

    fn renderLossPanel(self: *TrainingPanel, writer: anytype) !void {
        const tm = &self.training_metrics;

        // Title row
        try writer.print("{s}│{s} {s}Loss{s}", .{
            self.theme.primary,
            self.theme.reset,
            self.theme.bold,
            self.theme.reset,
        });
        try self.writeRepeat(writer, " ", 31);
        try writer.print("{s}│{s} {s}Learning Rate{s}", .{
            self.theme.primary,
            self.theme.reset,
            self.theme.bold,
            self.theme.reset,
        });
        try self.writeRepeat(writer, " ", 19);
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });

        // Sparkline row
        var loss_buf: [128]u8 = undefined;
        var lr_buf: [128]u8 = undefined;

        const loss_normalized = tm.train_loss.getNormalized();
        const loss_sparkline = widgets.SparklineChart.render(&loss_normalized, &loss_buf);

        const lr_normalized = tm.learning_rate.getNormalized();
        const lr_sparkline = widgets.SparklineChart.render(&lr_normalized, &lr_buf);

        try writer.print("{s}│{s} {s}", .{ self.theme.primary, self.theme.reset, loss_sparkline });
        try self.writeRepeat(writer, " ", 35 - @min(35, loss_sparkline.len / 3)); // UTF-8 chars
        try writer.print("{s}│{s} {s}", .{ self.theme.primary, self.theme.reset, lr_sparkline });
        try self.writeRepeat(writer, " ", 32 - @min(32, lr_sparkline.len / 3));
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });

        // Values row
        const train_loss = tm.train_loss.latest();
        const val_loss = tm.val_loss.latest();
        const lr = tm.learning_rate.latest();

        try writer.print("{s}│{s} train: {d:.4}  val: {d:.4}", .{
            self.theme.primary,
            self.theme.reset,
            train_loss,
            val_loss,
        });
        try self.writeRepeat(writer, " ", 10);
        try writer.print("{s}│{s} current: {d:.6}", .{
            self.theme.primary,
            self.theme.reset,
            lr,
        });
        try self.writeRepeat(writer, " ", 15);
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });

        // Progress row
        try writer.print("{s}│{s} epoch: {d}/{d}    step: {d}/{d}", .{
            self.theme.primary,
            self.theme.reset,
            tm.current_epoch,
            tm.total_epochs,
            tm.current_step,
            tm.total_steps,
        });
        try self.writeRepeat(writer, " ", 6);
        try writer.print("{s}│{s} schedule: cosine", .{
            self.theme.primary,
            self.theme.reset,
        });
        try self.writeRepeat(writer, " ", 14);
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });
    }

    // =========================================================================
    // Resources Panel
    // =========================================================================

    fn renderResourcesPanel(self: *TrainingPanel, writer: anytype) !void {
        const tm = &self.training_metrics;

        // Title row
        try writer.print("{s}│{s} {s}Resources{s}", .{
            self.theme.primary,
            self.theme.reset,
            self.theme.bold,
            self.theme.reset,
        });
        try self.writeRepeat(writer, " ", 26);
        try writer.print("{s}│{s} {s}Checkpoints{s}", .{
            self.theme.primary,
            self.theme.reset,
            self.theme.bold,
            self.theme.reset,
        });
        try self.writeRepeat(writer, " ", 21);
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });

        // GPU row (placeholder - would need actual GPU stats)
        var gauge_buf: [64]u8 = undefined;
        const gpu_gauge = widgets.ProgressGauge.render(0, 16, &gauge_buf);
        try writer.print("{s}│{s} GPU:  {s} N/A", .{
            self.theme.primary,
            self.theme.reset,
            gpu_gauge,
        });
        try self.writeRepeat(writer, " ", 10);

        // Checkpoint info
        const ckpt_count = tm.checkpoint_count;
        if (ckpt_count > 0) {
            try writer.print("{s}│{s} ✓ {d} checkpoints saved", .{
                self.theme.primary,
                self.theme.reset,
                ckpt_count,
            });
        } else {
            try writer.print("{s}│{s}   No checkpoints yet", .{
                self.theme.primary,
                self.theme.reset,
            });
        }
        try self.writeRepeat(writer, " ", 10);
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });

        // Memory row
        const mem_gauge = widgets.ProgressGauge.render(0, 16, &gauge_buf);
        try writer.print("{s}│{s} VRAM: {s} N/A", .{
            self.theme.primary,
            self.theme.reset,
            mem_gauge,
        });
        try self.writeRepeat(writer, " ", 10);

        // Last checkpoint
        if (tm.checkpoint_count > 0) {
            const size_mb = @as(f64, @floatFromInt(tm.last_checkpoint_size)) / (1024.0 * 1024.0);
            try writer.print("{s}│{s} ★ latest: {d:.1} MB", .{
                self.theme.primary,
                self.theme.reset,
                size_mb,
            });
        } else {
            try writer.print("{s}│{s}", .{
                self.theme.primary,
                self.theme.reset,
            });
        }
        try self.writeRepeat(writer, " ", 12);
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });
    }

    // =========================================================================
    // Footer
    // =========================================================================

    fn renderFooter(self: *TrainingPanel, writer: anytype) !void {
        try writer.print("{s}│{s} [r] Refresh  [h] History  [q] Quit  [←/→] Switch runs  [?] Help", .{
            self.theme.primary,
            self.theme.reset,
        });
        try self.writeRepeat(writer, " ", 4);
        try writer.print("{s}│{s}\n", .{ self.theme.primary, self.theme.reset });

        try writer.print("{s}╰", .{self.theme.primary});
        try self.writeRepeat(writer, "─", self.width - 2);
        try writer.print("╯{s}\n", .{self.theme.reset});
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    const DividerType = enum { top, middle, bottom };

    fn renderDivider(self: *TrainingPanel, writer: anytype, div_type: DividerType) !void {
        const chars = switch (div_type) {
            .top => .{ "├", "┬", "┤" },
            .middle => .{ "├", "┼", "┤" },
            .bottom => .{ "├", "┴", "┤" },
        };

        try writer.print("{s}{s}", .{ self.theme.primary, chars[0] });
        try self.writeRepeat(writer, "─", 35);
        try writer.print("{s}", .{chars[1]});
        try self.writeRepeat(writer, "─", self.width - 38);
        try writer.print("{s}{s}\n", .{ chars[2], self.theme.reset });
    }

    fn writeRepeat(self: *TrainingPanel, writer: anytype, char: []const u8, count: usize) !void {
        _ = self;
        for (0..count) |_| {
            try writer.print("{s}", .{char});
        }
    }

    // =========================================================================
    // Input Handling
    // =========================================================================

    pub fn handleKey(self: *TrainingPanel, key: u8) ?Action {
        return switch (key) {
            'q', 'Q' => .quit,
            'r', 'R' => .refresh,
            'h', 'H' => blk: {
                self.mode = if (self.mode == .live) .history else .live;
                break :blk .refresh;
            },
            '?' => .help,
            else => null,
        };
    }

    pub const Action = enum {
        quit,
        refresh,
        help,
        next_run,
        prev_run,
    };

    // =========================================================================
    // Interactive Mode
    // =========================================================================

    /// Run the panel in interactive mode with a terminal
    pub fn runInteractive(self: *TrainingPanel, term: *terminal.Terminal) !void {
        // Enter raw mode
        try term.enter();
        defer term.exit() catch {};

        self.running = true;

        // Initial load
        self.loadMetricsFile(self.buildMetricsPath()) catch {};

        while (self.running) {
            // Pull latest metrics incrementally
            _ = self.pollMetrics() catch {};

            // Clear and render
            try term.clear();

            // Create a wrapper writer for the terminal
            const TermWriter = struct {
                terminal: *terminal.Terminal,

                pub const Error = error{};
                pub fn print(writer: @This(), comptime fmt: []const u8, print_args: anytype) Error!void {
                    var buf: [1024]u8 = undefined;
                    const output = std.fmt.bufPrint(&buf, fmt, print_args) catch return;
                    writer.terminal.write(output) catch {};
                }
            };
            try self.render(TermWriter{ .terminal = term });

            // Wait for input (blocking)
            const event = term.readEvent() catch break;

            switch (event) {
                .key => |key| {
                    if (self.handleKeyEvent(key)) {
                        break;
                    }
                },
                .mouse => {},
            }
        }
    }

    /// Handle a key event, returns true if should quit
    fn handleKeyEvent(self: *TrainingPanel, key: events.Key) bool {
        // If help overlay is showing, any key closes it
        if (self.show_help) {
            self.show_help = false;
            return false;
        }

        switch (key.code) {
            .ctrl_c => return true,
            .escape => return true,
            .character => {
                if (key.char) |ch| {
                    if (self.handleKey(ch)) |action| {
                        switch (action) {
                            .quit => return true,
                            .refresh => {
                                _ = self.pollMetrics() catch {};
                            },
                            .help => {
                                self.show_help = true;
                            },
                            .next_run => {
                                if (self.mode == .history) {
                                    self.switchToNextRun();
                                }
                            },
                            .prev_run => {
                                if (self.mode == .history) {
                                    self.switchToPrevRun();
                                }
                            },
                        }
                    }
                }
            },
            .left => {
                if (self.mode == .history) {
                    self.switchToPrevRun();
                }
            },
            .right => {
                if (self.mode == .history) {
                    self.switchToNextRun();
                }
            },
            else => {},
        }
        return false;
    }

    /// Switch to the next run in history mode
    fn switchToNextRun(self: *TrainingPanel) void {
        if (self.available_runs.items.len == 0) {
            self.scanAvailableRuns() catch return;
        }
        if (self.available_runs.items.len == 0) return;

        self.current_run_index = (self.current_run_index + 1) % self.available_runs.items.len;
        self.loadRunAtIndex(self.current_run_index);
    }

    /// Switch to the previous run in history mode
    fn switchToPrevRun(self: *TrainingPanel) void {
        if (self.available_runs.items.len == 0) {
            self.scanAvailableRuns() catch return;
        }
        if (self.available_runs.items.len == 0) return;

        if (self.current_run_index == 0) {
            self.current_run_index = self.available_runs.items.len - 1;
        } else {
            self.current_run_index -= 1;
        }
        self.loadRunAtIndex(self.current_run_index);
    }

    /// Load metrics from run at given index
    fn loadRunAtIndex(self: *TrainingPanel, index: usize) void {
        if (index >= self.available_runs.items.len) return;

        const run_name = self.available_runs.items[index];

        // Reset metrics
        self.training_metrics = metrics.TrainingMetrics{};
        self.last_file_pos = 0;

        // Build path to run's metrics file
        const run_path = std.fmt.allocPrint(self.allocator, "{s}/{s}/metrics.jsonl", .{
            self.config.log_dir,
            run_name,
        }) catch return;
        defer self.allocator.free(run_path);

        self.loadMetricsFile(run_path) catch {};
    }

    /// Scan log directory for available training runs
    fn scanAvailableRuns(self: *TrainingPanel) !void {
        // Clear existing runs
        for (self.available_runs.items) |run_name| {
            self.allocator.free(run_name);
        }
        self.available_runs.clearRetainingCapacity();

        var io_backend = cli_io.initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        // Open log directory
        var dir = std.Io.Dir.cwd().openDir(io, self.config.log_dir, .{ .iterate = true }) catch return;
        defer dir.close(io);

        // Iterate through subdirectories
        var iter = dir.iterate();
        while (true) {
            const maybe_entry = iter.next(io) catch break;
            const entry = maybe_entry orelse break;

            if (entry.kind == .directory) {
                // Check if directory has a metrics.jsonl file
                const metrics_path = std.fmt.allocPrint(self.allocator, "{s}/{s}/metrics.jsonl", .{
                    self.config.log_dir,
                    entry.name,
                }) catch continue;
                defer self.allocator.free(metrics_path);

                // Try to stat the file to verify it exists
                _ = std.Io.Dir.cwd().statFile(io, metrics_path, .{}) catch continue;

                // Add to available runs
                const name_copy = self.allocator.dupe(u8, entry.name) catch continue;
                self.available_runs.append(self.allocator, name_copy) catch {
                    self.allocator.free(name_copy);
                    continue;
                };
            }
        }
    }

    // =========================================================================
    // Metrics Loading
    // =========================================================================

    /// Load metrics from a JSONL file
    pub fn loadMetricsFile(self: *TrainingPanel, path: []const u8) !void {
        var io_backend = cli_io.initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        // Read file content using Zig 0.16 API
        const content = std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(10 * 1024 * 1024)) catch |err| {
            // File doesn't exist yet or read error - handle gracefully
            if (err == error.FileNotFound) return;
            return;
        };
        defer self.allocator.free(content);

        // Parse line by line
        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            const event = metrics.MetricsParser.parseLine(line);
            self.training_metrics.update(event);
        }
        self.last_file_pos = content.len;
    }

    /// Poll for new metrics (incremental read from last position)
    pub fn pollMetrics(self: *TrainingPanel) !bool {
        const log_path = self.buildMetricsPath();

        var io_backend = cli_io.initIoBackend(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        // Read full file content using Zig 0.16 API
        // Note: For incremental reads, we re-read the full file and track position
        const content = std.Io.Dir.cwd().readFileAlloc(io, log_path, self.allocator, .limited(1024 * 1024)) catch |err| {
            if (err == error.FileNotFound) return false;
            return false;
        };
        defer self.allocator.free(content);

        const file_size = content.len;

        if (file_size <= self.last_file_pos) {
            return false; // No new data
        }

        // Get only new content since last position
        const new_content = content[self.last_file_pos..];

        if (new_content.len == 0) return false;

        // Update position
        self.last_file_pos = file_size;

        // Parse new lines
        var has_updates = false;
        var lines = std.mem.splitScalar(u8, new_content, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            const event = metrics.MetricsParser.parseLine(line);
            self.training_metrics.update(event);
            has_updates = true;
        }

        return has_updates;
    }

    pub fn buildMetricsPath(self: *TrainingPanel) []const u8 {
        return self.metrics_path;
    }

    /// Update metrics from a single event
    pub fn updateMetric(self: *TrainingPanel, event: metrics.MetricEvent) void {
        self.training_metrics.update(event);
    }
};

// ===============================================================================
// Tests
// ===============================================================================

test "TrainingPanel initializes" {
    const theme = &themes.themes.default;

    var panel = TrainingPanel.init(std.testing.allocator, theme, .{});
    defer panel.deinit();

    try std.testing.expectEqual(Mode.live, panel.mode);
}

test "TrainingPanel handles keys" {
    const theme = &themes.themes.default;

    var panel = TrainingPanel.init(std.testing.allocator, theme, .{});
    defer panel.deinit();

    try std.testing.expectEqual(TrainingPanel.Action.quit, panel.handleKey('q'));
    try std.testing.expectEqual(TrainingPanel.Action.refresh, panel.handleKey('r'));
    try std.testing.expectEqual(TrainingPanel.Action.refresh, panel.handleKey('h'));
    try std.testing.expectEqual(Mode.history, panel.mode);
}

test "TrainingPanel help overlay toggle" {
    const theme = &themes.themes.default;

    var panel = TrainingPanel.init(std.testing.allocator, theme, .{});
    defer panel.deinit();

    // Help starts hidden
    try std.testing.expect(!panel.show_help);

    // Pressing '?' should return help action
    try std.testing.expectEqual(TrainingPanel.Action.help, panel.handleKey('?'));

    // Verify initial state for run switching
    try std.testing.expectEqual(@as(usize, 0), panel.available_runs.items.len);
    try std.testing.expectEqual(@as(usize, 0), panel.current_run_index);
}
