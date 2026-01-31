//! Model Management Panel
//!
//! Interactive TUI panel for managing AI models:
//! - View cached/downloaded models
//! - Monitor download progress
//! - Switch active model
//! - Remove models from cache
//!
//! Part of the TUI/CLI enhancement suite.

const std = @import("std");
const abi = @import("abi");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const events = @import("events.zig");
const widgets = @import("widgets.zig");
const box = widgets.box;
const RingBuffer = @import("ring_buffer.zig").RingBuffer;

/// Model Management Panel for viewing and managing AI models
pub const ModelManagementPanel = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,

    // Data
    cached_models: std.ArrayListUnmanaged(ModelEntry),
    active_downloads: std.ArrayListUnmanaged(DownloadState),
    transfer_rate_history: RingBuffer(f32, 60),

    // State
    selected_model: usize,
    scroll_offset: usize,
    active_model_id: ?[]const u8,
    show_details: bool,
    show_help: bool,

    // Polling
    last_refresh: i64,
    refresh_interval_ms: u64,

    const Self = @This();

    /// Entry representing a cached model
    pub const ModelEntry = struct {
        id: []const u8,
        name: []const u8,
        size_bytes: u64,
        path: []const u8,
        format: []const u8,
        is_active: bool,

        pub fn deinit(self: *ModelEntry, allocator: std.mem.Allocator) void {
            allocator.free(self.id);
            allocator.free(self.name);
            allocator.free(self.path);
            allocator.free(self.format);
        }
    };

    /// State of an active download
    pub const DownloadState = struct {
        model_id: []const u8,
        model_name: []const u8,
        total_bytes: u64,
        downloaded_bytes: u64,
        speed_bytes_per_sec: f32,
        eta_seconds: ?u32,
        status: DownloadStatus,

        pub fn getPercent(self: *const DownloadState) u8 {
            if (self.total_bytes == 0) return 0;
            return @intCast((self.downloaded_bytes * 100) / self.total_bytes);
        }
    };

    /// Download status states
    pub const DownloadStatus = enum {
        pending,
        downloading,
        verifying,
        completed,
        failed,
        paused,

        pub fn toString(self: DownloadStatus) []const u8 {
            return switch (self) {
                .pending => "Pending",
                .downloading => "Downloading",
                .verifying => "Verifying",
                .completed => "Completed",
                .failed => "Failed",
                .paused => "Paused",
            };
        }

        pub fn toIcon(self: DownloadStatus) []const u8 {
            return switch (self) {
                .pending => "◌",
                .downloading => "↓",
                .verifying => "◐",
                .completed => "✓",
                .failed => "✗",
                .paused => "⏸",
            };
        }
    };

    /// User actions from key input
    pub const Action = enum {
        quit,
        refresh,
        download,
        remove,
        set_active,
        show_info,
        toggle_help,
        move_up,
        move_down,
        page_up,
        page_down,
        confirm,
        cancel,
    };

    /// Initialize a new Model Management Panel
    pub fn init(
        allocator: std.mem.Allocator,
        term: *terminal.Terminal,
        theme: *const themes.Theme,
    ) Self {
        return .{
            .allocator = allocator,
            .term = term,
            .theme = theme,
            .cached_models = .empty,
            .active_downloads = .empty,
            .transfer_rate_history = RingBuffer(f32, 60).init(),
            .selected_model = 0,
            .scroll_offset = 0,
            .active_model_id = null,
            .show_details = false,
            .show_help = false,
            .last_refresh = 0,
            .refresh_interval_ms = 1000,
        };
    }

    /// Clean up panel resources
    pub fn deinit(self: *Self) void {
        for (self.cached_models.items) |*model| {
            model.deinit(self.allocator);
        }
        self.cached_models.deinit(self.allocator);
        self.active_downloads.deinit(self.allocator);
    }

    /// Update panel data (poll for changes)
    pub fn update(self: *Self) !void {
        // Get current timestamp
        const now = abi.utils.unixMs();

        // Check if enough time has passed for refresh
        if (now - self.last_refresh < @as(i64, @intCast(self.refresh_interval_ms))) {
            return;
        }
        self.last_refresh = now;

        // Poll model manager for cached models
        var manager = try abi.ai.models.Manager.init(self.allocator, .{ .auto_scan = false });
        defer manager.deinit();

        var io_backend = std.Io.Threaded.init(self.allocator, .{
            .environ = std.process.Environ.empty,
        });
        defer io_backend.deinit();

        manager.scanCacheDirWithIo(io_backend.io()) catch {};

        const prev_active = if (self.active_model_id) |id|
            self.allocator.dupe(u8, id) catch null
        else
            null;
        defer if (prev_active) |id| self.allocator.free(id);

        self.clearModels();

        for (manager.listModels()) |model| {
            const id_copy = self.allocator.dupe(u8, model.name) catch continue;
            const name_copy = self.allocator.dupe(u8, model.name) catch {
                self.allocator.free(id_copy);
                continue;
            };
            const path_copy = self.allocator.dupe(u8, model.path) catch {
                self.allocator.free(id_copy);
                self.allocator.free(name_copy);
                continue;
            };
            const format_copy = self.allocator.dupe(u8, @tagName(model.format)) catch {
                self.allocator.free(id_copy);
                self.allocator.free(name_copy);
                self.allocator.free(path_copy);
                continue;
            };

            const is_active = if (prev_active) |id|
                std.mem.eql(u8, id, model.name)
            else
                false;

            self.cached_models.append(self.allocator, .{
                .id = id_copy,
                .name = name_copy,
                .size_bytes = model.size_bytes,
                .path = path_copy,
                .format = format_copy,
                .is_active = is_active,
            }) catch {
                self.allocator.free(id_copy);
                self.allocator.free(name_copy);
                self.allocator.free(path_copy);
                self.allocator.free(format_copy);
                continue;
            };

            if (is_active) {
                self.active_model_id = id_copy;
            }
        }

        if (self.active_model_id == null and self.cached_models.items.len > 0) {
            self.cached_models.items[0].is_active = true;
            self.active_model_id = self.cached_models.items[0].id;
        }

        if (self.selected_model >= self.cached_models.items.len) {
            self.selected_model = if (self.cached_models.items.len > 0) self.cached_models.items.len - 1 else 0;
        }

        // Update transfer rate history if we have active downloads
        if (self.active_downloads.items.len > 0) {
            var total_speed: f32 = 0;
            for (self.active_downloads.items) |dl| {
                total_speed += dl.speed_bytes_per_sec;
            }
            self.transfer_rate_history.push(total_speed);
        }
    }

    /// Add a mock model for testing
    pub fn addMockModel(self: *Self, id: []const u8, name: []const u8, size_bytes: u64, is_active: bool) !void {
        const id_copy = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(id_copy);

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const path_copy = try self.allocator.dupe(u8, "~/.cache/abi/models/");
        errdefer self.allocator.free(path_copy);

        const format_copy = try self.allocator.dupe(u8, "GGUF");
        errdefer self.allocator.free(format_copy);

        try self.cached_models.append(self.allocator, .{
            .id = id_copy,
            .name = name_copy,
            .size_bytes = size_bytes,
            .path = path_copy,
            .format = format_copy,
            .is_active = is_active,
        });

        if (is_active) {
            self.active_model_id = id_copy;
        }
    }

    /// Clear all models
    pub fn clearModels(self: *Self) void {
        for (self.cached_models.items) |*model| {
            model.deinit(self.allocator);
        }
        self.cached_models.clearRetainingCapacity();
        self.active_model_id = null;
    }

    /// Render the panel
    pub fn render(
        self: *Self,
        start_row: usize,
        start_col: usize,
        width: usize,
        height: usize,
    ) !void {
        if (height < 6) return; // Minimum height required

        try self.renderHeader(start_row, start_col, width);
        try self.renderModelList(start_row + 2, start_col, width, height - 6);
        try self.renderDownloads(start_row + height - 4, start_col, width);
        try self.renderFooter(start_row + height - 1, start_col, width);
    }

    fn renderHeader(self: *Self, row: usize, col: usize, width: usize) !void {
        const title = "Model Management";
        const model_count = self.cached_models.items.len;

        try self.term.moveTo(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.tl);
        try self.term.write(box.dh);
        try self.term.write(" ");
        try self.term.write(self.theme.accent);
        try self.term.write(title);
        try self.term.write(self.theme.reset);
        try self.term.write(self.theme.text_dim);

        // Model count
        var count_buf: [32]u8 = undefined;
        const count_str = std.fmt.bufPrint(&count_buf, " ({d} cached) ", .{model_count}) catch "";
        try self.term.write(count_str);
        try self.term.write(self.theme.reset);

        // Fill remaining width with border
        const used = title.len + count_str.len + 4; // tl + dh + space + trailing
        if (used < width) {
            var i: usize = used;
            while (i < width - 1) : (i += 1) {
                try self.term.write(self.theme.border);
                try self.term.write(box.dh);
            }
        }
        try self.term.write(self.theme.border);
        try self.term.write(box.tr);
        try self.term.write(self.theme.reset);
    }

    fn renderModelList(
        self: *Self,
        start_row: usize,
        col: usize,
        width: usize,
        height: usize,
    ) !void {
        const models = self.cached_models.items;

        // Empty state
        if (models.len == 0) {
            try self.term.moveTo(start_row, col);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);
            try self.term.write(self.theme.text_dim);
            try self.term.write("  No models cached. Press 'd' to download.");
            try self.term.write(self.theme.reset);

            // Pad to right border
            try self.term.moveTo(start_row, col + width - 1);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);
            return;
        }

        const visible_count = @min(models.len, height);

        for (0..visible_count) |i| {
            const model_idx = self.scroll_offset + i;
            if (model_idx >= models.len) break;

            const model = models[model_idx];
            const is_selected = model_idx == self.selected_model;
            const row = start_row + i;

            try self.term.moveTo(row, col);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);

            // Selection highlight
            if (is_selected) {
                try self.term.write(self.theme.selection_bg);
            }

            // Active indicator
            const active_icon = if (model.is_active) "●" else "○";
            if (model.is_active) {
                try self.term.write(self.theme.success);
            } else {
                try self.term.write(self.theme.text_dim);
            }
            try self.term.write(" ");
            try self.term.write(active_icon);
            try self.term.write(" ");
            try self.term.write(self.theme.reset);

            if (is_selected) {
                try self.term.write(self.theme.selection_bg);
            }

            // Model name (truncate if needed)
            const max_name_width = if (width > 35) width - 35 else 10;
            const name_len = @min(model.name.len, max_name_width);
            try self.term.write(model.name[0..name_len]);

            // Pad name column
            var pad: usize = name_len;
            while (pad < max_name_width) : (pad += 1) {
                try self.term.write(" ");
            }

            // Format
            try self.term.write(self.theme.text_dim);
            try self.term.write("  ");
            const fmt_len = @min(model.format.len, 6);
            try self.term.write(model.format[0..fmt_len]);
            try self.term.write(self.theme.reset);

            if (is_selected) {
                try self.term.write(self.theme.selection_bg);
            }

            // Size
            try self.term.write("  ");
            var size_buf: [16]u8 = undefined;
            const size_mb = @as(f64, @floatFromInt(model.size_bytes)) / (1024 * 1024);
            const size_str = if (size_mb >= 1024)
                std.fmt.bufPrint(&size_buf, "{d:.1} GB", .{size_mb / 1024}) catch "??"
            else
                std.fmt.bufPrint(&size_buf, "{d:.0} MB", .{size_mb}) catch "??";
            try self.term.write(size_str);

            // Status
            try self.term.write("  ");
            try self.term.write(self.theme.success);
            try self.term.write("Ready");
            try self.term.write(self.theme.reset);

            if (is_selected) {
                try self.term.write(self.theme.reset);
            }

            // Right border
            try self.term.moveTo(row, col + width - 1);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);
        }

        // Fill remaining rows if needed
        var remaining_row = start_row + visible_count;
        while (remaining_row < start_row + height) : (remaining_row += 1) {
            try self.term.moveTo(remaining_row, col);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);

            try self.term.moveTo(remaining_row, col + width - 1);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);
        }
    }

    fn renderDownloads(self: *Self, row: usize, col: usize, width: usize) !void {
        // Separator line
        try self.term.moveTo(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.lsep);
        var i: usize = 1;
        while (i < width - 1) : (i += 1) {
            try self.term.write(box.h);
        }
        try self.term.write(box.rsep);
        try self.term.write(self.theme.reset);

        // Download progress section
        try self.term.moveTo(row + 1, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);

        if (self.active_downloads.items.len == 0) {
            try self.term.write(self.theme.text_dim);
            try self.term.write(" No active downloads");
            try self.term.write(self.theme.reset);
        } else {
            const dl = self.active_downloads.items[0];
            const percent = dl.getPercent();

            try self.term.write(" ");
            try self.term.write(dl.status.toIcon());
            try self.term.write(" ");

            // Model name (truncate if needed)
            const max_name = if (width > 50) 20 else 10;
            const name_len = @min(dl.model_name.len, max_name);
            try self.term.write(dl.model_name[0..name_len]);
            try self.term.write("  ");

            // Progress bar
            var bar_buf: [64]u8 = undefined;
            const bar_width: usize = if (width > 60) 25 else 15;
            const bar = widgets.ProgressGauge.render(percent, bar_width, &bar_buf);
            try self.term.write(bar);

            // Percentage
            var pct_buf: [8]u8 = undefined;
            const pct_str = std.fmt.bufPrint(&pct_buf, " {d}%", .{percent}) catch "";
            try self.term.write(pct_str);

            // Speed
            if (dl.speed_bytes_per_sec > 0) {
                var speed_buf: [16]u8 = undefined;
                const speed_mb = dl.speed_bytes_per_sec / (1024 * 1024);
                const speed_str = std.fmt.bufPrint(&speed_buf, "  {d:.1} MB/s", .{speed_mb}) catch "";
                try self.term.write(self.theme.text_dim);
                try self.term.write(speed_str);
                try self.term.write(self.theme.reset);
            }
        }

        // Right border for download row
        try self.term.moveTo(row + 1, col + width - 1);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);

        // Transfer rate sparkline row
        try self.term.moveTo(row + 2, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);

        if (self.transfer_rate_history.len() > 0) {
            try self.term.write(self.theme.text_dim);
            try self.term.write(" Transfer rate: ");
            try self.term.write(self.theme.reset);

            // Simple ASCII sparkline
            const max_sparkline = if (width > 50) 30 else 15;
            var spark_count: usize = 0;
            var iter = self.transfer_rate_history.iterator();
            const max_rate = self.transfer_rate_history.max() orelse 1.0;

            while (iter.next()) |rate| {
                if (spark_count >= max_sparkline) break;
                const normalized = if (max_rate > 0) rate / max_rate else 0;
                const spark_char: []const u8 = if (normalized > 0.75)
                    "█"
                else if (normalized > 0.5)
                    "▆"
                else if (normalized > 0.25)
                    "▄"
                else if (normalized > 0)
                    "▂"
                else
                    "▁";
                try self.term.write(self.theme.accent);
                try self.term.write(spark_char);
                spark_count += 1;
            }
            try self.term.write(self.theme.reset);
        }

        // Right border for sparkline row
        try self.term.moveTo(row + 2, col + width - 1);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);
    }

    fn renderFooter(self: *Self, row: usize, col: usize, width: usize) !void {
        try self.term.moveTo(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.bl);

        const help_text = " [d]ownload  [r]emove  [s]et active  [i]nfo  [?]help  [q]uit ";
        try self.term.write(self.theme.text_dim);
        try self.term.write(help_text);
        try self.term.write(self.theme.reset);

        // Fill remaining with border
        var i: usize = help_text.len + 1;
        while (i < width - 1) : (i += 1) {
            try self.term.write(self.theme.border);
            try self.term.write(box.dh);
        }
        try self.term.write(box.br);
        try self.term.write(self.theme.reset);
    }

    /// Handle keyboard input
    pub fn handleKey(_: *Self, key: events.Key) ?Action {
        switch (key.code) {
            .character => switch (key.char) {
                'q' => return .quit,
                'j' => return .move_down,
                'k' => return .move_up,
                'd' => return .download,
                'r' => return .remove,
                's' => return .set_active,
                'i' => return .show_info,
                '?' => return .toggle_help,
                else => {},
            },
            .up => return .move_up,
            .down => return .move_down,
            .page_up => return .page_up,
            .page_down => return .page_down,
            .enter => return .confirm,
            .escape => return .quit,
            else => {},
        }
        return null;
    }

    /// Move selection up
    pub fn moveUp(self: *Self) void {
        if (self.selected_model > 0) {
            self.selected_model -= 1;
            if (self.selected_model < self.scroll_offset) {
                self.scroll_offset = self.selected_model;
            }
        }
    }

    /// Move selection down
    pub fn moveDown(self: *Self) void {
        if (self.selected_model + 1 < self.cached_models.items.len) {
            self.selected_model += 1;
            // Adjust scroll if needed (assuming visible height of 10)
            if (self.selected_model >= self.scroll_offset + 10) {
                self.scroll_offset = self.selected_model - 9;
            }
        }
    }

    /// Page up
    pub fn pageUp(self: *Self) void {
        if (self.selected_model >= 10) {
            self.selected_model -= 10;
        } else {
            self.selected_model = 0;
        }
        if (self.selected_model < self.scroll_offset) {
            self.scroll_offset = self.selected_model;
        }
    }

    /// Page down
    pub fn pageDown(self: *Self) void {
        const max_idx = if (self.cached_models.items.len > 0)
            self.cached_models.items.len - 1
        else
            0;

        if (self.selected_model + 10 < max_idx) {
            self.selected_model += 10;
        } else {
            self.selected_model = max_idx;
        }
        if (self.selected_model >= self.scroll_offset + 10) {
            self.scroll_offset = self.selected_model - 9;
        }
    }

    /// Get currently selected model
    pub fn getSelectedModel(self: *const Self) ?*const ModelEntry {
        if (self.selected_model < self.cached_models.items.len) {
            return &self.cached_models.items[self.selected_model];
        }
        return null;
    }

    /// Set a model as active
    pub fn setActiveModel(self: *Self, model_id: []const u8) void {
        for (self.cached_models.items) |*model| {
            model.is_active = std.mem.eql(u8, model.id, model_id);
            if (model.is_active) {
                self.active_model_id = model.id;
            }
        }
    }

    /// Toggle help display
    pub fn toggleHelp(self: *Self) void {
        self.show_help = !self.show_help;
    }

    /// Get number of cached models
    pub fn modelCount(self: *const Self) usize {
        return self.cached_models.items.len;
    }

    /// Get number of active downloads
    pub fn downloadCount(self: *const Self) usize {
        return self.active_downloads.items.len;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "ModelManagementPanel initializes correctly" {
    // Compile-time check - we can't easily test rendering without a real terminal
    const allocator = std.testing.allocator;
    _ = allocator;
    _ = ModelManagementPanel;
}

test "ModelEntry size formatting" {
    // Test size calculation logic
    const size_1gb: u64 = 1024 * 1024 * 1024;
    const size_mb = @as(f64, @floatFromInt(size_1gb)) / (1024 * 1024);
    try std.testing.expect(size_mb >= 1024);

    const size_500mb: u64 = 500 * 1024 * 1024;
    const size_mb2 = @as(f64, @floatFromInt(size_500mb)) / (1024 * 1024);
    try std.testing.expect(size_mb2 < 1024);
}

test "DownloadState percentage calculation" {
    const dl = ModelManagementPanel.DownloadState{
        .model_id = "test",
        .model_name = "Test Model",
        .total_bytes = 1000,
        .downloaded_bytes = 500,
        .speed_bytes_per_sec = 100,
        .eta_seconds = 5,
        .status = .downloading,
    };

    try std.testing.expectEqual(@as(u8, 50), dl.getPercent());
}

test "DownloadState zero total bytes" {
    const dl = ModelManagementPanel.DownloadState{
        .model_id = "test",
        .model_name = "Test",
        .total_bytes = 0,
        .downloaded_bytes = 0,
        .speed_bytes_per_sec = 0,
        .eta_seconds = null,
        .status = .pending,
    };

    try std.testing.expectEqual(@as(u8, 0), dl.getPercent());
}

test "DownloadStatus toString" {
    try std.testing.expectEqualStrings("Downloading", ModelManagementPanel.DownloadStatus.downloading.toString());
    try std.testing.expectEqualStrings("Completed", ModelManagementPanel.DownloadStatus.completed.toString());
    try std.testing.expectEqualStrings("Failed", ModelManagementPanel.DownloadStatus.failed.toString());
}

test "DownloadStatus toIcon" {
    try std.testing.expectEqualStrings("↓", ModelManagementPanel.DownloadStatus.downloading.toIcon());
    try std.testing.expectEqualStrings("✓", ModelManagementPanel.DownloadStatus.completed.toIcon());
    try std.testing.expectEqualStrings("✗", ModelManagementPanel.DownloadStatus.failed.toIcon());
}

test "Action enum values" {
    // Ensure all actions are defined
    const actions = [_]ModelManagementPanel.Action{
        .quit,
        .refresh,
        .download,
        .remove,
        .set_active,
        .show_info,
        .toggle_help,
        .move_up,
        .move_down,
        .page_up,
        .page_down,
        .confirm,
        .cancel,
    };
    try std.testing.expectEqual(@as(usize, 13), actions.len);
}
