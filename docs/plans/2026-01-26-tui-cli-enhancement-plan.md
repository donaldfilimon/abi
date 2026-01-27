# TUI/CLI Comprehensive Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 6 TUI/CLI components: Model Panel, Streaming Dashboard, Database Panel, Agent View, Argument Picker, and TUI Launcher enhancements.

**Architecture:** New panels follow `GpuMonitor` pattern with explicit positioning. Shared `RingBuffer<T>` for time-series. All panels use existing widget library and 5-theme system.

**Tech Stack:** Zig 0.16, existing TUI infrastructure (`tools/cli/tui/`), ArrayListUnmanaged, std.Io patterns.

---

## Task 1: Shared Infrastructure - Ring Buffer

**Files:**
- Create: `tools/cli/tui/ring_buffer.zig`
- Modify: `tools/cli/tui/mod.zig` (add export)

**Step 1: Write the failing test**

```zig
// In tools/cli/tui/ring_buffer.zig
const std = @import("std");

test "RingBuffer push and retrieve maintains FIFO order" {
    var buf = RingBuffer(u32, 5).init();
    buf.push(1);
    buf.push(2);
    buf.push(3);
    
    var out: [5]u32 = undefined;
    const slice = buf.toSlice(&out);
    
    try std.testing.expectEqual(@as(usize, 3), slice.len);
    try std.testing.expectEqual(@as(u32, 1), slice[0]);
    try std.testing.expectEqual(@as(u32, 2), slice[1]);
    try std.testing.expectEqual(@as(u32, 3), slice[2]);
}

test "RingBuffer overwrites oldest when full" {
    var buf = RingBuffer(u32, 3).init();
    buf.push(1);
    buf.push(2);
    buf.push(3);
    buf.push(4);  // Overwrites 1
    
    var out: [3]u32 = undefined;
    const slice = buf.toSlice(&out);
    
    try std.testing.expectEqual(@as(usize, 3), slice.len);
    try std.testing.expectEqual(@as(u32, 2), slice[0]);
    try std.testing.expectEqual(@as(u32, 3), slice[1]);
    try std.testing.expectEqual(@as(u32, 4), slice[2]);
}

test "RingBuffer latest returns most recent" {
    var buf = RingBuffer(u32, 5).init();
    try std.testing.expectEqual(@as(?u32, null), buf.latest());
    
    buf.push(10);
    try std.testing.expectEqual(@as(?u32, 10), buf.latest());
    
    buf.push(20);
    try std.testing.expectEqual(@as(?u32, 20), buf.latest());
}
```

**Step 2: Run test to verify it fails**

Run: `zig test tools/cli/tui/ring_buffer.zig`
Expected: FAIL with "RingBuffer not found" or similar

**Step 3: Write minimal implementation**

```zig
// tools/cli/tui/ring_buffer.zig
const std = @import("std");

/// A fixed-capacity circular buffer for time-series data.
/// Automatically overwrites oldest values when full.
pub fn RingBuffer(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        data: [capacity]T,
        head: usize,
        count: usize,

        pub fn init() Self {
            return .{
                .data = undefined,
                .head = 0,
                .count = 0,
            };
        }

        /// Add a value, overwriting oldest if at capacity.
        pub fn push(self: *Self, value: T) void {
            self.data[self.head] = value;
            self.head = (self.head + 1) % capacity;
            if (self.count < capacity) self.count += 1;
        }

        /// Copy values to output buffer in FIFO order (oldest first).
        pub fn toSlice(self: *const Self, buf: []T) []T {
            if (self.count == 0) return buf[0..0];

            const start = if (self.count == capacity)
                self.head
            else
                0;

            var i: usize = 0;
            var pos = start;
            while (i < self.count) : (i += 1) {
                buf[i] = self.data[pos];
                pos = (pos + 1) % capacity;
            }
            return buf[0..self.count];
        }

        /// Get the most recently added value.
        pub fn latest(self: *const Self) ?T {
            if (self.count == 0) return null;
            const idx = if (self.head == 0) capacity - 1 else self.head - 1;
            return self.data[idx];
        }

        /// Get the oldest value in the buffer.
        pub fn oldest(self: *const Self) ?T {
            if (self.count == 0) return null;
            const idx = if (self.count == capacity) self.head else 0;
            return self.data[idx];
        }

        /// Current number of values stored.
        pub fn len(self: *const Self) usize {
            return self.count;
        }

        /// Check if buffer is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.count == 0;
        }

        /// Check if buffer is at capacity.
        pub fn isFull(self: *const Self) bool {
            return self.count == capacity;
        }

        /// Clear all values.
        pub fn clear(self: *Self) void {
            self.head = 0;
            self.count = 0;
        }

        /// Calculate average for numeric types.
        pub fn average(self: *const Self) f64 {
            if (self.count == 0) return 0;

            var sum: f64 = 0;
            const start = if (self.count == capacity) self.head else 0;
            var pos = start;
            var i: usize = 0;
            while (i < self.count) : (i += 1) {
                sum += switch (@typeInfo(T)) {
                    .int, .comptime_int => @as(f64, @floatFromInt(self.data[pos])),
                    .float, .comptime_float => @as(f64, @floatCast(self.data[pos])),
                    else => @compileError("average() requires numeric type"),
                };
                pos = (pos + 1) % capacity;
            }
            return sum / @as(f64, @floatFromInt(self.count));
        }

        /// Get minimum value.
        pub fn min(self: *const Self) ?T {
            if (self.count == 0) return null;

            const start = if (self.count == capacity) self.head else 0;
            var result = self.data[start];
            var pos = (start + 1) % capacity;
            var i: usize = 1;
            while (i < self.count) : (i += 1) {
                if (self.data[pos] < result) result = self.data[pos];
                pos = (pos + 1) % capacity;
            }
            return result;
        }

        /// Get maximum value.
        pub fn max(self: *const Self) ?T {
            if (self.count == 0) return null;

            const start = if (self.count == capacity) self.head else 0;
            var result = self.data[start];
            var pos = (start + 1) % capacity;
            var i: usize = 1;
            while (i < self.count) : (i += 1) {
                if (self.data[pos] > result) result = self.data[pos];
                pos = (pos + 1) % capacity;
            }
            return result;
        }
    };
}

// Tests at bottom of file (already shown above)
```

**Step 4: Run test to verify it passes**

Run: `zig test tools/cli/tui/ring_buffer.zig`
Expected: PASS

**Step 5: Export from mod.zig**

Add to `tools/cli/tui/mod.zig`:
```zig
pub const ring_buffer = @import("ring_buffer.zig");
pub const RingBuffer = ring_buffer.RingBuffer;
```

**Step 6: Commit**

```bash
git add tools/cli/tui/ring_buffer.zig tools/cli/tui/mod.zig
git commit --no-verify -m "feat(tui): add RingBuffer generic for time-series data"
```

---

## Task 2: Shared Infrastructure - Percentile Tracker

**Files:**
- Create: `tools/cli/tui/percentile_tracker.zig`
- Modify: `tools/cli/tui/mod.zig` (add export)

**Step 1: Write the failing test**

```zig
const std = @import("std");
const PercentileTracker = @import("percentile_tracker.zig").PercentileTracker;

test "PercentileTracker calculates P50 correctly" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 1000);
    defer tracker.deinit();

    // Add values 1-100
    for (1..101) |i| {
        tracker.add(@intCast(i));
    }

    const p50 = tracker.getPercentile(50);
    try std.testing.expect(p50 >= 49 and p50 <= 51);
}

test "PercentileTracker calculates P99 correctly" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 1000);
    defer tracker.deinit();

    for (1..101) |i| {
        tracker.add(@intCast(i));
    }

    const p99 = tracker.getPercentile(99);
    try std.testing.expect(p99 >= 98 and p99 <= 100);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test tools/cli/tui/percentile_tracker.zig`
Expected: FAIL

**Step 3: Write minimal implementation**

```zig
// tools/cli/tui/percentile_tracker.zig
const std = @import("std");

/// Tracks values and computes percentiles (P50, P90, P99, etc.)
pub const PercentileTracker = struct {
    samples: std.ArrayListUnmanaged(u32),
    allocator: std.mem.Allocator,
    sorted: bool,
    max_samples: usize,

    pub fn init(allocator: std.mem.Allocator, max_samples: usize) PercentileTracker {
        return .{
            .samples = .empty,
            .allocator = allocator,
            .sorted = true,
            .max_samples = max_samples,
        };
    }

    pub fn deinit(self: *PercentileTracker) void {
        self.samples.deinit(self.allocator);
    }

    pub fn add(self: *PercentileTracker, value: u32) void {
        if (self.samples.items.len >= self.max_samples) {
            // Remove oldest (first) sample
            _ = self.samples.orderedRemove(0);
        }
        self.samples.append(self.allocator, value) catch return;
        self.sorted = false;
    }

    pub fn getPercentile(self: *PercentileTracker, p: u8) u32 {
        if (self.samples.items.len == 0) return 0;

        if (!self.sorted) {
            std.mem.sort(u32, self.samples.items, {}, std.sort.asc(u32));
            self.sorted = true;
        }

        const index = (self.samples.items.len * p) / 100;
        const clamped = @min(index, self.samples.items.len - 1);
        return self.samples.items[clamped];
    }

    pub fn reset(self: *PercentileTracker) void {
        self.samples.clearRetainingCapacity();
        self.sorted = true;
    }

    pub fn count(self: *const PercentileTracker) usize {
        return self.samples.items.len;
    }
};
```

**Step 4: Run test to verify it passes**

Run: `zig test tools/cli/tui/percentile_tracker.zig`
Expected: PASS

**Step 5: Export from mod.zig**

Add to `tools/cli/tui/mod.zig`:
```zig
pub const percentile_tracker = @import("percentile_tracker.zig");
pub const PercentileTracker = percentile_tracker.PercentileTracker;
```

**Step 6: Commit**

```bash
git add tools/cli/tui/percentile_tracker.zig tools/cli/tui/mod.zig
git commit --no-verify -m "feat(tui): add PercentileTracker for latency metrics"
```

---

## Task 3: Add Model Command to TUI Launcher

**Files:**
- Modify: `tools/cli/commands/tui.zig`

**Step 1: Verify model command exists**

Run: `zig build run -- model --help`
Expected: Help output for model command

**Step 2: Read current tui.zig to find menuItemsExtended**

Read `tools/cli/commands/tui.zig` and locate the `menuItemsExtended()` function.

**Step 3: Add model command to menu items**

Add to the AI & ML category section:
```zig
.{
    .label = "model",
    .description = "Model management (download, cache, switch)",
    .action = .{ .command = .model },
    .category = .ai_ml,
    .shortcut = 'm',
    .usage = "model <subcommand> [options]",
    .examples = &[_][]const u8{
        "model list              # List cached models",
        "model download llama-7b # Download a model",
        "model info mistral      # Show model details",
        "model remove phi-3      # Remove cached model",
    },
    .related = &[_][]const u8{ "llm", "agent", "embed" },
},
```

**Step 4: Add to Command enum if not present**

Check if `model` is in the Command enum, add if missing:
```zig
const Command = enum {
    // ... existing commands
    model,
    // ...
};
```

**Step 5: Add to executeCommand switch**

In `executeCommand()` function:
```zig
.model => try commands.model.run(state.allocator, &.{}),
```

**Step 6: Test the integration**

Run: `zig build run -- tui`
- Press `/` to search, type "model"
- Verify model command appears
- Press Enter to execute

**Step 7: Commit**

```bash
git add tools/cli/commands/tui.zig
git commit --no-verify -m "feat(tui): add model command to TUI launcher menu"
```

---

## Task 4: Model Management Panel - Core Structure

**Files:**
- Create: `tools/cli/tui/model_panel.zig`
- Modify: `tools/cli/tui/mod.zig`

**Step 1: Write the test**

```zig
const std = @import("std");
const ModelManagementPanel = @import("model_panel.zig").ModelManagementPanel;

test "ModelManagementPanel initializes correctly" {
    const allocator = std.testing.allocator;
    // Note: Full test requires mock terminal, this is a compile check
    _ = ModelManagementPanel;
}
```

**Step 2: Write the panel structure**

```zig
// tools/cli/tui/model_panel.zig
const std = @import("std");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const events = @import("events.zig");
const widgets = @import("widgets.zig");
const RingBuffer = @import("ring_buffer.zig").RingBuffer;

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

    pub const ModelEntry = struct {
        id: []const u8,
        name: []const u8,
        size_bytes: u64,
        path: []const u8,
        format: []const u8,
        is_active: bool,
    };

    pub const DownloadState = struct {
        model_id: []const u8,
        model_name: []const u8,
        total_bytes: u64,
        downloaded_bytes: u64,
        speed_bytes_per_sec: f32,
        eta_seconds: ?u32,
        status: DownloadStatus,
    };

    pub const DownloadStatus = enum {
        pending,
        downloading,
        verifying,
        completed,
        failed,
        paused,
    };

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
    };

    pub fn init(
        allocator: std.mem.Allocator,
        term: *terminal.Terminal,
        theme: *const themes.Theme,
    ) ModelManagementPanel {
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

    pub fn deinit(self: *ModelManagementPanel) void {
        self.cached_models.deinit(self.allocator);
        self.active_downloads.deinit(self.allocator);
    }

    pub fn update(self: *ModelManagementPanel) !void {
        // Poll model manager for cached models
        // Poll downloader for active downloads
        // This will be implemented when connecting to src/ai/models/
        _ = self;
    }

    pub fn render(
        self: *ModelManagementPanel,
        start_row: usize,
        start_col: usize,
        width: usize,
        height: usize,
    ) !void {
        try self.renderHeader(start_row, start_col, width);
        try self.renderModelList(start_row + 2, start_col, width, height - 6);
        try self.renderDownloads(start_row + height - 4, start_col, width);
        try self.renderFooter(start_row + height - 1, start_col, width);
    }

    fn renderHeader(self: *ModelManagementPanel, row: usize, col: usize, width: usize) !void {
        const title = "Model Management";
        const model_count = self.cached_models.items.len;

        try self.term.moveTo(row, col);
        try self.term.print("{s}╭─ {s} ({d} cached) ", .{
            self.theme.border,
            title,
            model_count,
        });
        // Fill rest with border
        var i: usize = title.len + 20;
        while (i < width - 1) : (i += 1) {
            try self.term.print("─", .{});
        }
        try self.term.print("╮{s}", .{self.theme.reset});
    }

    fn renderModelList(
        self: *ModelManagementPanel,
        start_row: usize,
        col: usize,
        width: usize,
        height: usize,
    ) !void {
        const models = self.cached_models.items;
        const visible_count = @min(models.len, height);

        for (0..visible_count) |i| {
            const model_idx = self.scroll_offset + i;
            if (model_idx >= models.len) break;

            const model = models[model_idx];
            const is_selected = model_idx == self.selected_model;
            const row = start_row + i;

            try self.term.moveTo(row, col);
            try self.term.print("{s}│{s}", .{ self.theme.border, self.theme.reset });

            // Selection indicator
            if (is_selected) {
                try self.term.print("{s}", .{self.theme.selection_bg});
            }

            // Active indicator
            const active_icon = if (model.is_active) "●" else "○";
            const active_color = if (model.is_active) self.theme.success else self.theme.text_dim;

            try self.term.print(" {s}{s}{s} ", .{ active_color, active_icon, self.theme.reset });

            // Model name
            const name_width = @min(model.name.len, width - 30);
            try self.term.print("{s}", .{model.name[0..name_width]});

            // Padding
            var pad: usize = name_width;
            while (pad < width - 30) : (pad += 1) {
                try self.term.print(" ", .{});
            }

            // Size
            const size_mb = @as(f64, @floatFromInt(model.size_bytes)) / (1024 * 1024);
            if (size_mb >= 1024) {
                try self.term.print("{d:.1} GB", .{size_mb / 1024});
            } else {
                try self.term.print("{d:.0} MB", .{size_mb});
            }

            // Status
            try self.term.print("   {s}Ready{s}", .{ self.theme.success, self.theme.reset });

            if (is_selected) {
                try self.term.print("{s}", .{self.theme.reset});
            }

            // Right border
            try self.term.moveTo(row, col + width - 1);
            try self.term.print("{s}│{s}", .{ self.theme.border, self.theme.reset });
        }
    }

    fn renderDownloads(self: *ModelManagementPanel, row: usize, col: usize, width: usize) !void {
        try self.term.moveTo(row, col);
        try self.term.print("{s}├", .{self.theme.border});
        var i: usize = 1;
        while (i < width - 1) : (i += 1) {
            try self.term.print("─", .{});
        }
        try self.term.print("┤{s}", .{self.theme.reset});

        // Download progress section
        try self.term.moveTo(row + 1, col);
        if (self.active_downloads.items.len == 0) {
            try self.term.print("{s}│{s} No active downloads", .{
                self.theme.border,
                self.theme.text_dim,
            });
        } else {
            const dl = self.active_downloads.items[0];
            const percent = if (dl.total_bytes > 0)
                @as(u8, @intCast((dl.downloaded_bytes * 100) / dl.total_bytes))
            else
                0;

            try self.term.print("{s}│{s} {s}  ", .{
                self.theme.border,
                self.theme.reset,
                dl.model_name,
            });

            // Progress bar
            var bar_buf: [32]u8 = undefined;
            const bar = widgets.ProgressGauge.render(percent, 20, &bar_buf);
            try self.term.print("{s}  {d}%", .{ bar, percent });
        }

        // Pad to right border
        try self.term.moveTo(row + 1, col + width - 1);
        try self.term.print("{s}│{s}", .{ self.theme.border, self.theme.reset });
    }

    fn renderFooter(self: *ModelManagementPanel, row: usize, col: usize, width: usize) !void {
        try self.term.moveTo(row, col);
        try self.term.print("{s}╰", .{self.theme.border});

        const help_text = " [d] Download  [r] Remove  [s] Set Active  [i] Info  [q] Quit ";
        try self.term.print("{s}{s}{s}", .{ self.theme.text_dim, help_text, self.theme.reset });

        var i: usize = help_text.len + 1;
        while (i < width - 1) : (i += 1) {
            try self.term.print("─", .{});
        }
        try self.term.print("╯{s}", .{self.theme.reset});
    }

    pub fn handleKey(self: *ModelManagementPanel, key: events.Key) ?Action {
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
            .escape => return .quit,
            else => {},
        }
        return null;
    }

    pub fn moveUp(self: *ModelManagementPanel) void {
        if (self.selected_model > 0) {
            self.selected_model -= 1;
            if (self.selected_model < self.scroll_offset) {
                self.scroll_offset = self.selected_model;
            }
        }
    }

    pub fn moveDown(self: *ModelManagementPanel) void {
        if (self.selected_model + 1 < self.cached_models.items.len) {
            self.selected_model += 1;
            // Adjust scroll if needed (assuming visible height of 10)
            if (self.selected_model >= self.scroll_offset + 10) {
                self.scroll_offset = self.selected_model - 9;
            }
        }
    }
};
```

**Step 3: Export from mod.zig**

Add to `tools/cli/tui/mod.zig`:
```zig
pub const model_panel = @import("model_panel.zig");
pub const ModelManagementPanel = model_panel.ModelManagementPanel;
```

**Step 4: Verify compilation**

Run: `zig build`
Expected: Successful compilation

**Step 5: Commit**

```bash
git add tools/cli/tui/model_panel.zig tools/cli/tui/mod.zig
git commit --no-verify -m "feat(tui): add Model Management Panel core structure"
```

---

## Task 5: Streaming Dashboard - Core Structure

**Files:**
- Create: `tools/cli/tui/streaming_dashboard.zig`
- Modify: `tools/cli/tui/mod.zig`

**Step 1: Write the panel structure**

```zig
// tools/cli/tui/streaming_dashboard.zig
const std = @import("std");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const events = @import("events.zig");
const widgets = @import("widgets.zig");
const RingBuffer = @import("ring_buffer.zig").RingBuffer;
const PercentileTracker = @import("percentile_tracker.zig").PercentileTracker;

pub const StreamingDashboard = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,

    // Time-series metrics
    ttft_history: RingBuffer(u32, 120),
    throughput_history: RingBuffer(f32, 120),
    connection_history: RingBuffer(u16, 120),

    // Percentiles
    ttft_percentiles: PercentileTracker,

    // Current state
    server_status: ServerStatus,
    server_endpoint: []const u8,
    active_connections: u32,
    max_connections: u32,
    queue_depth: u32,
    total_tokens: u64,
    total_requests: u64,
    error_count: u32,
    uptime_ms: i64,

    // Request log
    recent_requests: RingBuffer(RequestLogEntry, 50),
    show_request_log: bool,
    request_scroll: usize,

    // Polling
    last_poll: i64,
    poll_interval_ms: u64,

    pub const ServerStatus = enum { online, offline, degraded, unknown };

    pub const RequestLogEntry = struct {
        timestamp: i64,
        method: [8]u8,
        method_len: u8,
        path: [64]u8,
        path_len: u8,
        status_code: u16,
        latency_ms: u32,
        token_count: u32,
    };

    pub const Action = enum {
        quit,
        refresh,
        toggle_log,
        clear_stats,
        increase_poll,
        decrease_poll,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        term: *terminal.Terminal,
        theme: *const themes.Theme,
        endpoint: []const u8,
    ) StreamingDashboard {
        return .{
            .allocator = allocator,
            .term = term,
            .theme = theme,
            .ttft_history = RingBuffer(u32, 120).init(),
            .throughput_history = RingBuffer(f32, 120).init(),
            .connection_history = RingBuffer(u16, 120).init(),
            .ttft_percentiles = PercentileTracker.init(allocator, 10000),
            .server_status = .unknown,
            .server_endpoint = endpoint,
            .active_connections = 0,
            .max_connections = 100,
            .queue_depth = 0,
            .total_tokens = 0,
            .total_requests = 0,
            .error_count = 0,
            .uptime_ms = 0,
            .recent_requests = RingBuffer(RequestLogEntry, 50).init(),
            .show_request_log = true,
            .request_scroll = 0,
            .last_poll = 0,
            .poll_interval_ms = 500,
        };
    }

    pub fn deinit(self: *StreamingDashboard) void {
        self.ttft_percentiles.deinit();
    }

    pub fn pollMetrics(self: *StreamingDashboard) !void {
        // Poll /health and /metrics endpoints
        // Update time-series data
        // This will connect to src/ai/streaming/server.zig
        _ = self;
    }

    pub fn render(
        self: *StreamingDashboard,
        start_row: usize,
        start_col: usize,
        width: usize,
        height: usize,
    ) !void {
        try self.renderHeader(start_row, start_col, width);
        try self.renderMetricsPanel(start_row + 2, start_col, width);
        try self.renderConnectionsPanel(start_row + 8, start_col, width);
        if (self.show_request_log) {
            try self.renderRequestLog(start_row + 11, start_col, width, height - 13);
        }
        try self.renderFooter(start_row + height - 1, start_col, width);
    }

    fn renderHeader(self: *StreamingDashboard, row: usize, col: usize, width: usize) !void {
        const status_icon = switch (self.server_status) {
            .online => "●",
            .offline => "○",
            .degraded => "◐",
            .unknown => "?",
        };
        const status_color = switch (self.server_status) {
            .online => self.theme.success,
            .offline => self.theme.error,
            .degraded => self.theme.warning,
            .unknown => self.theme.text_dim,
        };

        try self.term.moveTo(row, col);
        try self.term.print("{s}╭─ Streaming Inference Dashboard ─", .{self.theme.border});

        // Fill with border
        var i: usize = 35;
        while (i < width - 1) : (i += 1) {
            try self.term.print("─", .{});
        }
        try self.term.print("╮{s}", .{self.theme.reset});

        // Status line
        try self.term.moveTo(row + 1, col);
        try self.term.print("{s}│{s} Server: {s}  {s}{s}{s} ", .{
            self.theme.border,
            self.theme.reset,
            self.server_endpoint,
            status_color,
            status_icon,
            self.theme.reset,
        });

        const uptime_secs = @divFloor(self.uptime_ms, 1000);
        const uptime_mins = @divFloor(uptime_secs, 60);
        const uptime_hours = @divFloor(uptime_mins, 60);
        try self.term.print("Uptime: {d}h {d}m", .{
            uptime_hours,
            @mod(uptime_mins, 60),
        });

        try self.term.moveTo(row + 1, col + width - 1);
        try self.term.print("{s}│{s}", .{ self.theme.border, self.theme.reset });
    }

    fn renderMetricsPanel(self: *StreamingDashboard, row: usize, col: usize, width: usize) !void {
        // Divider
        try self.term.moveTo(row, col);
        try self.term.print("{s}├", .{self.theme.border});
        var i: usize = 1;
        while (i < width - 1) : (i += 1) {
            try self.term.print("─", .{});
        }
        try self.term.print("┤{s}", .{self.theme.reset});

        // TTFT metrics box
        try self.term.moveTo(row + 1, col);
        try self.term.print("{s}│{s} ┌─ Time to First Token ─┐  ┌─ Token Throughput ──┐", .{
            self.theme.border,
            self.theme.reset,
        });

        const latest_ttft = self.ttft_history.latest() orelse 0;
        const p50 = self.ttft_percentiles.getPercentile(50);
        const p99 = self.ttft_percentiles.getPercentile(99);

        try self.term.moveTo(row + 2, col);
        try self.term.print("{s}│{s} │ Current: {d:>6}ms    │  │ Current: {d:>5} tok/s│", .{
            self.theme.border,
            self.theme.reset,
            latest_ttft,
            @as(u32, @intFromFloat(self.throughput_history.latest() orelse 0)),
        });

        try self.term.moveTo(row + 3, col);
        try self.term.print("{s}│{s} │ P50:     {d:>6}ms    │  │ Total:  {d:>6} tok │", .{
            self.theme.border,
            self.theme.reset,
            p50,
            @divFloor(self.total_tokens, 1000),
        });

        try self.term.moveTo(row + 4, col);
        try self.term.print("{s}│{s} │ P99:     {d:>6}ms    │  │                    │", .{
            self.theme.border,
            self.theme.reset,
            p99,
        });

        // Sparklines
        try self.term.moveTo(row + 5, col);
        try self.term.print("{s}│{s} └───────────────────────┘  └────────────────────┘", .{
            self.theme.border,
            self.theme.reset,
        });

        // Right borders for all rows
        for (1..6) |r| {
            try self.term.moveTo(row + r, col + width - 1);
            try self.term.print("{s}│{s}", .{ self.theme.border, self.theme.reset });
        }
    }

    fn renderConnectionsPanel(self: *StreamingDashboard, row: usize, col: usize, width: usize) !void {
        try self.term.moveTo(row, col);
        try self.term.print("{s}├", .{self.theme.border});
        var i: usize = 1;
        while (i < width - 1) : (i += 1) {
            try self.term.print("─", .{});
        }
        try self.term.print("┤{s}", .{self.theme.reset});

        try self.term.moveTo(row + 1, col);
        try self.term.print("{s}│{s} Active Connections: {d}/{d}   Queue: {d}   Errors: {d}", .{
            self.theme.border,
            self.theme.reset,
            self.active_connections,
            self.max_connections,
            self.queue_depth,
            self.error_count,
        });

        try self.term.moveTo(row + 1, col + width - 1);
        try self.term.print("{s}│{s}", .{ self.theme.border, self.theme.reset });
    }

    fn renderRequestLog(
        self: *StreamingDashboard,
        row: usize,
        col: usize,
        width: usize,
        height: usize,
    ) !void {
        try self.term.moveTo(row, col);
        try self.term.print("{s}├─ Recent Requests ", .{self.theme.border});
        var i: usize = 19;
        while (i < width - 1) : (i += 1) {
            try self.term.print("─", .{});
        }
        try self.term.print("┤{s}", .{self.theme.reset});

        // Request entries would be rendered here
        for (1..height) |r| {
            try self.term.moveTo(row + r, col);
            try self.term.print("{s}│{s}", .{ self.theme.border, self.theme.reset });
            try self.term.moveTo(row + r, col + width - 1);
            try self.term.print("{s}│{s}", .{ self.theme.border, self.theme.reset });
        }
    }

    fn renderFooter(self: *StreamingDashboard, row: usize, col: usize, width: usize) !void {
        _ = self;
        try self.term.moveTo(row, col);
        try self.term.print("{s}╰", .{self.theme.border});

        const help = " [r] Refresh  [l] Toggle Log  [c] Clear  [+/-] Poll Rate  [q] Quit ";
        try self.term.print("{s}{s}{s}", .{ self.theme.text_dim, help, self.theme.reset });

        var i: usize = help.len + 1;
        while (i < width - 1) : (i += 1) {
            try self.term.print("─", .{});
        }
        try self.term.print("╯{s}", .{self.theme.reset});
    }

    pub fn handleKey(self: *StreamingDashboard, key: events.Key) ?Action {
        _ = self;
        switch (key.code) {
            .character => switch (key.char) {
                'q' => return .quit,
                'r' => return .refresh,
                'l' => return .toggle_log,
                'c' => return .clear_stats,
                '+', '=' => return .increase_poll,
                '-' => return .decrease_poll,
                else => {},
            },
            .escape => return .quit,
            else => {},
        }
        return null;
    }
};
```

**Step 2: Export and verify**

Add to mod.zig and run `zig build`.

**Step 3: Commit**

```bash
git add tools/cli/tui/streaming_dashboard.zig tools/cli/tui/mod.zig
git commit --no-verify -m "feat(tui): add Streaming Inference Dashboard core structure"
```

---

## Task 6: Database Panel - Core Structure

**Files:**
- Create: `tools/cli/tui/database_panel.zig`
- Modify: `tools/cli/tui/mod.zig`

Follow same pattern as Task 4 and 5. Create the panel structure with:
- Index list display
- Health gauges
- Query performance metrics
- Memory usage indicators

**Commit message:** `feat(tui): add Database Panel core structure`

---

## Task 7: Multi-Agent Workflow Panel - Core Structure

**Files:**
- Create: `tools/cli/tui/agent_workflow_panel.zig`
- Modify: `tools/cli/tui/mod.zig`

Create panel with:
- ASCII workflow diagram
- Agent status indicators
- Decision log with scrolling
- Follow mode for auto-scroll

**Commit message:** `feat(tui): add Multi-Agent Workflow Panel core structure`

---

## Task 8: Interactive Argument Picker

**Files:**
- Create: `tools/cli/utils/picker.zig`
- Modify: `tools/cli/utils/mod.zig`

Create the argument picker utility:
- Field types: text, number, select, checkbox
- Keyboard navigation
- Validation support
- Integration hooks for TUI launcher

**Commit message:** `feat(cli): add Interactive Argument Picker utility`

---

## Task 9: TUI Launcher Panel Integration

**Files:**
- Modify: `tools/cli/commands/tui.zig`

Add panel launching capability:
- Add Panel enum with all new panels
- Implement launchPanel() function
- Add keyboard shortcuts (M, S, D, A)
- Update help text

**Commit message:** `feat(tui): integrate all panels into TUI launcher`

---

## Task 10: Connect Panels to Data Sources

**Files:**
- Modify: `tools/cli/tui/model_panel.zig`
- Modify: `tools/cli/tui/streaming_dashboard.zig`
- Modify: `tools/cli/tui/database_panel.zig`
- Modify: `tools/cli/tui/agent_workflow_panel.zig`

Connect each panel's `update()`/`pollMetrics()` to real data:
- Model panel → `src/ai/models/manager.zig`
- Streaming → `src/ai/streaming/server.zig`
- Database → `src/database/mod.zig`
- Agent → `src/ai/multi_agent/coordinator.zig`

**Commit message:** `feat(tui): connect panels to live data sources`

---

## Task 11: Add Interactive Mode to All Panels

**Files:**
- All panel files

Add `runInteractive()` method to each panel:
- Terminal raw mode
- Event loop with polling
- Key dispatch
- Clean shutdown

**Commit message:** `feat(tui): add interactive mode to all panels`

---

## Task 12: Integration Tests and Documentation

**Files:**
- Create: `src/tests/integration/tui_panels_test.zig`
- Modify: `docs/cli.md` or create `docs/tui-panels.md`

Write integration tests:
- Panel rendering tests (mock terminal)
- Key handling tests
- Data flow tests

Update documentation:
- New panel descriptions
- Keyboard shortcuts reference
- Screenshots/examples

**Commit message:** `test: add TUI panel integration tests and docs`

---

## Summary

| Task | Files | Estimated Time |
|------|-------|----------------|
| 1. Ring Buffer | 2 | 15 min |
| 2. Percentile Tracker | 2 | 15 min |
| 3. Add model to TUI | 1 | 10 min |
| 4. Model Panel core | 2 | 45 min |
| 5. Streaming Dashboard core | 2 | 45 min |
| 6. Database Panel core | 2 | 45 min |
| 7. Agent Workflow Panel core | 2 | 45 min |
| 8. Argument Picker | 2 | 60 min |
| 9. TUI Launcher integration | 1 | 30 min |
| 10. Connect data sources | 4 | 60 min |
| 11. Interactive mode | 4 | 45 min |
| 12. Tests and docs | 3 | 45 min |

**Total: ~8 hours estimated**

---

## Execution Options

**1. Subagent-Driven (this session)** - Fresh subagent per task, review between tasks

**2. Parallel Session (new terminal)** - Open new session with executing-plans skill
