//! Memory tracking dashboard panel for the unified TUI.
//!
//! Displays live memory statistics from a TrackingAllocator:
//! - Current / peak bytes with human-readable formatting
//! - Bytes sparkline (last 60 ticks)
//! - Allocation rate sparkline (last 60 ticks)
//! - Leak trend detection (consecutive ticks where active_allocations increased)
//!
//! When no tracker is connected, shows a placeholder message.

const std = @import("std");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const Panel = @import("../panel.zig");

const tracking = @import("abi").services.shared.utils.memory.tracking;
pub const TrackingStats = tracking.TrackingStats;
pub const TrackingAllocator = tracking.TrackingAllocator;

/// WDBX core allocator stats (optional; connect via connectWdbxTracker).
const wdbx_alloc = @import("abi").wdbx.core.alloc;

/// Fixed-size ring buffer for sparkline history.
fn RingBuffer(comptime capacity: usize) type {
    return struct {
        data: [capacity]u64 = [_]u64{0} ** capacity,
        write_idx: usize = 0,
        count: usize = 0,

        const Self = @This();

        pub fn push(self: *Self, value: u64) void {
            self.data[self.write_idx] = value;
            self.write_idx = (self.write_idx + 1) % capacity;
            if (self.count < capacity) self.count += 1;
        }

        pub fn len(self: *const Self) usize {
            return self.count;
        }

        /// Get values in insertion order (oldest first).
        pub fn orderedSlice(self: *const Self, out: *[capacity]u64) []const u64 {
            if (self.count < capacity) {
                @memcpy(out[0..self.count], self.data[0..self.count]);
                return out[0..self.count];
            }
            // Wrap around: read_start = write_idx (oldest)
            const tail_len = capacity - self.write_idx;
            @memcpy(out[0..tail_len], self.data[self.write_idx..capacity]);
            @memcpy(out[tail_len..capacity], self.data[0..self.write_idx]);
            return out[0..capacity];
        }
    };
}

pub const MemoryPanel = struct {
    allocator: std.mem.Allocator,
    tracker: ?*TrackingAllocator,
    wdbx_tracker: ?*wdbx_alloc.TrackingAllocator = null,

    bytes_history: RingBuffer(60),
    alloc_rate_history: RingBuffer(60),
    leak_trend_count: u32,

    tick_count: u64,
    prev_total_allocations: u64,
    prev_active_allocations: u64,
    last_stats: TrackingStats,

    pub fn init(allocator: std.mem.Allocator) MemoryPanel {
        return .{
            .allocator = allocator,
            .tracker = null,
            .bytes_history = .{},
            .alloc_rate_history = .{},
            .leak_trend_count = 0,
            .tick_count = 0,
            .prev_total_allocations = 0,
            .prev_active_allocations = 0,
            .last_stats = std.mem.zeroes(TrackingStats),
        };
    }

    /// Connect to a TrackingAllocator for live stats.
    pub fn connectTracker(self: *MemoryPanel, tracker: *TrackingAllocator) void {
        self.tracker = tracker;
    }

    /// Connect to WDBX engine allocator for live stats (e.g. when using abi.wdbx with tracking).
    pub fn connectWdbxTracker(self: *MemoryPanel, tracker: *wdbx_alloc.TrackingAllocator) void {
        self.wdbx_tracker = tracker;
    }

    pub fn render(self: *MemoryPanel, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        var y = rect.y;
        const col: u16 = rect.x + 2;
        var buf: [128]u8 = undefined;

        // Title
        try term.moveTo(y, rect.x);
        try term.write(theme.bold);
        try term.write(theme.primary);
        try term.write(" Memory");
        try term.write(theme.reset);
        y += 2;

        if (self.tracker == null and self.wdbx_tracker == null) {
            try term.moveTo(y, col);
            try term.write(theme.text_dim);
            try term.write("No tracker connected");
            try term.write(theme.reset);
            return;
        }

        const stats = self.last_stats;

        // Current / Peak bytes
        try term.moveTo(y, col);
        try term.write(theme.text);
        const cur_str = std.fmt.bufPrint(&buf, "Current: {s}  Peak: {s}", .{
            formatBytes(stats.current_bytes),
            formatBytes(stats.peak_bytes),
        }) catch "";
        try term.write(cur_str);
        try term.write(theme.reset);
        y += 1;

        // Active allocations
        try term.moveTo(y, col);
        try term.write(theme.text);
        const alloc_str = std.fmt.bufPrint(&buf, "Active: {d}  Total: {d}  Freed: {d}", .{
            stats.active_allocations,
            stats.total_allocations,
            stats.total_frees,
        }) catch "";
        try term.write(alloc_str);
        try term.write(theme.reset);
        y += 2;

        // Bytes sparkline
        try term.moveTo(y, col);
        try term.write(theme.bold);
        try term.write(theme.accent);
        try term.write("Bytes Usage");
        try term.write(theme.reset);
        y += 1;

        try self.renderSparkline(term, y, col, &self.bytes_history, theme);
        y += 2;

        // Alloc rate sparkline
        try term.moveTo(y, col);
        try term.write(theme.bold);
        try term.write(theme.accent);
        try term.write("Alloc Rate");
        try term.write(theme.reset);
        y += 1;

        try self.renderSparkline(term, y, col, &self.alloc_rate_history, theme);
        y += 2;

        // Leak warning
        if (self.leak_trend_count > 5) {
            try term.moveTo(y, col);
            try term.write(theme.@"error");
            try term.write(theme.bold);
            const leak_str = std.fmt.bufPrint(&buf, "LEAK WARNING: {d} consecutive growth ticks", .{
                self.leak_trend_count,
            }) catch "LEAK WARNING";
            try term.write(leak_str);
            try term.write(theme.reset);
        } else if (stats.active_allocations == 0) {
            try term.moveTo(y, col);
            try term.write(theme.success);
            try term.write("No leaks detected");
            try term.write(theme.reset);
        }
    }

    fn renderSparkline(
        _: *const MemoryPanel,
        term: *terminal.Terminal,
        y: u16,
        col: u16,
        ring: anytype,
        theme: *const themes.Theme,
    ) !void {
        const bars = [_][]const u8{ "\u{2581}", "\u{2582}", "\u{2583}", "\u{2584}", "\u{2585}", "\u{2586}", "\u{2587}", "\u{2588}" };

        try term.moveTo(y, col);
        try term.write("  ");

        var ordered: [60]u64 = undefined;
        const values = ring.orderedSlice(&ordered);

        if (values.len == 0) {
            try term.write(theme.text_dim);
            try term.write("(no data)");
            try term.write(theme.reset);
            return;
        }

        // Find max for scaling
        var max_val: u64 = 1;
        for (values) |v| {
            if (v > max_val) max_val = v;
        }

        // Show up to 40 bars
        const show = @min(values.len, 40);
        const start = values.len - show;
        try term.write(theme.info);
        for (values[start..]) |v| {
            const scaled = (v * 7) / max_val;
            const idx = @min(7, scaled);
            try term.write(bars[idx]);
        }
        try term.write(theme.reset);
    }

    pub fn tick(self: *MemoryPanel) anyerror!void {
        self.tick_count += 1;

        if (self.tracker) |tracker| {
            const stats = tracker.getStats();
            self.updateStatsFrom(stats);
        } else if (self.wdbx_tracker) |tracker| {
            const s = tracker.getStats();
            const stats: TrackingStats = .{
                .total_allocations = s.total_allocations,
                .total_frees = s.total_frees,
                .active_allocations = s.active_allocations,
                .current_bytes = s.current_bytes,
                .peak_bytes = s.peak_bytes,
                .total_bytes_allocated = s.total_bytes_allocated,
                .total_bytes_freed = s.total_bytes_freed,
                .failed_allocations = s.failed_allocations,
                .double_frees = s.double_frees,
                .invalid_frees = s.invalid_frees,
            };
            self.updateStatsFrom(stats);
        }
    }

    fn updateStatsFrom(self: *MemoryPanel, stats: TrackingStats) void {
        self.bytes_history.push(stats.current_bytes);
        const rate = stats.total_allocations -| self.prev_total_allocations;
        self.alloc_rate_history.push(rate);
        self.prev_total_allocations = stats.total_allocations;
        if (stats.active_allocations > self.prev_active_allocations) {
            self.leak_trend_count += 1;
        } else {
            self.leak_trend_count = 0;
        }
        self.prev_active_allocations = stats.active_allocations;
        self.last_stats = stats;
    }

    pub fn handleEvent(_: *MemoryPanel, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *MemoryPanel) []const u8 {
        return "Memory";
    }

    pub fn shortcutHint(_: *MemoryPanel) []const u8 {
        return "F9";
    }

    pub fn deinit(_: *MemoryPanel) void {}

    pub fn asPanel(self: *MemoryPanel) Panel {
        return Panel.from(MemoryPanel, self);
    }
};

fn formatBytes(bytes: u64) std.fmt.Formatter(formatBytesFn) {
    return .{ .data = bytes };
}

fn formatBytesFn(
    bytes: u64,
    comptime _: []const u8,
    _: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    if (bytes >= 1024 * 1024 * 1024) {
        try writer.print("{d:.1} GB", .{@as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0)});
    } else if (bytes >= 1024 * 1024) {
        try writer.print("{d:.1} MB", .{@as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0)});
    } else if (bytes >= 1024) {
        try writer.print("{d:.1} KB", .{@as(f64, @floatFromInt(bytes)) / 1024.0});
    } else {
        try writer.print("{d} B", .{bytes});
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "null tracker: tick and render don't crash" {
    var panel = MemoryPanel.init(std.testing.allocator);
    try panel.tick();
    try panel.tick();

    // Verify name/shortcut
    try std.testing.expectEqualStrings("Memory", panel.name());
    try std.testing.expectEqualStrings("F9", panel.shortcutHint());
}

test "real tracker: allocate memory, tick, verify history" {
    var tracker = TrackingAllocator.init(std.testing.allocator, .{});
    defer tracker.deinit();

    var panel = MemoryPanel.init(std.testing.allocator);
    panel.connectTracker(&tracker);

    const talloc = tracker.allocator();

    // Allocate some memory
    const ptr = try talloc.alloc(u8, 256);
    defer talloc.free(ptr);

    // Tick several times
    for (0..5) |_| {
        try panel.tick();
    }

    try std.testing.expectEqual(@as(usize, 5), panel.bytes_history.len());
    try std.testing.expect(panel.last_stats.current_bytes >= 256);
}

test "leak detection: allocate without freeing, track trend" {
    var tracker = TrackingAllocator.init(std.testing.allocator, .{});
    defer {
        // Clean up intentional leak
        var iter = tracker.allocations.iterator();
        while (iter.next()) |entry| {
            const ptr: [*]u8 = @ptrFromInt(entry.key_ptr.*);
            std.testing.allocator.free(ptr[0..entry.value_ptr.size]);
        }
        tracker.deinit();
    }

    var panel = MemoryPanel.init(std.testing.allocator);
    panel.connectTracker(&tracker);

    const talloc = tracker.allocator();

    // Allocate without freeing each tick — active_allocations keeps growing
    for (0..10) |_| {
        _ = try talloc.alloc(u8, 64);
        try panel.tick();
    }

    // Leak trend should be > 0
    try std.testing.expect(panel.leak_trend_count > 0);
}

test "panel vtable: create via Panel.from" {
    var panel = MemoryPanel.init(std.testing.allocator);
    const p = Panel.from(MemoryPanel, &panel);

    try std.testing.expectEqualStrings("Memory", p.getName());
    try std.testing.expectEqualStrings("F9", p.shortcutHint());
    try p.tick();
    const consumed = try p.handleEvent(.{ .key = .{ .code = .escape } });
    try std.testing.expect(!consumed);
}

test "ring buffer wraps correctly" {
    var ring: RingBuffer(4) = .{};
    ring.push(10);
    ring.push(20);
    ring.push(30);
    ring.push(40);
    ring.push(50); // overwrites 10

    try std.testing.expectEqual(@as(usize, 4), ring.len());

    var out: [4]u64 = undefined;
    const values = ring.orderedSlice(&out);
    try std.testing.expectEqual(@as(usize, 4), values.len);
    try std.testing.expectEqual(@as(u64, 20), values[0]);
    try std.testing.expectEqual(@as(u64, 50), values[3]);
}

test {
    std.testing.refAllDecls(@This());
}
