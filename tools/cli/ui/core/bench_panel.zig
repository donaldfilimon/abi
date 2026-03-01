//! Benchmark Results Panel
//!
//! TUI panel for displaying benchmark results:
//! - Suite summary (pass/fail/skip counts)
//! - Individual benchmark timing table
//! - Throughput comparison bars
//! - History sparklines for tracked metrics

const std = @import("std");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const render_utils = @import("render_utils.zig");
const widgets = @import("widgets.zig");
const RingBuffer = @import("ring_buffer.zig").RingBuffer;

pub const BenchmarkPanel = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,

    // Benchmark results
    suites: [5]SuiteResult,
    selected_suite: usize,
    throughput_history: RingBuffer(u16, 60),
    update_counter: u64,

    const Self = @This();

    pub const SuiteResult = struct {
        name: []const u8,
        passed: u16,
        failed: u16,
        skipped: u16,
        avg_ns: u64,
        throughput_ops: u32,
        status: enum { running, complete, idle },
    };

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) Self {
        return .{
            .allocator = allocator,
            .term = term,
            .theme = theme,
            .suites = .{
                .{ .name = "Hash (SHA-256)", .passed = 48, .failed = 0, .skipped = 1, .avg_ns = 245, .throughput_ops = 4_080_000, .status = .complete },
                .{ .name = "SIMD Vector", .passed = 32, .failed = 0, .skipped = 0, .avg_ns = 12, .throughput_ops = 83_000_000, .status = .complete },
                .{ .name = "JSON Parse", .passed = 24, .failed = 1, .skipped = 0, .avg_ns = 1_850, .throughput_ops = 540_000, .status = .complete },
                .{ .name = "Allocator", .passed = 16, .failed = 0, .skipped = 2, .avg_ns = 38, .throughput_ops = 26_000_000, .status = .idle },
                .{ .name = "Network I/O", .passed = 8, .failed = 0, .skipped = 4, .avg_ns = 125_000, .throughput_ops = 8_000, .status = .idle },
            },
            .selected_suite = 0,
            .throughput_history = RingBuffer(u16, 60).init(),
            .update_counter = 0,
        };
    }

    pub fn deinit(_: *Self) void {}

    pub fn update(self: *Self) !void {
        self.update_counter += 1;

        // Record selected suite throughput (clamped to u16 range as K ops/s)
        self.throughput_history.push(@intCast(@min(65535, self.suites[self.selected_suite].throughput_ops / 1000)));
    }

    pub fn moveUp(self: *Self) void {
        if (self.selected_suite > 0) self.selected_suite -= 1;
    }

    pub fn moveDown(self: *Self) void {
        if (self.selected_suite < self.suites.len - 1) self.selected_suite += 1;
    }

    pub fn render(self: *Self, row: u16, col: u16, width: u16, height: u16) !void {
        const theme_val = self.theme;
        const term = self.term;

        if (height < 6 or width < 30) return;

        // Title
        try term.moveTo(row, col);
        try term.write(theme_val.bold);
        try term.write(theme_val.primary);
        try term.write(" BENCHMARKS ");
        try term.write(theme_val.reset);

        // Border
        try render_utils.drawBox(term, .{ .x = col, .y = row + 1, .width = width, .height = height -| 1 }, .single, theme_val);

        // Suite table header
        try term.moveTo(row + 2, col + 2);
        try term.write(theme_val.bold);
        try term.write(theme_val.text);
        try term.write("Suite            Pass  Fail  Skip  Avg ns     Ops/s");
        try term.write(theme_val.reset);

        // Suite rows
        var buf: [64]u8 = undefined;
        for (self.suites, 0..) |suite, i| {
            const r = row + 3 + @as(u16, @intCast(i));
            if (r >= row + height -| 5) break;

            try term.moveTo(r, col + 2);

            // Selection indicator
            if (i == self.selected_suite) {
                try term.write(theme_val.accent);
                try term.write("\xe2\x96\xb6 ");
            } else {
                try term.write("  ");
            }

            // Name
            try term.write(theme_val.text);
            try render_utils.writePadded(term, suite.name, 15);

            // Passed
            try term.write(theme_val.success);
            const p = std.fmt.bufPrint(&buf, "{d:>4}  ", .{suite.passed}) catch "?    ";
            try term.write(p);

            // Failed
            if (suite.failed > 0) {
                try term.write(theme_val.@"error");
            } else {
                try term.write(theme_val.text_dim);
            }
            const f = std.fmt.bufPrint(&buf, "{d:>4}  ", .{suite.failed}) catch "?    ";
            try term.write(f);

            // Skipped
            try term.write(theme_val.text_dim);
            const s = std.fmt.bufPrint(&buf, "{d:>4}  ", .{suite.skipped}) catch "?    ";
            try term.write(s);

            // Avg ns
            try term.write(theme_val.text);
            const ns = std.fmt.bufPrint(&buf, "{d:>8}  ", .{suite.avg_ns}) catch "?        ";
            try term.write(ns);

            // Throughput
            try term.write(theme_val.accent);
            const ops = std.fmt.bufPrint(&buf, "{d:>10}", .{suite.throughput_ops}) catch "?";
            try term.write(ops);
            try term.write(theme_val.reset);
        }

        // Status indicator for selected suite
        if (height > 10) {
            const selected = self.suites[self.selected_suite];
            try term.moveTo(row + 9, col + 2);
            try term.write(theme_val.text_dim);
            const status_str = switch (selected.status) {
                .running => "RUNNING",
                .complete => "COMPLETE",
                .idle => "IDLE",
            };
            const detail = std.fmt.bufPrint(&buf, "Selected: {s} [{s}]", .{ selected.name, status_str }) catch "?";
            try term.write(detail);
            try term.write(theme_val.reset);
        }

        // Throughput sparkline
        if (height > 12 and width > 36) {
            try term.moveTo(row + 11, col + 2);
            try term.write(theme_val.text_dim);
            try term.write("Throughput (K ops/s):");
            try term.write(theme_val.reset);
            try term.moveTo(row + 12, col + 2);
            try self.renderSparkline(@min(width -| 6, 50));
        }
    }

    /// Render throughput sparkline by normalizing u16 values to 0-100 u8 range.
    fn renderSparkline(self: *Self, max_width: u16) !void {
        if (self.throughput_history.count == 0) return;

        var raw_buf: [60]u16 = undefined;
        const raw_values = self.throughput_history.toSlice(&raw_buf);

        const max_val: u16 = self.throughput_history.max() orelse 1;
        var norm_buf: [60]u8 = undefined;
        const count = @min(raw_values.len, @as(usize, max_width));
        for (raw_values[0..count], 0..) |v, idx| {
            norm_buf[idx] = @intCast(@min(100, (@as(u32, v) * 100) / @max(1, @as(u32, max_val))));
        }

        var spark_out: [200]u8 = undefined;
        const sparkline = widgets.SparklineChart.render(norm_buf[0..count], &spark_out);
        try self.term.write(self.theme.info);
        try self.term.write(sparkline);
        try self.term.write(self.theme.reset);
    }
};

test {
    std.testing.refAllDecls(@This());
}
