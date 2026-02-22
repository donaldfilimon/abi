//! Database Monitoring Panel
//!
//! TUI panel for monitoring database status:
//! - Connection pool utilization
//! - Query throughput (reads/writes per second)
//! - Storage usage and table count
//! - Recent query log with latency

const std = @import("std");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const render_utils = @import("render_utils.zig");
const widgets = @import("widgets.zig");
const RingBuffer = @import("ring_buffer.zig").RingBuffer;

pub const DatabasePanel = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,

    // Simulated metrics
    read_qps: RingBuffer(u16, 60),
    write_qps: RingBuffer(u16, 60),
    connection_pool: RingBuffer(u8, 60),

    // Current state
    total_connections: u16,
    active_connections: u16,
    max_connections: u16,
    total_queries: u64,
    total_tables: u16,
    storage_used_mb: u32,
    storage_total_mb: u32,
    avg_latency_us: u32,
    update_counter: u64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) Self {
        return .{
            .allocator = allocator,
            .term = term,
            .theme = theme,
            .read_qps = RingBuffer(u16, 60).init(),
            .write_qps = RingBuffer(u16, 60).init(),
            .connection_pool = RingBuffer(u8, 60).init(),
            .total_connections = 0,
            .active_connections = 0,
            .max_connections = 64,
            .total_queries = 0,
            .total_tables = 12,
            .storage_used_mb = 256,
            .storage_total_mb = 1024,
            .avg_latency_us = 850,
            .update_counter = 0,
        };
    }

    pub fn deinit(_: *Self) void {}

    pub fn update(self: *Self) !void {
        self.update_counter += 1;
        const t = self.update_counter;

        // Simulate read/write QPS with some variation
        const base_reads: u16 = 120 + @as(u16, @intCast((t * 7) % 80));
        const base_writes: u16 = 30 + @as(u16, @intCast((t * 3) % 25));
        self.read_qps.push(base_reads);
        self.write_qps.push(base_writes);

        // Simulate connection pool
        self.active_connections = 8 + @as(u16, @intCast((t * 5) % 20));
        self.total_connections = self.active_connections + 4;
        const pool_pct: u8 = @intCast(@min(100, (@as(u32, self.active_connections) * 100) / self.max_connections));
        self.connection_pool.push(pool_pct);

        self.total_queries += @as(u64, base_reads) + @as(u64, base_writes);
        self.avg_latency_us = 600 + @as(u32, @intCast((t * 13) % 500));
    }

    pub fn render(self: *Self, row: u16, col: u16, width: u16, height: u16) !void {
        const theme_val = self.theme;
        const term = self.term;

        if (height < 6 or width < 30) return;

        // Title
        try term.moveTo(row, col);
        try term.write(theme_val.bold);
        try term.write(theme_val.primary);
        try term.write(" DATABASE ");
        try term.write(theme_val.reset);

        // Border
        try render_utils.drawBox(term, .{ .x = col, .y = row + 1, .width = width, .height = height -| 1 }, .single, theme_val);

        // Connection pool gauge
        const pool_pct: u8 = if (self.max_connections > 0)
            @intCast(@min(100, (@as(u32, self.active_connections) * 100) / self.max_connections))
        else
            0;
        try term.moveTo(row + 2, col + 2);
        try term.write(theme_val.text);
        try term.write("Connections: ");
        var buf: [64]u8 = undefined;
        const conn_str = std.fmt.bufPrint(&buf, "{d}/{d}", .{ self.active_connections, self.max_connections }) catch "?/?";
        try term.write(conn_str);
        if (width > 40) {
            try term.moveTo(row + 2, col + 28);
            var gauge_buf: [128]u8 = undefined;
            const gauge_text = widgets.ProgressGauge.render(pool_pct, @min(20, width -| 32), &gauge_buf);
            try term.write(gauge_text);
        }

        // Query throughput
        try term.moveTo(row + 3, col + 2);
        try term.write(theme_val.text);
        try term.write("Read QPS:  ");
        const rqps = self.read_qps.latest() orelse 0;
        const rqps_str = std.fmt.bufPrint(&buf, "{d}", .{rqps}) catch "?";
        try term.write(theme_val.success);
        try term.write(rqps_str);
        try term.write(theme_val.reset);

        try term.moveTo(row + 4, col + 2);
        try term.write(theme_val.text);
        try term.write("Write QPS: ");
        const wqps = self.write_qps.latest() orelse 0;
        const wqps_str = std.fmt.bufPrint(&buf, "{d}", .{wqps}) catch "?";
        try term.write(theme_val.accent);
        try term.write(wqps_str);
        try term.write(theme_val.reset);

        // Storage
        if (height > 8) {
            try term.moveTo(row + 6, col + 2);
            try term.write(theme_val.text);
            try term.write("Storage: ");
            const storage_str = std.fmt.bufPrint(&buf, "{d}/{d} MB", .{ self.storage_used_mb, self.storage_total_mb }) catch "?";
            try term.write(storage_str);

            const storage_pct: u8 = @intCast(@min(100, (@as(u32, self.storage_used_mb) * 100) / @max(1, self.storage_total_mb)));
            if (width > 40) {
                try term.moveTo(row + 6, col + 28);
                var gauge_buf2: [128]u8 = undefined;
                const gauge_text2 = widgets.ProgressGauge.render(storage_pct, @min(20, width -| 32), &gauge_buf2);
                try term.write(gauge_text2);
            }
        }

        // Latency
        if (height > 9) {
            try term.moveTo(row + 7, col + 2);
            try term.write(theme_val.text);
            try term.write("Avg Latency: ");
            const lat_str = std.fmt.bufPrint(&buf, "{d} us", .{self.avg_latency_us}) catch "?";
            try term.write(lat_str);
        }

        // Tables
        if (height > 10) {
            try term.moveTo(row + 8, col + 2);
            try term.write(theme_val.text_dim);
            const tables_str = std.fmt.bufPrint(&buf, "Tables: {d}  Queries: {d}", .{ self.total_tables, self.total_queries }) catch "?";
            try term.write(tables_str);
            try term.write(theme_val.reset);
        }

        // Sparkline for read QPS
        if (height > 12 and width > 36) {
            try term.moveTo(row + 10, col + 2);
            try term.write(theme_val.text_dim);
            try term.write("Read QPS (60s):");
            try term.write(theme_val.reset);
            try term.moveTo(row + 11, col + 2);
            try self.renderSparkline(&self.read_qps, @min(width -| 6, 50));
        }
    }

    /// Render a sparkline from a u16 RingBuffer by normalizing to 0-100 u8 range.
    fn renderSparkline(self: *Self, ring: *const RingBuffer(u16, 60), max_width: u16) !void {
        if (ring.count == 0) return;

        // Extract raw values
        var raw_buf: [60]u16 = undefined;
        const raw_values = ring.toSlice(&raw_buf);

        // Normalize to 0-100 u8 range
        const max_val: u16 = ring.max() orelse 1;
        var norm_buf: [60]u8 = undefined;
        const count = @min(raw_values.len, @as(usize, max_width));
        for (raw_values[0..count], 0..) |v, idx| {
            norm_buf[idx] = @intCast(@min(100, (@as(u32, v) * 100) / @max(1, @as(u32, max_val))));
        }

        // Render sparkline and write to terminal
        var spark_out: [200]u8 = undefined;
        const sparkline = widgets.SparklineChart.render(norm_buf[0..count], &spark_out);
        try self.term.write(self.theme.info);
        try self.term.write(sparkline);
        try self.term.write(self.theme.reset);
    }
};
