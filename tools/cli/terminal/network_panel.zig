//! Network Status Panel
//!
//! TUI panel for monitoring network status:
//! - Endpoint health checks
//! - Latency sparklines per endpoint
//! - Connection pool status
//! - Bandwidth utilization

const std = @import("std");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const render_utils = @import("render_utils.zig");
const widgets = @import("widgets.zig");
const RingBuffer = @import("ring_buffer.zig").RingBuffer;

pub const NetworkPanel = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,

    // Simulated endpoint metrics
    endpoints: [4]EndpointStatus,
    latency_history: RingBuffer(u16, 60),
    bandwidth_in: RingBuffer(u16, 60),
    bandwidth_out: RingBuffer(u16, 60),

    // Aggregate state
    total_requests: u64,
    total_errors: u32,
    active_sockets: u16,
    update_counter: u64,

    const Self = @This();

    pub const EndpointStatus = struct {
        name: []const u8,
        url: []const u8,
        status: enum { up, down, degraded },
        latency_ms: u16,
        requests_per_sec: u16,
    };

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) Self {
        return .{
            .allocator = allocator,
            .term = term,
            .theme = theme,
            .endpoints = .{
                .{ .name = "API Gateway", .url = "/api/v1", .status = .up, .latency_ms = 12, .requests_per_sec = 450 },
                .{ .name = "Auth Service", .url = "/auth", .status = .up, .latency_ms = 8, .requests_per_sec = 120 },
                .{ .name = "LLM Proxy", .url = "/llm", .status = .degraded, .latency_ms = 230, .requests_per_sec = 45 },
                .{ .name = "Storage", .url = "/storage", .status = .up, .latency_ms = 15, .requests_per_sec = 80 },
            },
            .latency_history = RingBuffer(u16, 60).init(),
            .bandwidth_in = RingBuffer(u16, 60).init(),
            .bandwidth_out = RingBuffer(u16, 60).init(),
            .total_requests = 0,
            .total_errors = 0,
            .active_sockets = 0,
            .update_counter = 0,
        };
    }

    pub fn deinit(_: *Self) void {}

    pub fn update(self: *Self) !void {
        self.update_counter += 1;
        const t = self.update_counter;

        // Vary endpoint metrics
        self.endpoints[0].latency_ms = 8 + @as(u16, @intCast((t * 3) % 15));
        self.endpoints[1].latency_ms = 5 + @as(u16, @intCast((t * 7) % 10));
        self.endpoints[2].latency_ms = 150 + @as(u16, @intCast((t * 11) % 200));
        self.endpoints[3].latency_ms = 10 + @as(u16, @intCast((t * 5) % 12));

        // Toggle degraded status occasionally
        self.endpoints[2].status = if ((t % 30) < 5) .down else if ((t % 30) < 10) .degraded else .up;

        const avg: u16 = (self.endpoints[0].latency_ms + self.endpoints[1].latency_ms +
            self.endpoints[2].latency_ms + self.endpoints[3].latency_ms) / 4;
        self.latency_history.push(avg);

        self.bandwidth_in.push(80 + @as(u16, @intCast((t * 17) % 120)));
        self.bandwidth_out.push(40 + @as(u16, @intCast((t * 13) % 60)));

        self.active_sockets = 24 + @as(u16, @intCast((t * 3) % 16));
        self.total_requests += 450 + (t % 100);
    }

    pub fn render(self: *Self, row: u16, col: u16, width: u16, height: u16) !void {
        const theme_val = self.theme;
        const term = self.term;

        if (height < 6 or width < 30) return;

        // Title
        try term.moveTo(row, col);
        try term.write(theme_val.bold);
        try term.write(theme_val.primary);
        try term.write(" NETWORK ");
        try term.write(theme_val.reset);

        // Border
        try render_utils.drawBox(term, .{ .x = col, .y = row + 1, .width = width, .height = height -| 1 }, .single, theme_val);

        // Endpoint table header
        try term.moveTo(row + 2, col + 2);
        try term.write(theme_val.bold);
        try term.write(theme_val.text);
        try term.write("Endpoint         Status   Latency   RPS");
        try term.write(theme_val.reset);

        // Endpoint rows
        var buf: [64]u8 = undefined;
        for (self.endpoints, 0..) |ep, i| {
            const r = row + 3 + @as(u16, @intCast(i));
            if (r >= row + height -| 4) break;

            try term.moveTo(r, col + 2);

            // Name (padded to 17 chars)
            try term.write(theme_val.text);
            try render_utils.writePadded(term, ep.name, 17);

            // Status
            const status_color = switch (ep.status) {
                .up => theme_val.success,
                .down => theme_val.@"error",
                .degraded => theme_val.warning,
            };
            try term.write(status_color);
            const status_str = switch (ep.status) {
                .up => "UP      ",
                .down => "DOWN    ",
                .degraded => "DEGRADE ",
            };
            try term.write(status_str);
            try term.write(theme_val.reset);

            // Latency
            try term.write(theme_val.text);
            const lat = std.fmt.bufPrint(&buf, "{d:>4} ms   ", .{ep.latency_ms}) catch "?    ";
            try term.write(lat);

            // RPS
            const rps = std.fmt.bufPrint(&buf, "{d}", .{ep.requests_per_sec}) catch "?";
            try term.write(rps);
            try term.write(theme_val.reset);
        }

        // Aggregate stats
        if (height > 10) {
            try term.moveTo(row + 8, col + 2);
            try term.write(theme_val.text_dim);
            const agg = std.fmt.bufPrint(&buf, "Sockets: {d}  Total: {d}  Errors: {d}", .{
                self.active_sockets,
                self.total_requests,
                self.total_errors,
            }) catch "?";
            try term.write(agg);
            try term.write(theme_val.reset);
        }

        // Latency sparkline
        if (height > 12 and width > 36) {
            try term.moveTo(row + 10, col + 2);
            try term.write(theme_val.text_dim);
            try term.write("Avg latency (60s):");
            try term.write(theme_val.reset);
            try term.moveTo(row + 11, col + 2);
            try self.renderSparkline(&self.latency_history, @min(width -| 6, 50));
        }
    }

    /// Render a sparkline from a u16 RingBuffer by normalizing to 0-100 u8 range.
    fn renderSparkline(self: *Self, ring: *const RingBuffer(u16, 60), max_width: u16) !void {
        if (ring.count == 0) return;

        var raw_buf: [60]u16 = undefined;
        const raw_values = ring.toSlice(&raw_buf);

        const max_val: u16 = ring.max() orelse 1;
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
