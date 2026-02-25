//! Streaming Inference Dashboard
//!
//! Real-time TUI dashboard for monitoring streaming inference:
//! - Server status and uptime
//! - Time to First Token (TTFT) metrics with percentiles
//! - Token throughput visualization
//! - Active connections and queue depth
//! - Recent request log with latency
//!
//! Part of the TUI/CLI enhancement suite.

const std = @import("std");
const abi = @import("abi");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const events = @import("events.zig");
const widgets = @import("widgets.zig");
const box = widgets.box;
const unicode = @import("unicode.zig");
const render_utils = @import("render_utils.zig");
const layout = @import("layout.zig");
const RingBuffer = @import("ring_buffer.zig").RingBuffer;
const PercentileTracker = @import("percentile_tracker.zig").PercentileTracker;

/// Streaming Inference Dashboard for monitoring real-time LLM inference
pub const StreamingDashboard = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,

    // Time-series metrics (2 minutes of history at 1 sample/sec)
    ttft_history: RingBuffer(u32, 120),
    throughput_history: RingBuffer(f32, 120),
    connection_history: RingBuffer(u16, 120),

    // Percentile tracking for TTFT
    ttft_percentiles: PercentileTracker,

    // Current server state
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
    recent_requests: std.ArrayListUnmanaged(RequestLogEntry),
    show_request_log: bool,
    request_scroll: usize,

    // Polling configuration
    last_poll: i64,
    poll_interval_ms: u64,

    const Self = @This();

    /// Server health status
    pub const ServerStatus = enum {
        online,
        offline,
        degraded,
        unknown,

        pub fn toIcon(self: ServerStatus) []const u8 {
            return switch (self) {
                .online => "●",
                .offline => "○",
                .degraded => "◐",
                .unknown => "?",
            };
        }

        pub fn toString(self: ServerStatus) []const u8 {
            return switch (self) {
                .online => "Online",
                .offline => "Offline",
                .degraded => "Degraded",
                .unknown => "Unknown",
            };
        }
    };

    /// Entry in the request log
    pub const RequestLogEntry = struct {
        timestamp: i64,
        method: [8]u8,
        method_len: u8,
        path: [64]u8,
        path_len: u8,
        status_code: u16,
        latency_ms: u32,
        token_count: u32,

        pub fn getMethod(self: *const RequestLogEntry) []const u8 {
            return self.method[0..self.method_len];
        }

        pub fn getPath(self: *const RequestLogEntry) []const u8 {
            return self.path[0..self.path_len];
        }

        pub fn init(method: []const u8, path: []const u8, status: u16, latency: u32, tokens: u32) RequestLogEntry {
            var entry = RequestLogEntry{
                .timestamp = abi.shared.utils.unixMs(),
                .method = undefined,
                .method_len = @intCast(@min(method.len, 8)),
                .path = undefined,
                .path_len = @intCast(@min(path.len, 64)),
                .status_code = status,
                .latency_ms = latency,
                .token_count = tokens,
            };
            @memset(&entry.method, 0);
            @memset(&entry.path, 0);
            @memcpy(entry.method[0..entry.method_len], method[0..entry.method_len]);
            @memcpy(entry.path[0..entry.path_len], path[0..entry.path_len]);
            return entry;
        }
    };

    /// User actions from key input
    pub const Action = enum {
        quit,
        refresh,
        toggle_log,
        clear_stats,
        increase_poll,
        decrease_poll,
        scroll_up,
        scroll_down,
    };

    /// Initialize a new Streaming Dashboard
    pub fn init(
        allocator: std.mem.Allocator,
        term: *terminal.Terminal,
        theme: *const themes.Theme,
        endpoint: []const u8,
    ) !Self {
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
            .recent_requests = .empty,
            .show_request_log = true,
            .request_scroll = 0,
            .last_poll = 0,
            .poll_interval_ms = 500,
        };
    }

    inline fn moveTo(self: *Self, row: usize, col: usize) !void {
        try self.term.moveTo(@as(u16, @intCast(row)), @as(u16, @intCast(col)));
    }

    /// Clean up dashboard resources
    pub fn deinit(self: *Self) void {
        self.ttft_percentiles.deinit();
        self.recent_requests.deinit(self.allocator);
    }

    /// Poll metrics from the streaming server
    pub fn pollMetrics(self: *Self) !void {
        const now = abi.shared.utils.unixMs();

        // Check if enough time has passed
        if (now - self.last_poll < @as(i64, @intCast(self.poll_interval_ms))) {
            return;
        }
        self.last_poll = now;

        const has_scheme = std.mem.startsWith(u8, self.server_endpoint, "http://") or
            std.mem.startsWith(u8, self.server_endpoint, "https://");
        const base_url = if (has_scheme)
            self.server_endpoint
        else
            try std.fmt.allocPrint(self.allocator, "http://{s}", .{self.server_endpoint});
        defer if (!has_scheme) self.allocator.free(base_url);

        const health_url = try std.fmt.allocPrint(self.allocator, "{s}/health", .{base_url});
        defer self.allocator.free(health_url);
        const metrics_url = try std.fmt.allocPrint(self.allocator, "{s}/metrics", .{base_url});
        defer self.allocator.free(metrics_url);

        var client = try abi.web.HttpClient.init(self.allocator);
        defer client.deinit();

        const health_response = client.getWithOptions(health_url, .{ .max_response_bytes = 4096 }) catch {
            self.server_status = .offline;
            return;
        };
        defer client.freeResponse(health_response);

        if (health_response.status != 200) {
            self.server_status = .offline;
            return;
        }
        self.server_status = .online;

        const metrics_response = client.getWithOptions(metrics_url, .{ .max_response_bytes = 64 * 1024 }) catch {
            self.server_status = .degraded;
            return;
        };
        defer client.freeResponse(metrics_response);

        if (metrics_response.status != 200) {
            self.server_status = .degraded;
            return;
        }

        const MetricsResponse = struct {
            status: ?[]const u8 = null,
            uptime_ms: ?u64 = null,
            active_streams: ?u32 = null,
            max_streams: ?u32 = null,
            queue_depth: ?u32 = null,
            total_tokens: ?u64 = null,
            total_requests: ?u64 = null,
            total_errors: ?u64 = null,
            ttft_ms_p50: ?u32 = null,
            ttft_ms_p95: ?u32 = null,
            ttft_ms_p99: ?u32 = null,
            throughput_tps: ?f64 = null,
        };

        const parsed = std.json.parseFromSlice(MetricsResponse, self.allocator, metrics_response.body, .{}) catch {
            self.server_status = .degraded;
            return;
        };
        defer parsed.deinit();

        const metrics = parsed.value;
        if (metrics.status) |status| {
            if (!std.mem.eql(u8, status, "ok")) {
                self.server_status = .degraded;
            }
        }

        if (metrics.uptime_ms) |uptime| {
            self.uptime_ms = @intCast(uptime);
        }
        if (metrics.active_streams) |active| {
            const clamped: u16 = if (active > std.math.maxInt(u16))
                std.math.maxInt(u16)
            else
                @intCast(active);
            self.recordConnections(clamped);
        }
        if (metrics.max_streams) |max_conn| {
            self.max_connections = max_conn;
        }
        if (metrics.queue_depth) |depth| {
            self.queue_depth = depth;
        }
        if (metrics.total_tokens) |total| {
            self.total_tokens = total;
        }
        if (metrics.total_requests) |total| {
            self.total_requests = total;
        }
        if (metrics.total_errors) |errors| {
            self.error_count = @intCast(errors);
        }
        if (metrics.ttft_ms_p50) |p50| {
            self.recordTtft(p50);
        }
        if (metrics.throughput_tps) |rate| {
            self.recordThroughput(@floatCast(rate));
        }
    }

    /// Alias for pollMetrics — satisfies the Dashboard panel contract.
    pub fn update(self: *Self) !void {
        return self.pollMetrics();
    }

    /// Record a new TTFT sample
    pub fn recordTtft(self: *Self, ttft_ms: u32) void {
        self.ttft_history.push(ttft_ms);
        self.ttft_percentiles.add(ttft_ms);
    }

    /// Record throughput sample
    pub fn recordThroughput(self: *Self, tokens_per_sec: f32) void {
        self.throughput_history.push(tokens_per_sec);
    }

    /// Record connection count
    pub fn recordConnections(self: *Self, count: u16) void {
        self.connection_history.push(count);
        self.active_connections = count;
    }

    /// Add a request to the log
    pub fn logRequest(self: *Self, method: []const u8, path: []const u8, status: u16, latency: u32, tokens: u32) !void {
        const entry = RequestLogEntry.init(method, path, status, latency, tokens);
        try self.recent_requests.append(self.allocator, entry);

        // Keep only last 50 requests
        if (self.recent_requests.items.len > 50) {
            _ = self.recent_requests.orderedRemove(0);
        }

        self.total_requests += 1;
        self.total_tokens += tokens;

        if (status >= 400) {
            self.error_count += 1;
        }
    }

    /// Update server status
    pub fn setServerStatus(self: *Self, status: ServerStatus) void {
        self.server_status = status;
    }

    /// Clear all statistics
    pub fn clearStats(self: *Self) void {
        self.ttft_history = RingBuffer(u32, 120).init();
        self.throughput_history = RingBuffer(f32, 120).init();
        self.connection_history = RingBuffer(u16, 120).init();
        self.ttft_percentiles.clear();
        self.recent_requests.clearRetainingCapacity();
        self.total_tokens = 0;
        self.total_requests = 0;
        self.error_count = 0;
        self.request_scroll = 0;
    }

    /// Render the dashboard
    pub fn render(
        self: *Self,
        start_row: usize,
        start_col: usize,
        width: usize,
        height: usize,
    ) !void {
        if (height < 4 or width < 30) return; // Minimum dimensions required

        // Dynamic layout: header=2, metrics=6, connections=3, footer=1
        // Request log gets whatever remains.
        const header_h: usize = 2;
        const footer_h: usize = 1;
        const metrics_h: usize = 6;
        const connections_h: usize = 3;
        const fixed_h = header_h + footer_h;

        try self.renderHeader(start_row, start_col, width);

        var current_row = start_row + header_h;
        const available = height -| fixed_h;

        if (available >= metrics_h) {
            try self.renderMetricsPanel(current_row, start_col, width);
            current_row += metrics_h;
        }

        if (available >= metrics_h + connections_h) {
            try self.renderConnectionsPanel(current_row, start_col, width);
            current_row += connections_h;
        }

        const log_space = (start_row + height) -| current_row -| footer_h;
        if (self.show_request_log and log_space > 2) {
            try self.renderRequestLog(current_row, start_col, width, log_space);
        }

        try self.renderFooter(start_row + height - 1, start_col, width);
    }

    fn renderHeader(self: *Self, row: usize, col: usize, width: usize) !void {
        // Top border with title
        try self.moveTo(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.tl);
        try self.term.write(box.h);
        try self.term.write(" ");
        try self.term.write(self.theme.accent);
        try self.term.write("Streaming Inference Dashboard");
        try self.term.write(self.theme.reset);
        try self.term.write(" ");

        // Fill with border
        const title = "Streaming Inference Dashboard";
        const title_width = unicode.displayWidth(title) + 2; // +2 for surrounding spaces
        try self.term.write(self.theme.border);
        const fill_count = if (width > title_width + 3) width - title_width - 3 else 0;
        try render_utils.writeRepeat(self.term, box.h, fill_count);
        try self.term.write(box.tr);
        try self.term.write(self.theme.reset);

        // Status line
        try self.moveTo(row + 1, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);

        // Server endpoint
        try self.term.write(" Server: ");
        try self.term.write(self.theme.accent);
        const endpoint_display = unicode.truncateToWidth(self.server_endpoint, 30);
        try self.term.write(endpoint_display);
        try self.term.write(self.theme.reset);
        try self.term.write("  ");

        // Status indicator
        const status_color = switch (self.server_status) {
            .online => self.theme.success,
            .offline => self.theme.@"error",
            .degraded => self.theme.warning,
            .unknown => self.theme.text_dim,
        };
        try self.term.write(status_color);
        try self.term.write(self.server_status.toIcon());
        try self.term.write(" ");
        try self.term.write(self.server_status.toString());
        try self.term.write(self.theme.reset);

        // Uptime
        try self.term.write("  Uptime: ");
        const uptime_secs = @divFloor(self.uptime_ms, 1000);
        const uptime_mins = @divFloor(uptime_secs, 60);
        const uptime_hours = @divFloor(uptime_mins, 60);

        var uptime_buf: [16]u8 = undefined;
        const uptime_str = std.fmt.bufPrint(&uptime_buf, "{d}h {d}m", .{
            uptime_hours,
            @mod(uptime_mins, 60),
        }) catch "??";
        try self.term.write(uptime_str);

        // Right border
        try self.moveTo(row + 1, col + width - 1);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);
    }

    fn renderMetricsPanel(self: *Self, row: usize, col: usize, width: usize) !void {
        // Separator
        try self.moveTo(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.lsep);
        try render_utils.writeRepeat(self.term, box.h, if (width > 2) width - 2 else 0);
        try self.term.write(box.rsep);
        try self.term.write(self.theme.reset);

        // TTFT and Throughput boxes header — adapt to available width
        try self.moveTo(row + 1, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);
        try self.term.write(" ");
        try self.term.write(self.theme.text_dim);
        const metrics_header = "┌─ Time to First Token ─┐  ┌─ Token Throughput ──┐";
        const inner_w = if (width > 2) width - 2 else 0;
        const header_clipped = unicode.truncateToWidth(metrics_header, inner_w -| 1);
        try self.term.write(header_clipped);
        try self.term.write(self.theme.reset);

        // TTFT Current
        const latest_ttft = self.ttft_history.latest() orelse 0;
        const latest_throughput = self.throughput_history.latest() orelse 0;

        try self.moveTo(row + 2, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);

        var ttft_buf: [64]u8 = undefined;
        const ttft_line = std.fmt.bufPrint(&ttft_buf, " │ Current: {d:>6}ms    │  │ Rate: {d:>6.0} tok/s │", .{
            latest_ttft,
            latest_throughput,
        }) catch " │ Current: ??          │  │ Rate: ??           │";
        try self.term.write(unicode.truncateToWidth(ttft_line, inner_w -| 1));

        // TTFT P50
        const p50 = self.ttft_percentiles.getPercentile(50);
        const total_k = @divFloor(self.total_tokens, 1000);

        try self.moveTo(row + 3, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);

        var p50_buf: [64]u8 = undefined;
        const p50_line = std.fmt.bufPrint(&p50_buf, " │ P50:     {d:>6}ms    │  │ Total: {d:>6}K tok │", .{
            p50,
            total_k,
        }) catch " │ P50:     ??          │  │ Total: ??          │";
        try self.term.write(unicode.truncateToWidth(p50_line, inner_w -| 1));

        // TTFT P99
        const p99 = self.ttft_percentiles.getPercentile(99);

        try self.moveTo(row + 4, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);

        var p99_buf: [64]u8 = undefined;
        const p99_line = std.fmt.bufPrint(&p99_buf, " │ P99:     {d:>6}ms    │  │ Reqs:  {d:>6}      │", .{
            p99,
            self.total_requests,
        }) catch " │ P99:     ??          │  │ Reqs:  ??          │";
        try self.term.write(unicode.truncateToWidth(p99_line, inner_w -| 1));

        // Box bottoms
        try self.moveTo(row + 5, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);
        try self.term.write(self.theme.text_dim);
        const metrics_footer = " └───────────────────────┘  └─────────────────────┘";
        const footer_clipped = unicode.truncateToWidth(metrics_footer, inner_w -| 1);
        try self.term.write(footer_clipped);
        try self.term.write(self.theme.reset);

        // Right borders for all rows
        for (1..6) |r| {
            try self.moveTo(row + r, col + width - 1);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);
        }
    }

    fn renderConnectionsPanel(self: *Self, row: usize, col: usize, width: usize) !void {
        // Separator
        try self.moveTo(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.lsep);
        try render_utils.writeRepeat(self.term, box.h, if (width > 2) width - 2 else 0);
        try self.term.write(box.rsep);
        try self.term.write(self.theme.reset);

        // Connection stats
        try self.moveTo(row + 1, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);

        // Active connections with color coding
        try self.term.write(" Active: ");
        const conn_ratio = if (self.max_connections > 0)
            @as(f32, @floatFromInt(self.active_connections)) / @as(f32, @floatFromInt(self.max_connections))
        else
            0;
        if (conn_ratio > 0.9) {
            try self.term.write(self.theme.@"error");
        } else if (conn_ratio > 0.7) {
            try self.term.write(self.theme.warning);
        } else {
            try self.term.write(self.theme.success);
        }

        var conn_buf: [32]u8 = undefined;
        const conn_str = std.fmt.bufPrint(&conn_buf, "{d}/{d}", .{ self.active_connections, self.max_connections }) catch "??";
        try self.term.write(conn_str);
        try self.term.write(self.theme.reset);

        // Queue depth
        try self.term.write("   Queue: ");
        if (self.queue_depth > 10) {
            try self.term.write(self.theme.warning);
        }
        var queue_buf: [16]u8 = undefined;
        const queue_str = std.fmt.bufPrint(&queue_buf, "{d}", .{self.queue_depth}) catch "??";
        try self.term.write(queue_str);
        try self.term.write(self.theme.reset);

        // Error count
        try self.term.write("   Errors: ");
        if (self.error_count > 0) {
            try self.term.write(self.theme.@"error");
        }
        var err_buf: [16]u8 = undefined;
        const err_str = std.fmt.bufPrint(&err_buf, "{d}", .{self.error_count}) catch "??";
        try self.term.write(err_str);
        try self.term.write(self.theme.reset);

        // Connection progress bar
        try self.term.write("   ");
        const conn_pct: u8 = @intFromFloat(@min(conn_ratio * 100, 100));
        var bar_buf: [32]u8 = undefined;
        const bar = widgets.ProgressGauge.render(conn_pct, 15, &bar_buf);
        try self.term.write(bar);

        // Right border
        try self.moveTo(row + 1, col + width - 1);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);

        // Sparkline row
        try self.moveTo(row + 2, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);
        try self.term.write(self.theme.text_dim);
        try self.term.write(" Connections: ");
        try self.term.write(self.theme.reset);

        // Connection history sparkline
        if (self.connection_history.len() > 0) {
            const max_conn = self.connection_history.max() orelse 1;
            var iter = self.connection_history.iterator();
            var spark_count: usize = 0;
            const max_spark: usize = if (width > 40) 30 else 15;

            while (iter.next()) |conn| {
                if (spark_count >= max_spark) break;
                const normalized = if (max_conn > 0) @as(f32, @floatFromInt(conn)) / @as(f32, @floatFromInt(max_conn)) else 0;
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

        // Right border
        try self.moveTo(row + 2, col + width - 1);
        try self.term.write(self.theme.border);
        try self.term.write(box.v);
        try self.term.write(self.theme.reset);
    }

    fn renderRequestLog(
        self: *Self,
        row: usize,
        col: usize,
        width: usize,
        height: usize,
    ) !void {
        // Header
        try self.moveTo(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.lsep);
        try self.term.write(box.h);
        const req_title = " Recent Requests ";
        try self.term.write(req_title);
        const req_title_width = unicode.displayWidth(req_title) + 1; // +1 for leading box.h
        const req_fill = if (width > req_title_width + 1) width - req_title_width - 1 else 0;
        try render_utils.writeRepeat(self.term, box.h, req_fill);
        try self.term.write(box.rsep);
        try self.term.write(self.theme.reset);

        // Request entries
        const requests = self.recent_requests.items;
        const visible_count = @min(requests.len, height - 1);

        if (requests.len == 0) {
            try self.moveTo(row + 1, col);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);
            try self.term.write(self.theme.text_dim);
            try self.term.write("  No requests recorded yet");
            try self.term.write(self.theme.reset);

            try self.moveTo(row + 1, col + width - 1);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);
        } else {
            // Show most recent first (reverse order)
            var shown: usize = 0;
            var idx = requests.len;
            while (idx > 0 and shown < visible_count) : (shown += 1) {
                idx -= 1;
                if (idx < self.request_scroll) break;

                const req = requests[idx];
                try self.moveTo(row + 1 + shown, col);
                try self.term.write(self.theme.border);
                try self.term.write(box.v);
                try self.term.write(self.theme.reset);

                // Method
                try self.term.write(" ");
                try self.term.write(self.theme.accent);
                try self.term.write(req.getMethod());
                try self.term.write(self.theme.reset);

                // Path (truncated)
                try self.term.write(" ");
                const path = req.getPath();
                const max_path: usize = if (width > 60) 25 else 15;
                const truncated_path = unicode.truncateToWidth(path, max_path);
                try self.term.write(truncated_path);

                // Status code with color
                try self.term.write("  ");
                if (req.status_code >= 500) {
                    try self.term.write(self.theme.@"error");
                } else if (req.status_code >= 400) {
                    try self.term.write(self.theme.warning);
                } else {
                    try self.term.write(self.theme.success);
                }
                var status_buf: [8]u8 = undefined;
                const status_str = std.fmt.bufPrint(&status_buf, "{d}", .{req.status_code}) catch "???";
                try self.term.write(status_str);
                try self.term.write(self.theme.reset);

                // Latency
                try self.term.write("  ");
                var lat_buf: [16]u8 = undefined;
                const lat_str = std.fmt.bufPrint(&lat_buf, "{d}ms", .{req.latency_ms}) catch "??ms";
                try self.term.write(lat_str);

                // Tokens
                try self.term.write("  ");
                var tok_buf: [16]u8 = undefined;
                const tok_str = std.fmt.bufPrint(&tok_buf, "{d} tok", .{req.token_count}) catch "?? tok";
                try self.term.write(self.theme.text_dim);
                try self.term.write(tok_str);
                try self.term.write(self.theme.reset);

                // Right border
                try self.moveTo(row + 1 + shown, col + width - 1);
                try self.term.write(self.theme.border);
                try self.term.write(box.v);
                try self.term.write(self.theme.reset);
            }
        }

        // Fill remaining rows
        var r: usize = if (requests.len == 0) 2 else visible_count + 1;
        while (r < height) : (r += 1) {
            try self.moveTo(row + r, col);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);

            try self.moveTo(row + r, col + width - 1);
            try self.term.write(self.theme.border);
            try self.term.write(box.v);
            try self.term.write(self.theme.reset);
        }
    }

    fn renderFooter(self: *Self, row: usize, col: usize, width: usize) !void {
        try self.moveTo(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(box.bl);

        const help = " [r]efresh  [l]og toggle  [c]lear  [+/-] poll rate  [q]uit ";
        try self.term.write(self.theme.text_dim);
        try self.term.write(help);
        try self.term.write(self.theme.reset);

        // Poll interval indicator
        try self.term.write(self.theme.text_dim);
        var poll_buf: [16]u8 = undefined;
        const poll_str = std.fmt.bufPrint(&poll_buf, " [{d}ms]", .{self.poll_interval_ms}) catch "";
        try self.term.write(poll_str);
        try self.term.write(self.theme.reset);

        // Fill remaining
        const used = unicode.displayWidth(help) + poll_str.len + 1;
        try self.term.write(self.theme.border);
        const footer_fill = if (width > used + 1) width - used - 1 else 0;
        try render_utils.writeRepeat(self.term, box.h, footer_fill);
        try self.term.write(box.br);
        try self.term.write(self.theme.reset);
    }

    /// Handle keyboard input
    pub fn handleKey(_: *Self, key: events.Key) ?Action {
        switch (key.code) {
            .character => switch (key.char) {
                'q' => return .quit,
                'r' => return .refresh,
                'l' => return .toggle_log,
                'c' => return .clear_stats,
                '+', '=' => return .increase_poll,
                '-' => return .decrease_poll,
                'j' => return .scroll_down,
                'k' => return .scroll_up,
                else => {},
            },
            .up => return .scroll_up,
            .down => return .scroll_down,
            .escape => return .quit,
            else => {},
        }
        return null;
    }

    /// Toggle request log visibility
    pub fn toggleLog(self: *Self) void {
        self.show_request_log = !self.show_request_log;
    }

    /// Increase poll interval
    pub fn increasePollRate(self: *Self) void {
        if (self.poll_interval_ms < 5000) {
            self.poll_interval_ms += 100;
        }
    }

    /// Decrease poll interval
    pub fn decreasePollRate(self: *Self) void {
        if (self.poll_interval_ms > 100) {
            self.poll_interval_ms -= 100;
        }
    }

    /// Scroll request log up
    pub fn scrollUp(self: *Self) void {
        if (self.request_scroll > 0) {
            self.request_scroll -= 1;
        }
    }

    /// Scroll request log down
    pub fn scrollDown(self: *Self) void {
        if (self.request_scroll + 10 < self.recent_requests.items.len) {
            self.request_scroll += 1;
        }
    }

    /// Get current TTFT P50
    pub fn getTtftP50(self: *Self) u32 {
        return self.ttft_percentiles.getPercentile(50);
    }

    /// Get current TTFT P99
    pub fn getTtftP99(self: *Self) u32 {
        return self.ttft_percentiles.getPercentile(99);
    }

    /// Get average throughput
    pub fn getAverageThroughput(self: *const Self) f32 {
        return @floatCast(self.throughput_history.average());
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "StreamingDashboard basic initialization" {
    _ = StreamingDashboard;
}

test "ServerStatus helpers" {
    try std.testing.expectEqualStrings("●", StreamingDashboard.ServerStatus.online.toIcon());
    try std.testing.expectEqualStrings("○", StreamingDashboard.ServerStatus.offline.toIcon());
    try std.testing.expectEqualStrings("Online", StreamingDashboard.ServerStatus.online.toString());
    try std.testing.expectEqualStrings("Degraded", StreamingDashboard.ServerStatus.degraded.toString());
}

test "RequestLogEntry creation" {
    const entry = StreamingDashboard.RequestLogEntry.init("POST", "/v1/chat/completions", 200, 150, 42);

    try std.testing.expectEqualStrings("POST", entry.getMethod());
    try std.testing.expectEqualStrings("/v1/chat/completions", entry.getPath());
    try std.testing.expectEqual(@as(u16, 200), entry.status_code);
    try std.testing.expectEqual(@as(u32, 150), entry.latency_ms);
    try std.testing.expectEqual(@as(u32, 42), entry.token_count);
}

test "RequestLogEntry truncation" {
    const long_path = "/this/is/a/very/long/path/that/should/be/truncated/to/fit/within/64/characters/limit";
    const entry = StreamingDashboard.RequestLogEntry.init("GET", long_path, 200, 10, 5);

    try std.testing.expectEqual(@as(u8, 64), entry.path_len);
}

test "Action enum completeness" {
    const actions = [_]StreamingDashboard.Action{
        .quit,
        .refresh,
        .toggle_log,
        .clear_stats,
        .increase_poll,
        .decrease_poll,
        .scroll_up,
        .scroll_down,
    };
    try std.testing.expectEqual(@as(usize, 8), actions.len);
}
