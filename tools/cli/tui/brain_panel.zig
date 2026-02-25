//! Brain Dashboard Panel — 6-panel data dashboard view.
//!
//! Displays WDBX vector database stats and Abbey learning metrics
//! in a 3x2 grid layout with sparklines, gauges, and bar charts.

const std = @import("std");
const terminal_mod = @import("terminal.zig");
const themes = @import("themes.zig");
const widgets = @import("widgets.zig");
const ring_buffer = @import("ring_buffer.zig");

const Terminal = terminal_mod.Terminal;
const Theme = themes.Theme;
const SparklineChart = widgets.SparklineChart;
const ProgressGauge = widgets.ProgressGauge;

pub const RingBuf120 = ring_buffer.RingBuffer(f32, 120);

// ===============================================================================
// Brain Dashboard Panel
// ===============================================================================

pub const BrainDashboardPanel = struct {
    term: *Terminal,
    theme: *const Theme,

    pub fn init(term: *Terminal, theme: *const Theme) BrainDashboardPanel {
        return .{ .term = term, .theme = theme };
    }

    /// Render the full 6-panel dashboard layout.
    pub fn render(
        self: *const BrainDashboardPanel,
        data: *const DashboardData,
        start_row: u16,
        start_col: u16,
        width: u16,
        height: u16,
    ) !void {
        const term = self.term;
        const theme = self.theme;

        if (width < 40 or height < 12) return;

        if (data.is_training_mode) {
            return self.renderTrainingMode(data, start_row, width, height);
        }

        const half_w = width / 2;
        const third_h = height / 3;
        const bottom_h = height - third_h * 2; // Use remaining height for bottom row
        const row0 = start_row;
        const row1 = start_row + third_h;
        const row2 = start_row + third_h * 2;

        // Row 1: WDBX Status | Learning Status
        try self.renderPanel(term, theme, "WDBX Status", row0, start_col, half_w, third_h);
        try self.renderWdbxStatus(data, row0 + 1, start_col + 1, half_w - 2, third_h - 2);

        try self.renderPanel(term, theme, "Learning Status", row0, start_col + half_w, half_w, third_h);
        try self.renderLearningStatus(data, row0 + 1, start_col + half_w + 1, half_w - 2, third_h - 2);

        // Row 2: Throughput | Reward History
        try self.renderPanel(term, theme, "Throughput", row1, start_col, half_w, third_h);
        try self.renderThroughput(data, row1 + 1, start_col + 1, half_w - 2, third_h - 2);

        try self.renderPanel(term, theme, "Reward History", row1, start_col + half_w, half_w, third_h);
        try self.renderRewardHistory(data, row1 + 1, start_col + half_w + 1, half_w - 2, third_h - 2);

        // Row 3: Similarity Scores | Attention Pattern (uses remaining height)
        try self.renderPanel(term, theme, "Similarity", row2, start_col, half_w, bottom_h);
        try self.renderSimilarity(data, row2 + 1, start_col + 1, half_w - 2, bottom_h -| 2);

        try self.renderPanel(term, theme, "Attention", row2, start_col + half_w, half_w, bottom_h);
        try self.renderAttention(data, row2 + 1, start_col + half_w + 1, half_w - 2, bottom_h -| 2);
    }

    // ── Panel Chrome ──────────────────────────────────────────────────────────

    fn renderPanel(
        self: *const BrainDashboardPanel,
        term: *Terminal,
        theme: *const Theme,
        title: []const u8,
        row: u16,
        col: u16,
        width: u16,
        height: u16,
    ) !void {
        _ = self;
        if (width < 4 or height < 2) return;
        const inner: usize = @as(usize, width) - 2;

        // Top border with title
        try setCursor(term, row, col);
        try term.write(theme.border);
        try term.write(widgets.box.tl);
        try term.write(theme.reset);
        try term.write(theme.bold);
        try term.write(theme.primary);
        try term.write(" ");
        const title_max = @min(title.len, inner -| 4);
        try term.write(title[0..title_max]);
        try term.write(" ");
        try term.write(theme.reset);
        try term.write(theme.border);
        const title_used = title_max + 2;
        if (title_used < inner) {
            for (0..inner - title_used) |_| try term.write(widgets.box.h);
        }
        try term.write(widgets.box.tr);
        try term.write(theme.reset);

        // Side borders
        for (1..height - 1) |r| {
            try setCursor(term, row + @as(u16, @intCast(r)), col);
            try term.write(theme.border);
            try term.write(widgets.box.v);
            try term.write(theme.reset);
            for (0..inner) |_| try term.write(" ");
            try term.write(theme.border);
            try term.write(widgets.box.v);
            try term.write(theme.reset);
        }

        // Bottom border
        try setCursor(term, row + height - 1, col);
        try term.write(theme.border);
        try term.write(widgets.box.bl);
        for (0..inner) |_| try term.write(widgets.box.h);
        try term.write(widgets.box.br);
        try term.write(theme.reset);
    }

    // ── Content Renderers ─────────────────────────────────────────────────────

    fn renderWdbxStatus(self: *const BrainDashboardPanel, data: *const DashboardData, row: u16, col: u16, width: u16, _: u16) !void {
        const term = self.term;
        const theme = self.theme;
        var buf: [64]u8 = undefined;

        try setCursor(term, row, col);
        try term.write(theme.text);
        try term.write(" Vectors: ");
        try term.write(theme.reset);
        try term.write(theme.bold);
        const vc = std.fmt.bufPrint(&buf, "{d}", .{data.vector_count}) catch "?";
        try term.write(vc);
        try term.write(theme.reset);

        if (width > 32) {
            try term.write(theme.text_dim);
            const dim = std.fmt.bufPrint(&buf, "  Dim: {d}", .{data.dimension}) catch "";
            try term.write(dim);
            try term.write(theme.reset);
        }

        try setCursor(term, row + 1, col);
        try term.write(theme.text);
        try term.write(" Size: ");
        try term.write(theme.reset);
        const size_str = formatBytes(data.db_size_bytes, &buf);
        try term.write(size_str);

        try setCursor(term, row + 2, col);
        try term.write(theme.text);
        try term.write(" Ops: ");
        try term.write(theme.reset);
        const ops = std.fmt.bufPrint(&buf, "{d} ins  {d} qry", .{ data.total_inserts, data.total_searches }) catch "?";
        try term.write(ops);
    }

    fn renderLearningStatus(self: *const BrainDashboardPanel, data: *const DashboardData, row: u16, col: u16, _: u16, _: u16) !void {
        const term = self.term;
        const theme = self.theme;
        var buf: [64]u8 = undefined;

        try setCursor(term, row, col);
        try term.write(theme.text);
        try term.write(" Phase: ");
        try term.write(theme.reset);
        try term.write(data.learning_phase.color(theme));
        try term.write(data.learning_phase.name());
        try term.write(theme.reset);

        try setCursor(term, row + 1, col);
        try term.write(theme.text);
        try term.write(" Episode: ");
        try term.write(theme.reset);
        const ep = std.fmt.bufPrint(&buf, "{d}", .{data.episode_count}) catch "?";
        try term.write(ep);

        try setCursor(term, row + 2, col);
        try term.write(theme.text);
        const eps = std.fmt.bufPrint(&buf, " Epsilon: {d:.3}", .{data.exploration_rate}) catch " Epsilon: ?";
        try term.write(eps);
        try term.write(theme.reset);

        // Replay buffer gauge
        if (data.replay_buffer_capacity > 0) {
            try setCursor(term, row + 3, col);
            try term.write(theme.text);
            try term.write(" Replay: ");
            try term.write(theme.reset);
            const pct: u8 = @intCast(@min(100, (data.replay_buffer_size * 100) / data.replay_buffer_capacity));
            var gauge_buf: [128]u8 = undefined;
            const gauge = ProgressGauge.render(pct, 12, &gauge_buf);
            try term.write(gauge);
            const pct_str = std.fmt.bufPrint(&buf, " {d}%", .{pct}) catch "";
            try term.write(pct_str);
        }
    }

    fn renderThroughput(self: *const BrainDashboardPanel, data: *const DashboardData, row: u16, col: u16, width: u16, _: u16) !void {
        const term = self.term;
        const theme = self.theme;
        var buf: [64]u8 = undefined;

        // Insert rate sparkline
        try setCursor(term, row, col);
        try term.write(theme.success);
        try term.write(" ins ");
        try term.write(theme.reset);

        var spark_vals: [120]u8 = undefined;
        const spark_count = ringToSparkline(&data.insert_rate, &spark_vals);
        const spark_width = @min(spark_count, @as(usize, width) -| 8);
        var spark_buf: [512]u8 = undefined;
        if (spark_width > 0) {
            const spark = SparklineChart.render(spark_vals[spark_count - spark_width .. spark_count], &spark_buf);
            try term.write(theme.success);
            try term.write(spark);
            try term.write(theme.reset);
        }

        const ins_rate = data.insert_rate.latest() orelse 0;
        const rate_str = std.fmt.bufPrint(&buf, " {d:.0}/s", .{ins_rate}) catch "";
        try term.write(theme.text_dim);
        try term.write(rate_str);
        try term.write(theme.reset);

        // Search rate sparkline
        try setCursor(term, row + 2, col);
        try term.write(theme.info);
        try term.write(" qry ");
        try term.write(theme.reset);

        var spark_vals2: [120]u8 = undefined;
        const spark_count2 = ringToSparkline(&data.search_rate, &spark_vals2);
        const spark_width2 = @min(spark_count2, @as(usize, width) -| 8);
        var spark_buf2: [512]u8 = undefined;
        if (spark_width2 > 0) {
            const spark2 = SparklineChart.render(spark_vals2[spark_count2 - spark_width2 .. spark_count2], &spark_buf2);
            try term.write(theme.info);
            try term.write(spark2);
            try term.write(theme.reset);
        }

        const search_rate = data.search_rate.latest() orelse 0;
        const rate_str2 = std.fmt.bufPrint(&buf, " {d:.0}/s", .{search_rate}) catch "";
        try term.write(theme.text_dim);
        try term.write(rate_str2);
        try term.write(theme.reset);
    }

    fn renderRewardHistory(self: *const BrainDashboardPanel, data: *const DashboardData, row: u16, col: u16, width: u16, _: u16) !void {
        const term = self.term;
        const theme = self.theme;
        var buf: [64]u8 = undefined;

        // Reward sparkline
        try setCursor(term, row, col);
        try term.write(theme.text);
        try term.write(" R ");
        try term.write(theme.reset);

        var spark_vals: [120]u8 = undefined;
        const spark_count = ringToSparkline(&data.reward_history, &spark_vals);
        const spark_width = @min(spark_count, @as(usize, width) -| 6);
        var spark_buf: [512]u8 = undefined;
        if (spark_width > 0) {
            const spark = SparklineChart.render(spark_vals[spark_count - spark_width .. spark_count], &spark_buf);
            try term.write(theme.warning);
            try term.write(spark);
            try term.write(theme.reset);
        }

        // Loss sparkline
        try setCursor(term, row + 2, col);
        try term.write(theme.text);
        try term.write(" L ");
        try term.write(theme.reset);

        var spark_vals2: [120]u8 = undefined;
        const spark_count2 = ringToSparkline(&data.loss_history, &spark_vals2);
        const spark_width2 = @min(spark_count2, @as(usize, width) -| 6);
        var spark_buf2: [512]u8 = undefined;
        if (spark_width2 > 0) {
            const spark2 = SparklineChart.render(spark_vals2[spark_count2 - spark_width2 .. spark_count2], &spark_buf2);
            try term.write(theme.@"error"); // themed red for loss
            try term.write(spark2);
            try term.write(theme.reset);
        }

        const loss_val = data.loss_history.latest() orelse 0;
        const loss_str = std.fmt.bufPrint(&buf, " {d:.3}", .{loss_val}) catch "";
        try term.write(theme.text_dim);
        try term.write(loss_str);
        try term.write(theme.reset);
    }

    fn renderSimilarity(self: *const BrainDashboardPanel, data: *const DashboardData, row: u16, col: u16, width: u16, _: u16) !void {
        const term = self.term;
        const theme = self.theme;
        var buf: [64]u8 = undefined;

        // Sparkline
        try setCursor(term, row, col);
        try term.write(theme.text);
        try term.write(" sim ");
        try term.write(theme.reset);

        var spark_vals: [120]u8 = undefined;
        const spark_count = ringToSparkline(&data.similarity_scores, &spark_vals);
        const spark_width = @min(spark_count, @as(usize, width) -| 8);
        var spark_buf: [512]u8 = undefined;
        if (spark_width > 0) {
            const spark = SparklineChart.render(spark_vals[spark_count - spark_width .. spark_count], &spark_buf);
            try term.write(theme.primary);
            try term.write(spark);
            try term.write(theme.reset);
        }

        // Stats row
        try setCursor(term, row + 2, col);
        const sim_min = data.similarity_scores.min() orelse 0;
        const sim_max = data.similarity_scores.max() orelse 0;
        const sim_avg = data.similarity_scores.average();
        const stats = std.fmt.bufPrint(&buf, " min={d:.2} avg={d:.2} max={d:.2}", .{
            sim_min, @as(f32, @floatCast(sim_avg)), sim_max,
        }) catch " (no data)";
        try term.write(theme.text_dim);
        try term.write(stats);
        try term.write(theme.reset);
    }

    fn renderAttention(self: *const BrainDashboardPanel, data: *const DashboardData, row: u16, col: u16, width: u16, height: u16) !void {
        const term = self.term;
        const theme = self.theme;

        // 8-head attention bar chart using block characters
        const bar_height = @min(height -| 1, @as(u16, 6));
        const bar_width = @min(@as(u16, @intCast(@as(usize, width) -| 2)) / 8, 4);

        for (0..8) |head| {
            const weight = data.attention_weights[head];
            const filled = @as(u16, @intFromFloat(@min(1.0, @max(0.0, weight)) * @as(f32, @floatFromInt(bar_height))));

            for (0..bar_height) |r| {
                const bar_row = row + bar_height - @as(u16, @intCast(r));
                const bar_col = col + 1 + @as(u16, @intCast(head)) * (bar_width + 1);
                try setCursor(term, bar_row, bar_col);

                if (@as(u16, @intCast(r)) < filled) {
                    // Color by weight intensity
                    const color = if (weight > 0.7) theme.success else if (weight > 0.4) theme.info else theme.text_dim;
                    try term.write(color);
                    for (0..bar_width) |_| try term.write("\u{2588}");
                    try term.write(theme.reset);
                } else {
                    try term.write(theme.text_muted);
                    for (0..bar_width) |_| try term.write("\u{2591}");
                    try term.write(theme.reset);
                }
            }

            // Head label
            var buf: [4]u8 = undefined;
            const label = std.fmt.bufPrint(&buf, "H{d}", .{head}) catch "?";
            try setCursor(term, row + bar_height + 1, col + 1 + @as(u16, @intCast(head)) * (bar_width + 1));
            try term.write(theme.text_dim);
            try term.write(label);
            try term.write(theme.reset);
        }
    }

    // ── Training Mode Renderers ───────────────────────────────────────────────

    fn renderTrainingMode(
        self: *const BrainDashboardPanel,
        data: *const DashboardData,
        start_row: u16,
        width: u16,
        height: u16,
    ) !void {
        const term = self.term;
        const theme = self.theme;
        const half_w = width / 2;
        const third_h = height / 3;
        const bottom_h = height - third_h * 2; // Use remaining height for bottom row
        const row0 = start_row;
        const row1 = start_row + third_h;
        const row2 = start_row + third_h * 2;
        // Use start_col=0 since renderTrainingMode is called from render()
        // which already accounts for start_col in the caller's coordinate space.
        const start_col: u16 = 0;

        // Row 1: Training Status | Optimizer
        try self.renderPanel(term, theme, "Training", row0, start_col, half_w, third_h);
        try self.renderTrainingStatus(data, row0 + 1, start_col + 1, half_w - 2, third_h - 2);
        try self.renderPanel(term, theme, "Optimizer", row0, start_col + half_w, half_w, third_h);
        try self.renderOptimizerStatus(data, row0 + 1, start_col + half_w + 1, half_w - 2, third_h - 2);

        // Row 2: Throughput | Loss / Accuracy
        try self.renderPanel(term, theme, "Throughput", row1, start_col, half_w, third_h);
        try self.renderThroughput(data, row1 + 1, start_col + 1, half_w - 2, third_h - 2);
        try self.renderPanel(term, theme, "Loss / Accuracy", row1, start_col + half_w, half_w, third_h);
        try self.renderRewardHistory(data, row1 + 1, start_col + half_w + 1, half_w - 2, third_h - 2);

        // Row 3: Perplexity | GPU (uses remaining height)
        try self.renderPanel(term, theme, "Perplexity", row2, start_col, half_w, bottom_h);
        try self.renderSimilarity(data, row2 + 1, start_col + 1, half_w - 2, bottom_h -| 2);
        try self.renderPanel(term, theme, "GPU", row2, start_col + half_w, half_w, bottom_h);
        try self.renderAttention(data, row2 + 1, start_col + half_w + 1, half_w - 2, bottom_h -| 2);
    }

    fn renderTrainingStatus(self: *const BrainDashboardPanel, data: *const DashboardData, row: u16, col: u16, _: u16, _: u16) !void {
        const term = self.term;
        const theme = self.theme;
        var buf: [64]u8 = undefined;

        try setCursor(term, row, col);
        try term.write(theme.text);
        try term.write(" Epoch: ");
        try term.write(theme.reset);
        try term.write(theme.bold);
        const ep = std.fmt.bufPrint(&buf, "{d}/{d}", .{ data.current_epoch, data.total_epochs }) catch "?";
        try term.write(ep);
        try term.write(theme.reset);

        try setCursor(term, row + 1, col);
        try term.write(theme.text);
        try term.write(" Step: ");
        try term.write(theme.reset);
        const step = std.fmt.bufPrint(&buf, "{d}", .{data.current_step}) catch "?";
        try term.write(step);

        try setCursor(term, row + 2, col);
        try term.write(theme.text);
        try term.write(" Loss: ");
        try term.write(theme.reset);
        const color = if (data.train_loss > 0.5) theme.warning else if (data.train_loss > 0.1) theme.info else theme.success;
        try term.write(color);
        const loss = std.fmt.bufPrint(&buf, "{d:.4}", .{data.train_loss}) catch "?";
        try term.write(loss);
        try term.write(theme.reset);

        try setCursor(term, row + 3, col);
        try term.write(theme.text);
        try term.write(" PPL: ");
        try term.write(theme.reset);
        const ppl = std.fmt.bufPrint(&buf, "{d:.1}", .{data.perplexity}) catch "?";
        try term.write(ppl);
    }

    fn renderOptimizerStatus(self: *const BrainDashboardPanel, data: *const DashboardData, row: u16, col: u16, _: u16, _: u16) !void {
        const term = self.term;
        const theme = self.theme;
        var buf: [64]u8 = undefined;

        try setCursor(term, row, col);
        try term.write(theme.text);
        try term.write(" Phase: ");
        try term.write(theme.reset);
        try term.write(data.learning_phase.color(theme));
        try term.write(data.learning_phase.name());
        try term.write(theme.reset);

        try setCursor(term, row + 1, col);
        try term.write(theme.text);
        try term.write(" LR: ");
        try term.write(theme.reset);
        const lr = std.fmt.bufPrint(&buf, "{e:.3}", .{data.learning_rate_current}) catch "?";
        try term.write(lr);

        try setCursor(term, row + 2, col);
        try term.write(theme.text);
        try term.write(" Acc: ");
        try term.write(theme.reset);
        try term.write(theme.bold);
        const acc = std.fmt.bufPrint(&buf, "{d:.1}%", .{data.train_accuracy * 100.0}) catch "?";
        try term.write(acc);
        try term.write(theme.reset);

        // GPU utilization gauge
        try setCursor(term, row + 3, col);
        try term.write(theme.text);
        try term.write(" GPU: ");
        try term.write(theme.reset);
        const pct: u8 = @intFromFloat(@min(100.0, data.gpu_utilization * 100.0));
        var gauge_buf: [128]u8 = undefined;
        const gauge = ProgressGauge.render(pct, 12, &gauge_buf);
        try term.write(gauge);
        const pct_str = std.fmt.bufPrint(&buf, " {d}%", .{pct}) catch "";
        try term.write(pct_str);
    }

    fn formatBytes(bytes: u64, buf: *[64]u8) []const u8 {
        const gb: u64 = 1024 * 1024 * 1024;
        const mb: u64 = 1024 * 1024;
        const kb: u64 = 1024;
        if (bytes >= gb) {
            return std.fmt.bufPrint(buf, "{d} GB", .{bytes / gb}) catch "?";
        }
        if (bytes >= mb) {
            return std.fmt.bufPrint(buf, "{d} MB", .{bytes / mb}) catch "?";
        }
        if (bytes >= kb) {
            return std.fmt.bufPrint(buf, "{d} KB", .{bytes / kb}) catch "?";
        }
        return std.fmt.bufPrint(buf, "{d} B", .{bytes}) catch "?";
    }
};

// ===============================================================================
// Dashboard Data
// ===============================================================================

pub const DashboardData = struct {
    // WDBX metrics
    vector_count: u64,
    insert_rate: RingBuf120,
    search_rate: RingBuf120,
    similarity_scores: RingBuf120,
    db_size_bytes: u64,
    dimension: u32,
    total_searches: u64,
    total_inserts: u64,

    // Abbey learning metrics
    learning_phase: @import("agent_panel.zig").LearningPhase,
    exploration_rate: f32,
    episode_count: u64,
    reward_history: RingBuf120,
    loss_history: RingBuf120,
    attention_weights: [8]f32,
    replay_buffer_size: u64,
    replay_buffer_capacity: u64,

    // Animation data
    node_activity: [64]f32,

    // Training mode fields
    is_training_mode: bool,
    train_loss: f32,
    train_accuracy: f32,
    learning_rate_current: f32,
    grad_norm: f32,
    perplexity: f32,
    tokens_per_sec: f32,
    gpu_utilization: f32,
    gpu_backend_name: [32]u8,
    current_epoch: u32,
    total_epochs: u32,
    current_step: u64,

    pub fn init() DashboardData {
        return .{
            .vector_count = 0,
            .insert_rate = RingBuf120.init(),
            .search_rate = RingBuf120.init(),
            .similarity_scores = RingBuf120.init(),
            .db_size_bytes = 0,
            .dimension = 128,
            .total_searches = 0,
            .total_inserts = 0,
            .learning_phase = .exploration,
            .exploration_rate = 1.0,
            .episode_count = 0,
            .reward_history = RingBuf120.init(),
            .loss_history = RingBuf120.init(),
            .attention_weights = [_]f32{0.0} ** 8,
            .replay_buffer_size = 0,
            .replay_buffer_capacity = 10000,
            .node_activity = [_]f32{0.0} ** 64,
            .is_training_mode = false,
            .train_loss = 0,
            .train_accuracy = 0,
            .learning_rate_current = 0,
            .grad_norm = 0,
            .perplexity = 0,
            .tokens_per_sec = 0,
            .gpu_utilization = 0,
            .gpu_backend_name = [_]u8{0} ** 32,
            .current_epoch = 0,
            .total_epochs = 0,
            .current_step = 0,
        };
    }

    /// Update with simulated demo data — called once per tick (~10Hz).
    /// This generates procedural time-varying data for the visual demo.
    pub fn updateSimulated(self: *DashboardData, tick: u64) void {
        const t = @as(f32, @floatFromInt(tick)) * 0.1;

        // WDBX metrics: smooth sinusoidal patterns with noise
        const base_insert = 50.0 + 30.0 * std.math.sin(t * 0.07);
        const base_search = 80.0 + 40.0 * std.math.sin(t * 0.05 + 1.0);
        self.insert_rate.push(@max(0, base_insert + 10.0 * std.math.sin(t * 0.31)));
        self.search_rate.push(@max(0, base_search + 15.0 * std.math.sin(t * 0.23)));

        self.total_inserts +%= 1;
        if (tick % 2 == 0) self.total_searches +%= 1;

        self.vector_count = 10000 + self.total_inserts * 5;
        self.db_size_bytes = self.vector_count * 512 + 1024 * 1024;

        // Similarity: oscillating between 0.6 and 0.95
        const sim = 0.78 + 0.15 * std.math.sin(t * 0.09 + 0.5);
        self.similarity_scores.push(@max(0, @min(1.0, sim)));

        // Learning: slow phase transitions
        const phase_val = (tick / 200) % 3;
        self.learning_phase = switch (phase_val) {
            0 => .exploration,
            1 => .exploitation,
            else => .converged,
        };

        self.exploration_rate = @max(0.01, 1.0 - @as(f32, @floatFromInt(@min(tick, 1000))) * 0.001);
        self.episode_count = tick / 10;

        // Reward: trending upward with variance
        const reward = 0.3 + 0.4 * std.math.sin(t * 0.03) + @as(f32, @floatFromInt(tick % 7)) * 0.02;
        self.reward_history.push(@max(0, @min(1.0, reward)));

        // Loss: trending downward
        const loss = @max(0.01, 0.8 - @as(f32, @floatFromInt(@min(tick, 500))) * 0.001 + 0.05 * std.math.sin(t * 0.17));
        self.loss_history.push(loss);

        // Attention: 8 heads with different frequencies
        for (&self.attention_weights, 0..) |*w, i| {
            const freq = 0.05 + @as(f32, @floatFromInt(i)) * 0.02;
            const phase = @as(f32, @floatFromInt(i)) * 0.8;
            w.* = 0.5 + 0.4 * std.math.sin(t * freq + phase);
        }

        // Replay buffer: slowly filling
        self.replay_buffer_size = @min(self.replay_buffer_capacity, tick * 10);

        // Map metrics to node activity for the brain animation
        self.updateNodeActivity(t);
    }

    fn updateNodeActivity(self: *DashboardData, t: f32) void {
        // Input layer (0-7): driven by insert rate
        const ins = (self.insert_rate.latest() orelse 0) / 100.0;
        for (0..8) |i| {
            const phase = @as(f32, @floatFromInt(i)) * 0.5;
            self.node_activity[i] = @min(1.0, ins * (0.7 + 0.3 * std.math.sin(t * 0.2 + phase)));
        }

        // Hidden layers (8-47): driven by search rate + attention
        const search = (self.search_rate.latest() orelse 0) / 120.0;
        for (8..48) |i| {
            const head = (i - 8) % 8;
            const attn = self.attention_weights[head];
            const phase = @as(f32, @floatFromInt(i)) * 0.3;
            self.node_activity[i] = @min(1.0, search * attn * (0.6 + 0.4 * std.math.sin(t * 0.15 + phase)));
        }

        // Output layer (48-63): driven by similarity + reward
        const sim = self.similarity_scores.latest() orelse 0;
        const reward = self.reward_history.latest() orelse 0;
        for (48..64) |i| {
            const phase = @as(f32, @floatFromInt(i)) * 0.4;
            self.node_activity[i] = @min(1.0, (sim + reward) * 0.5 * (0.6 + 0.4 * std.math.sin(t * 0.1 + phase)));
        }
    }
};

// ===============================================================================
// Helpers
// ===============================================================================

fn ringToSparkline(ring: *const RingBuf120, out: *[120]u8) usize {
    var raw_buf: [120]f32 = undefined;
    const raw = ring.toSlice(&raw_buf);
    if (raw.len == 0) {
        out[0] = 50;
        return 1;
    }

    // Find range for normalization
    var vmin: f32 = raw[0];
    var vmax: f32 = raw[0];
    for (raw) |v| {
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    const range = if (vmax - vmin > 0.001) vmax - vmin else 1.0;

    for (raw, 0..) |v, i| {
        out[i] = @intFromFloat(@min(100.0, @max(0.0, (v - vmin) / range * 100.0)));
    }
    return raw.len;
}

fn setCursor(term: *Terminal, row: u16, col: u16) !void {
    // Match agent_panel/gpu_monitor convention: saturating -1 for 0-indexed coords.
    try term.moveTo(row -| 1, col -| 1);
}

test {
    std.testing.refAllDecls(@This());
}
