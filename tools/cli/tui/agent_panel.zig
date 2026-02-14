//! Agent Status Panel for TUI
//!
//! Displays AI agent learning progress, decision history,
//! and real-time performance metrics.

const std = @import("std");
const abi = @import("abi");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const widgets = @import("widgets.zig");

// ===============================================================================
// Types
// ===============================================================================

/// Agent learning phase
pub const LearningPhase = enum {
    exploration,
    exploitation,
    converged,

    pub fn name(self: LearningPhase) []const u8 {
        return switch (self) {
            .exploration => "Exploring",
            .exploitation => "Exploiting",
            .converged => "Converged",
        };
    }

    pub fn color(self: LearningPhase, theme: *const themes.Theme) []const u8 {
        return switch (self) {
            .exploration => theme.warning,
            .exploitation => theme.info,
            .converged => theme.success,
        };
    }
};

/// Decision history entry
pub const DecisionEntry = struct {
    timestamp: i64,
    workload_type: []const u8,
    selected_backend: []const u8,
    actual_time_ms: u64,
    predicted_time_ms: u64,
    reward: f32,
};

/// Reward history buffer for sparkline visualization
pub const RewardHistory = struct {
    pub const HISTORY_SIZE = 30;

    /// Ring buffer of normalized reward values (0-100)
    values: [HISTORY_SIZE]u8,
    /// Current write position
    pos: usize,
    /// Number of values written
    count: usize,

    pub fn init() RewardHistory {
        return .{
            .values = [_]u8{50} ** HISTORY_SIZE,
            .pos = 0,
            .count = 0,
        };
    }

    pub fn push(self: *RewardHistory, reward: f32) void {
        // Normalize reward from [-10, 10] to [0, 100]
        const normalized = @min(100, @max(0, @as(u8, @intFromFloat((reward + 10.0) * 5.0))));
        self.values[self.pos] = normalized;
        self.pos = (self.pos + 1) % HISTORY_SIZE;
        if (self.count < HISTORY_SIZE) {
            self.count += 1;
        }
    }

    /// Get values in chronological order
    pub fn getValues(self: *const RewardHistory) [HISTORY_SIZE]u8 {
        var result: [HISTORY_SIZE]u8 = [_]u8{50} ** HISTORY_SIZE;
        if (self.count == 0) return result;

        const start = if (self.count < HISTORY_SIZE) 0 else self.pos;

        for (0..self.count) |i| {
            const idx = (start + i) % HISTORY_SIZE;
            result[i] = self.values[idx];
        }
        return result;
    }
};

// ===============================================================================
// Agent Panel Widget
// ===============================================================================

pub const AgentPanel = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,

    /// Current learning phase
    phase: LearningPhase,
    /// Episode count
    episode_count: u64,
    /// Total accumulated reward
    total_reward: f32,
    /// Average reward per episode
    avg_reward: f32,
    /// Current exploration rate (epsilon)
    exploration_rate: f32,
    /// Decision history (recent decisions)
    decision_history: std.ArrayListUnmanaged(DecisionEntry),
    /// Reward history for sparkline
    reward_history: RewardHistory,
    /// Update counter
    update_counter: u64,

    /// Sparkline characters (from lowest to highest)
    const sparkline_chars = [_][]const u8{ " ", "\u{2581}", "\u{2582}", "\u{2583}", "\u{2584}", "\u{2585}", "\u{2586}", "\u{2587}", "\u{2588}" };

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) AgentPanel {
        return .{
            .allocator = allocator,
            .term = term,
            .theme = theme,
            .phase = .exploration,
            .episode_count = 0,
            .total_reward = 0,
            .avg_reward = 0,
            .exploration_rate = 1.0,
            .decision_history = .empty,
            .reward_history = RewardHistory.init(),
            .update_counter = 0,
        };
    }

    pub fn deinit(self: *AgentPanel) void {
        self.decision_history.deinit(self.allocator);
    }

    /// Update agent statistics
    pub fn updateStats(self: *AgentPanel, episode: u64, total_reward: f32, exploration_rate: f32) void {
        self.episode_count = episode;
        self.total_reward = total_reward;
        self.exploration_rate = exploration_rate;

        // Update phase based on exploration rate
        if (exploration_rate > 0.5) {
            self.phase = .exploration;
        } else if (exploration_rate > 0.05) {
            self.phase = .exploitation;
        } else {
            self.phase = .converged;
        }

        // Track reward history
        self.reward_history.push(total_reward);

        // Calculate average reward
        if (episode > 0) {
            self.avg_reward = total_reward / @as(f32, @floatFromInt(episode));
        }
    }

    /// Record a scheduling decision
    pub fn recordDecision(self: *AgentPanel, entry: DecisionEntry) !void {
        // Keep only last 20 decisions using O(n) efficient removal
        if (self.decision_history.items.len >= 20) {
            const items = self.decision_history.items;
            std.mem.copyForwards(DecisionEntry, items[0 .. items.len - 1], items[1..]);
            self.decision_history.shrinkRetainingCapacity(items.len - 1);
        }
        try self.decision_history.append(self.allocator, entry);
    }

    /// Update with simulated data (for demo purposes)
    pub fn update(self: *AgentPanel) !void {
        self.update_counter += 1;

        // Simulate learning progress
        const episodes = self.update_counter;
        const epsilon_decay = 0.995;
        const new_epsilon = @max(0.01, std.math.pow(f64, epsilon_decay, @as(f64, @floatFromInt(episodes))));

        // Simulate reward (starts negative, improves over time)
        const base_reward: f32 = -5.0 + @as(f32, @floatFromInt(@min(episodes, 100))) * 0.1;
        const noise: f32 = @as(f32, @floatFromInt(episodes % 10)) * 0.2 - 1.0;
        const reward = base_reward + noise;

        self.updateStats(episodes, reward, @floatCast(new_epsilon));

        // Simulate a decision every few updates
        if (self.update_counter % 5 == 0) {
            const backends = [_][]const u8{ "CUDA", "Vulkan", "Metal", "CPU" };
            const workloads = [_][]const u8{ "MatMul", "Attention", "FFN", "Embed" };

            try self.recordDecision(.{
                .timestamp = abi.shared.utils.unixMs(),
                .workload_type = workloads[self.update_counter % 4],
                .selected_backend = backends[(self.update_counter / 5) % 4],
                .actual_time_ms = 50 + (self.update_counter % 100),
                .predicted_time_ms = 60 + (self.update_counter % 80),
                .reward = reward / 10.0,
            });
        }
    }

    /// Render the agent panel widget
    pub fn render(self: *AgentPanel, start_row: u16, start_col: u16, width: u16, height: u16) !void {
        var current_row = start_row;

        // Render header
        try self.renderHeader(current_row, start_col, width);
        current_row += 2;

        // Render stats
        if (current_row < start_row + height - 2) {
            try self.renderStats(current_row, start_col, width);
            current_row += 2;
        }

        // Render reward sparkline
        if (current_row < start_row + height - 2) {
            try self.renderSparkline(current_row, start_col, width);
            current_row += 2;
        }

        // Render recent decisions
        if (current_row < start_row + height - 2) {
            try self.renderDecisions(current_row, start_col, width, start_row + height - current_row - 1);
        }
    }

    fn renderHeader(self: *AgentPanel, row: u16, col: u16, width: u16) !void {
        try self.setCursorPosition(row, col);

        // Draw top border
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.tl);
        try self.writeRepeat(widgets.box.h, width - 2);
        try self.term.write(widgets.box.tr);
        try self.term.write(self.theme.reset);

        // Title line with phase indicator
        try self.setCursorPosition(row + 1, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        try self.term.write(" ");
        try self.term.write(self.theme.bold);
        try self.term.write(self.theme.category_ai);
        try self.term.write("Agent Status");
        try self.term.write(self.theme.reset);

        // Phase indicator
        try self.term.write(" [");
        try self.term.write(self.phase.color(self.theme));
        try self.term.write(self.phase.name());
        try self.term.write(self.theme.reset);
        try self.term.write("]");

        // Pad and close
        const content_len = 12 + 3 + self.phase.name().len + 2;
        if (content_len < width - 2) {
            try self.writeRepeat(" ", width - 2 - content_len);
        }
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
    }

    fn renderStats(self: *AgentPanel, row: u16, col: u16, width: u16) !void {
        // Separator line
        try self.setCursorPosition(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.lsep);
        try self.writeRepeat(widgets.box.h, width - 2);
        try self.term.write(widgets.box.rsep);
        try self.term.write(self.theme.reset);

        // Stats line
        try self.setCursorPosition(row + 1, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        try self.term.write(" ");

        // Episode count
        try self.term.write("Ep: ");
        try self.term.write(self.theme.accent);
        var buf: [64]u8 = undefined;
        const ep_str = std.fmt.bufPrint(&buf, "{d}", .{self.episode_count}) catch "?";
        try self.term.write(ep_str);
        try self.term.write(self.theme.reset);

        // Average reward
        try self.term.write("  Avg: ");
        if (self.avg_reward > 0) {
            try self.term.write(self.theme.success);
        } else {
            try self.term.write(self.theme.warning);
        }
        const avg_str = std.fmt.bufPrint(&buf, "{d:.2}", .{self.avg_reward}) catch "?";
        try self.term.write(avg_str);
        try self.term.write(self.theme.reset);

        // Exploration rate (epsilon)
        try self.term.write("  \xce\xb5: "); // epsilon character
        try self.term.write(self.theme.info);
        const eps_str = std.fmt.bufPrint(&buf, "{d:.3}", .{self.exploration_rate}) catch "?";
        try self.term.write(eps_str);
        try self.term.write(self.theme.reset);

        // Pad and close
        const content_len = 4 + ep_str.len + 7 + avg_str.len + 5 + eps_str.len + 2;
        if (content_len < width - 2) {
            try self.writeRepeat(" ", width - 2 - content_len);
        }
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
    }

    fn renderSparkline(self: *AgentPanel, row: u16, col: u16, width: u16) !void {
        const inner_width: u16 = if (width >= 2) width - 2 else 0;
        // Separator line
        try self.setCursorPosition(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.lsep);
        try self.writeRepeat(widgets.box.h, inner_width);
        try self.term.write(widgets.box.rsep);
        try self.term.write(self.theme.reset);

        // Sparkline content line
        try self.setCursorPosition(row + 1, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        try self.term.write(" Reward: ");

        const history = self.reward_history.getValues();
        const available = if (width > 15) width - 15 else 0;
        const max_chars = @min(RewardHistory.HISTORY_SIZE, available);

        for (0..max_chars) |i| {
            const val = history[i];
            const char_idx = @min(8, val / 12);
            const char = sparkline_chars[char_idx];

            // Color based on value (green for high, red for low)
            if (val > 70) {
                try self.term.write(self.theme.success);
            } else if (val > 40) {
                try self.term.write(self.theme.warning);
            } else {
                try self.term.write(self.theme.@"error");
            }
            try self.term.write(char);
        }
        try self.term.write(self.theme.reset);

        // Pad and close (use u32 for content_len to avoid overflow; max_chars is at most HISTORY_SIZE)
        const content_len: u32 = 9 + @as(u32, max_chars);
        if (content_len < inner_width) {
            try self.writeRepeat(" ", inner_width - @as(u16, @intCast(content_len)));
        }
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
    }

    fn renderDecisions(self: *AgentPanel, row: u16, col: u16, width: u16, max_rows: u16) !void {
        // Separator line
        try self.setCursorPosition(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.lsep);
        try self.writeRepeat(widgets.box.h, width - 2);
        try self.term.write(widgets.box.rsep);
        try self.term.write(self.theme.reset);

        // Header row
        try self.setCursorPosition(row + 1, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        try self.term.write(" ");
        try self.term.write(self.theme.secondary);
        try self.term.write("Recent Decisions:");
        try self.term.write(self.theme.reset);

        const content_len = 18;
        if (content_len < width - 2) {
            try self.writeRepeat(" ", width - 2 - content_len);
        }
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);

        // Decision rows
        const available_rows = if (max_rows > 3) max_rows - 3 else 0;
        const max_decisions = @min(self.decision_history.items.len, available_rows);
        const start_idx = if (self.decision_history.items.len > max_decisions)
            self.decision_history.items.len - max_decisions
        else
            0;

        var current_row = row + 2;
        for (self.decision_history.items[start_idx..]) |decision| {
            try self.setCursorPosition(current_row, col);

            try self.term.write(self.theme.border);
            try self.term.write(widgets.box.v);
            try self.term.write(self.theme.reset);
            try self.term.write("  ");

            // Workload type
            try self.term.write(self.theme.text_dim);
            var buf: [64]u8 = undefined;
            const wl = decision.workload_type[0..@min(8, decision.workload_type.len)];
            const wl_padded = std.fmt.bufPrint(&buf, "{s:<8}", .{wl}) catch wl;
            try self.term.write(wl_padded);
            try self.term.write(self.theme.reset);

            try self.term.write(" \xe2\x86\x92 "); // -> arrow

            // Selected backend
            try self.term.write(self.theme.primary);
            const be = decision.selected_backend[0..@min(6, decision.selected_backend.len)];
            const be_padded = std.fmt.bufPrint(&buf, "{s:<6}", .{be}) catch be;
            try self.term.write(be_padded);
            try self.term.write(self.theme.reset);

            // Time
            const time_str = std.fmt.bufPrint(&buf, " {d:>4}ms", .{decision.actual_time_ms}) catch "";
            try self.term.write(time_str);

            // Reward
            if (decision.reward > 0) {
                try self.term.write(self.theme.success);
            } else {
                try self.term.write(self.theme.@"error");
            }
            const reward_str = if (decision.reward >= 0)
                std.fmt.bufPrint(&buf, " +{d:.2}", .{decision.reward}) catch ""
            else
                std.fmt.bufPrint(&buf, " {d:.2}", .{decision.reward}) catch "";
            try self.term.write(reward_str);
            try self.term.write(self.theme.reset);

            // Pad and close
            const row_content_len = 2 + 8 + 4 + 6 + time_str.len + reward_str.len;
            if (row_content_len < width - 2) {
                try self.writeRepeat(" ", width - 2 - row_content_len);
            }
            try self.term.write(self.theme.border);
            try self.term.write(widgets.box.v);
            try self.term.write(self.theme.reset);

            current_row += 1;
        }

        // Bottom border
        try self.setCursorPosition(current_row, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.bl);
        try self.writeRepeat(widgets.box.h, width - 2);
        try self.term.write(widgets.box.br);
        try self.term.write(self.theme.reset);
    }

    fn setCursorPosition(self: *AgentPanel, row: u16, col: u16) !void {
        var buf: [16]u8 = undefined;
        const seq = std.fmt.bufPrint(&buf, "\x1b[{d};{d}H", .{ row, col }) catch return;
        try self.term.write(seq);
    }

    fn writeRepeat(self: *AgentPanel, char: []const u8, count: usize) !void {
        for (0..count) |_| {
            try self.term.write(char);
        }
    }
};

// ===============================================================================
// Tests
// ===============================================================================

test "learning phase names and colors" {
    try std.testing.expectEqualStrings("Exploring", LearningPhase.exploration.name());
    try std.testing.expectEqualStrings("Exploiting", LearningPhase.exploitation.name());
    try std.testing.expectEqualStrings("Converged", LearningPhase.converged.name());
}

test "reward history push and get" {
    var history = RewardHistory.init();

    history.push(5.0);
    history.push(-3.0);
    history.push(0.0);

    const values = history.getValues();
    try std.testing.expect(values[0] > 0);
    try std.testing.expect(history.count == 3);
}

test "reward history ring buffer" {
    var history = RewardHistory.init();

    // Fill buffer
    for (0..RewardHistory.HISTORY_SIZE) |i| {
        const reward: f32 = @as(f32, @floatFromInt(i)) - 10.0; // Range -10 to +20
        history.push(reward);
    }

    // Add more to test wrap-around
    history.push(5.0);
    history.push(7.0);

    const values = history.getValues();
    // Values should wrap around
    try std.testing.expect(history.count == RewardHistory.HISTORY_SIZE);
    try std.testing.expect(values[RewardHistory.HISTORY_SIZE - 2] > 0);
    try std.testing.expect(values[RewardHistory.HISTORY_SIZE - 1] > 0);
}

test "decision entry structure" {
    const entry = DecisionEntry{
        .timestamp = 1234567890,
        .workload_type = "MatMul",
        .selected_backend = "CUDA",
        .actual_time_ms = 50,
        .predicted_time_ms = 60,
        .reward = 0.5,
    };
    try std.testing.expect(entry.actual_time_ms == 50);
    try std.testing.expectEqualStrings("MatMul", entry.workload_type);
}

test "phase determination by exploration rate" {
    // Test exploration phase (rate > 0.5)
    const theme = &themes.themes.default;

    try std.testing.expectEqualStrings(theme.warning, LearningPhase.exploration.color(theme));
    try std.testing.expectEqualStrings(theme.info, LearningPhase.exploitation.color(theme));
    try std.testing.expectEqualStrings(theme.success, LearningPhase.converged.color(theme));
}

test "agent panel init and deinit" {
    const allocator = std.testing.allocator;
    var term = terminal.Terminal.init(allocator);
    defer term.deinit();

    const theme = &themes.themes.default;
    var panel = AgentPanel.init(allocator, &term, theme);
    defer panel.deinit();

    try std.testing.expectEqual(@as(usize, 0), panel.decision_history.items.len);
    try std.testing.expectEqual(LearningPhase.exploration, panel.phase);
    try std.testing.expectEqual(@as(f32, 1.0), panel.exploration_rate);
}

test "agent panel update stats" {
    const allocator = std.testing.allocator;
    var term = terminal.Terminal.init(allocator);
    defer term.deinit();

    const theme = &themes.themes.default;
    var panel = AgentPanel.init(allocator, &term, theme);
    defer panel.deinit();

    // Test exploration phase
    panel.updateStats(10, 5.0, 0.8);
    try std.testing.expectEqual(LearningPhase.exploration, panel.phase);

    // Test exploitation phase
    panel.updateStats(50, 10.0, 0.2);
    try std.testing.expectEqual(LearningPhase.exploitation, panel.phase);

    // Test converged phase
    panel.updateStats(100, 15.0, 0.01);
    try std.testing.expectEqual(LearningPhase.converged, panel.phase);
}

test "agent panel record decision" {
    const allocator = std.testing.allocator;
    var term = terminal.Terminal.init(allocator);
    defer term.deinit();

    const theme = &themes.themes.default;
    var panel = AgentPanel.init(allocator, &term, theme);
    defer panel.deinit();

    try panel.recordDecision(.{
        .timestamp = 1234567890,
        .workload_type = "MatMul",
        .selected_backend = "CUDA",
        .actual_time_ms = 50,
        .predicted_time_ms = 60,
        .reward = 0.5,
    });

    try std.testing.expectEqual(@as(usize, 1), panel.decision_history.items.len);
}

test "agent panel decision history limit" {
    const allocator = std.testing.allocator;
    var term = terminal.Terminal.init(allocator);
    defer term.deinit();

    const theme = &themes.themes.default;
    var panel = AgentPanel.init(allocator, &term, theme);
    defer panel.deinit();

    // Add more than 20 decisions
    for (0..25) |i| {
        try panel.recordDecision(.{
            .timestamp = @intCast(i),
            .workload_type = "MatMul",
            .selected_backend = "CUDA",
            .actual_time_ms = 50 + i,
            .predicted_time_ms = 60,
            .reward = 0.5,
        });
    }

    // Should be capped at 20
    try std.testing.expectEqual(@as(usize, 20), panel.decision_history.items.len);
}
