//! Training Brain Mapper
//!
//! Maps TrainingMetrics and GPU stats into brain DashboardData and
//! node activity for the 3D brain visualization. Translates training
//! pipeline state into visual elements:
//!
//! - Input layer (0-7): throughput activity
//! - Hidden layer 1 (8-23): gradient/LR dynamics
//! - Hidden layer 2 (24-47): loss landscape (chaotic vs converging)
//! - Output layer (48-63): accuracy/convergence glow

const std = @import("std");
const brain_panel = @import("brain_panel.zig");
const training_metrics = @import("training_metrics.zig");

/// GPU stats for visualization (mirrors training_bridge.GpuTrainingStats).
pub const GpuStats = struct {
    total_gpu_ops: u64 = 0,
    gpu_time_ns: u64 = 0,
    cpu_fallback_ops: u64 = 0,
    utilization: f32 = 0,
    backend_name: []const u8 = "none",
    gpu_available: bool = false,
};

pub const TrainingBrainMapper = struct {
    tick: u64,

    pub fn init() TrainingBrainMapper {
        return .{ .tick = 0 };
    }

    /// Map training metrics and GPU stats into DashboardData.
    /// Populates all 6 panels and node activity for the brain animation.
    pub fn updateDashboardData(
        self: *TrainingBrainMapper,
        data: *brain_panel.DashboardData,
        metrics: *const training_metrics.TrainingMetrics,
        gpu_stats: ?GpuStats,
    ) void {
        self.tick += 1;
        const t = @as(f32, @floatFromInt(self.tick)) * 0.1;

        data.is_training_mode = true;

        // -- Panel 1: Training Status (replaces WDBX Status) --
        data.current_epoch = metrics.current_epoch;
        data.total_epochs = metrics.total_epochs;
        data.current_step = metrics.current_step;
        data.vector_count = metrics.current_step;

        // -- Panel 2: Optimizer (replaces Learning Status) --
        data.learning_rate_current = metrics.learning_rate.latest();
        data.exploration_rate = @min(1.0, data.learning_rate_current * 10000.0);
        data.episode_count = metrics.current_epoch;

        // Determine training phase from loss trend
        const loss = metrics.train_loss.latest();
        if (loss > 0.5) {
            data.learning_phase = .exploration;
        } else if (loss > 0.1) {
            data.learning_phase = .exploitation;
        } else {
            data.learning_phase = .converged;
        }

        // -- Panel 3: Throughput (tokens/sec sparkline) --
        const elapsed = metrics.elapsedSeconds();
        data.tokens_per_sec = if (elapsed > 0 and metrics.current_step > 0)
            @as(f32, @floatFromInt(metrics.current_step)) / @as(f32, @floatFromInt(elapsed))
        else
            0;
        data.insert_rate.push(data.tokens_per_sec);
        data.search_rate.push(data.tokens_per_sec * 0.8);

        // -- Panel 4: Loss / Accuracy (replaces Reward History) --
        data.train_loss = loss;
        data.loss_history.push(loss);

        const val = metrics.val_loss.latest();
        data.train_accuracy = if (val > 0.001)
            @min(1.0, 1.0 / (1.0 + val))
        else if (loss > 0.001)
            @min(1.0, 1.0 / (1.0 + loss))
        else
            0;
        data.reward_history.push(data.train_accuracy);

        // -- Panel 5: Perplexity (replaces Similarity) --
        data.perplexity = if (loss > 0) @exp(@min(loss, 10.0)) else 1.0;
        data.similarity_scores.push(@min(1.0, 1.0 / (1.0 + data.perplexity * 0.01)));

        // -- Panel 6: GPU Utilization (replaces Attention bars) --
        if (gpu_stats) |stats| {
            data.gpu_utilization = stats.utilization;
            const total_ops = stats.total_gpu_ops + stats.cpu_fallback_ops;
            const gpu_ratio = if (total_ops > 0)
                @as(f32, @floatFromInt(stats.total_gpu_ops)) / @as(f32, @floatFromInt(total_ops))
            else
                0;

            data.attention_weights[0] = gpu_ratio;
            data.attention_weights[1] = stats.utilization;
            data.attention_weights[2] = if (stats.gpu_time_ns > 0)
                @min(1.0, @as(f32, @floatFromInt(stats.gpu_time_ns)) / 1e9)
            else
                0;
            data.attention_weights[3] = if (stats.cpu_fallback_ops > 0)
                @min(1.0, @as(f32, @floatFromInt(stats.cpu_fallback_ops)) / 100.0)
            else
                0;
            for (4..8) |i| {
                data.attention_weights[i] = stats.utilization *
                    (0.5 + 0.5 * std.math.sin(t + @as(f32, @floatFromInt(i))));
            }

            const name = stats.backend_name;
            const copy_len = @min(name.len, data.gpu_backend_name.len);
            @memcpy(data.gpu_backend_name[0..copy_len], name[0..copy_len]);
        } else {
            for (&data.attention_weights) |*w| w.* = 0.1;
        }

        // -- Node Activity for Brain Animation --
        updateNodeActivity(data, t);
    }

    fn updateNodeActivity(data: *brain_panel.DashboardData, t: f32) void {
        // Input layer (0-7): Data throughput (tokens/sec normalized)
        const throughput = @min(1.0, data.tokens_per_sec / 1000.0);
        for (0..8) |i| {
            const phase = @as(f32, @floatFromInt(i)) * 0.5;
            data.node_activity[i] = @min(1.0, throughput * (0.7 + 0.3 * std.math.sin(t * 0.2 + phase)));
        }

        // Hidden layer 1 (8-23): Gradient norm / LR activity
        const lr_norm = @min(1.0, data.learning_rate_current * 10000.0);
        for (8..24) |i| {
            const phase = @as(f32, @floatFromInt(i)) * 0.3;
            data.node_activity[i] = @min(1.0, lr_norm * (0.6 + 0.4 * std.math.sin(t * 0.15 + phase)));
        }

        // Hidden layer 2 (24-47): Loss dynamics
        // High loss = chaotic flickering, low loss = settling down
        const loss_activity = @min(1.0, data.train_loss * 2.0);
        for (24..48) |i| {
            const phase = @as(f32, @floatFromInt(i)) * 0.25;
            const chaos = if (data.train_loss > 0.5)
                0.4 * std.math.sin(t * 0.5 + phase)
            else
                0.1 * std.math.sin(t * 0.1 + phase);
            data.node_activity[i] = @min(1.0, @max(0.0, loss_activity * 0.7 + chaos));
        }

        // Output layer (48-63): Accuracy / convergence glow
        const acc = data.train_accuracy;
        for (48..64) |i| {
            const phase = @as(f32, @floatFromInt(i)) * 0.4;
            const base = if (acc > 0.8)
                acc // Steady glow when converging
            else
                acc * (0.5 + 0.5 * std.math.sin(t * 0.3 + phase)); // Flickering
            data.node_activity[i] = @min(1.0, base);
        }
    }
};

// =============================================================================
// Tests
// =============================================================================

test "TrainingBrainMapper init" {
    const mapper = TrainingBrainMapper.init();
    try std.testing.expectEqual(@as(u64, 0), mapper.tick);
}

test "TrainingBrainMapper updates dashboard data" {
    var mapper = TrainingBrainMapper.init();
    var data = brain_panel.DashboardData.init();
    var metrics = training_metrics.TrainingMetrics{};

    // Simulate a few training events
    metrics.update(.{ .event_type = .scalar, .tag = "loss/train", .value = 0.5, .step = 100, .timestamp = 1000 });
    metrics.update(.{ .event_type = .progress, .epoch = 2, .total_epochs = 10, .step = 100, .total_steps = 500, .timestamp = 1001 });

    mapper.updateDashboardData(&data, &metrics, null);

    try std.testing.expect(data.is_training_mode);
    try std.testing.expectEqual(@as(u32, 2), data.current_epoch);
    try std.testing.expectEqual(@as(u32, 10), data.total_epochs);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data.train_loss, 0.01);
}

test "TrainingBrainMapper node activity" {
    var mapper = TrainingBrainMapper.init();
    var data = brain_panel.DashboardData.init();
    var metrics = training_metrics.TrainingMetrics{};

    metrics.update(.{ .event_type = .scalar, .tag = "loss/train", .value = 0.8, .step = 50, .timestamp = 1000 });

    mapper.updateDashboardData(&data, &metrics, null);

    // High loss should produce active hidden layer
    var has_activity = false;
    for (24..48) |i| {
        if (data.node_activity[i] > 0.01) has_activity = true;
    }
    try std.testing.expect(has_activity);
}
