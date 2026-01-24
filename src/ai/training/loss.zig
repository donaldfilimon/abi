//! Loss functions for neural network training.
//!
//! Provides loss computation and gradient generation for language model training:
//! - Cross-entropy loss (for classification)
//! - Cross-entropy with label smoothing
//! - Perplexity computation
//! - Fused softmax + cross-entropy for numerical stability

const std = @import("std");

/// Cross-entropy loss for language modeling.
/// Computes: loss = -sum(target * log(softmax(logits)))
pub const CrossEntropyLoss = struct {
    allocator: std.mem.Allocator,
    /// Cached softmax probabilities for backward pass
    probs: ?[]f32,
    /// Cached target indices (sparse representation)
    targets: ?[]u32,
    /// Vocabulary size
    vocab_size: u32,
    /// Label smoothing factor (0 = no smoothing)
    label_smoothing: f32,
    /// Ignore index (e.g., for padding tokens, -1 = none)
    ignore_index: i32,
    /// Whether to reduce over batch
    reduction: ReductionType,
    /// Last computed loss value
    last_loss: f32,
    /// Number of valid tokens (non-ignored)
    valid_count: u32,

    pub const ReductionType = enum {
        none, // Return per-sample losses
        mean, // Average over all samples
        sum, // Sum over all samples
    };

    pub fn init(allocator: std.mem.Allocator, vocab_size: u32) CrossEntropyLoss {
        return .{
            .allocator = allocator,
            .probs = null,
            .targets = null,
            .vocab_size = vocab_size,
            .label_smoothing = 0.0,
            .ignore_index = -1,
            .reduction = .mean,
            .last_loss = 0,
            .valid_count = 0,
        };
    }

    pub fn deinit(self: *CrossEntropyLoss) void {
        if (self.probs) |p| {
            self.allocator.free(p);
        }
        if (self.targets) |t| {
            self.allocator.free(t);
        }
        self.* = undefined;
    }

    /// Set label smoothing factor.
    pub fn setLabelSmoothing(self: *CrossEntropyLoss, smoothing: f32) void {
        self.label_smoothing = @max(0.0, @min(1.0, smoothing));
    }

    /// Set ignore index for padding tokens.
    pub fn setIgnoreIndex(self: *CrossEntropyLoss, idx: i32) void {
        self.ignore_index = idx;
    }

    /// Compute cross-entropy loss.
    /// logits: [batch_size, vocab_size] - raw model outputs
    /// targets: [batch_size] - target token indices
    /// Returns the loss value.
    pub fn forward(
        self: *CrossEntropyLoss,
        logits: []const f32,
        targets: []const u32,
    ) !f32 {
        const batch_size = targets.len;

        // Allocate/reallocate cache if needed
        const total_size = batch_size * self.vocab_size;
        if (self.probs == null or self.probs.?.len != total_size) {
            if (self.probs) |p| {
                self.allocator.free(p);
            }
            self.probs = try self.allocator.alloc(f32, total_size);
        }
        if (self.targets == null or self.targets.?.len != batch_size) {
            if (self.targets) |t| {
                self.allocator.free(t);
            }
            self.targets = try self.allocator.alloc(u32, batch_size);
        }

        // Cache targets
        @memcpy(self.targets.?, targets);

        // Compute softmax for each sample and accumulate loss
        var total_loss: f32 = 0;
        var valid_count: u32 = 0;

        for (0..batch_size) |i| {
            const target = targets[i];

            // Skip ignored indices
            if (self.ignore_index >= 0 and target == @as(u32, @intCast(self.ignore_index))) {
                continue;
            }

            const logit_offset = i * self.vocab_size;
            const logit_slice = logits[logit_offset .. logit_offset + self.vocab_size];
            const prob_slice = self.probs.?[logit_offset .. logit_offset + self.vocab_size];

            // Compute numerically stable softmax
            softmaxStable(logit_slice, prob_slice);

            // Compute loss
            if (self.label_smoothing > 0) {
                // Label smoothing: mix one-hot with uniform
                const smooth = self.label_smoothing;
                const confidence = 1.0 - smooth;
                const uniform = smooth / @as(f32, @floatFromInt(self.vocab_size));

                var loss: f32 = 0;
                for (0..self.vocab_size) |v| {
                    var target_prob = uniform;
                    if (v == target) {
                        target_prob += confidence;
                    }
                    // Cross entropy: -target * log(pred)
                    const log_prob = safeLog(prob_slice[v]);
                    loss -= target_prob * log_prob;
                }
                total_loss += loss;
            } else {
                // Standard cross entropy
                const log_prob = safeLog(prob_slice[target]);
                total_loss -= log_prob;
            }

            valid_count += 1;
        }

        self.valid_count = valid_count;

        // Apply reduction
        self.last_loss = switch (self.reduction) {
            .none => total_loss,
            .mean => if (valid_count > 0) total_loss / @as(f32, @floatFromInt(valid_count)) else 0,
            .sum => total_loss,
        };

        return self.last_loss;
    }

    /// Compute gradients for cross-entropy loss.
    /// Returns gradient w.r.t. logits: [batch_size, vocab_size]
    /// For cross-entropy with softmax: d_logits = softmax - target
    pub fn backward(self: *CrossEntropyLoss, d_logits: []f32) void {
        if (self.probs == null or self.targets == null) return;

        const batch_size = self.targets.?.len;

        // Compute gradient: d_logits = probs - one_hot(target)
        for (0..batch_size) |i| {
            const target = self.targets.?[i];
            const offset = i * self.vocab_size;

            // Skip ignored indices
            if (self.ignore_index >= 0 and target == @as(u32, @intCast(self.ignore_index))) {
                @memset(d_logits[offset .. offset + self.vocab_size], 0);
                continue;
            }

            // Copy probs as gradient base
            const prob_slice = self.probs.?[offset .. offset + self.vocab_size];
            @memcpy(d_logits[offset .. offset + self.vocab_size], prob_slice);

            if (self.label_smoothing > 0) {
                // With label smoothing
                const smooth = self.label_smoothing;
                const confidence = 1.0 - smooth;
                const uniform = smooth / @as(f32, @floatFromInt(self.vocab_size));

                for (0..self.vocab_size) |v| {
                    var target_prob = uniform;
                    if (v == target) {
                        target_prob += confidence;
                    }
                    d_logits[offset + v] -= target_prob;
                }
            } else {
                // Standard gradient: softmax - one_hot
                d_logits[offset + target] -= 1.0;
            }

            // Scale by reduction factor
            if (self.reduction == .mean and self.valid_count > 0) {
                const scale = 1.0 / @as(f32, @floatFromInt(self.valid_count));
                for (d_logits[offset .. offset + self.vocab_size]) |*g| {
                    g.* *= scale;
                }
            }
        }
    }

    /// Get last computed loss.
    pub fn getLoss(self: *const CrossEntropyLoss) f32 {
        return self.last_loss;
    }

    /// Get perplexity (exp of loss).
    pub fn getPerplexity(self: *const CrossEntropyLoss) f32 {
        return @exp(self.last_loss);
    }
};

/// Numerically stable softmax.
fn softmaxStable(input: []const f32, output: []f32) void {
    // Find max for numerical stability
    var max_val: f32 = -std.math.inf(f32);
    for (input) |v| {
        if (v > max_val) max_val = v;
    }

    // Compute exp(x - max) and sum
    var sum: f32 = 0;
    for (input, 0..) |v, i| {
        output[i] = @exp(v - max_val);
        sum += output[i];
    }

    // Normalize
    const inv_sum = 1.0 / sum;
    for (output) |*v| {
        v.* *= inv_sum;
    }
}

/// Safe logarithm with minimum bound.
fn safeLog(x: f32) f32 {
    const min_val: f32 = 1e-10;
    return @log(@max(x, min_val));
}

/// Compute perplexity from loss.
pub fn perplexity(loss: f32) f32 {
    return @exp(loss);
}

/// Mean Squared Error loss.
pub const MSELoss = struct {
    reduction: CrossEntropyLoss.ReductionType,
    last_loss: f32,

    pub fn init() MSELoss {
        return .{
            .reduction = .mean,
            .last_loss = 0,
        };
    }

    /// Forward pass: MSE = mean((pred - target)^2)
    pub fn forward(self: *MSELoss, predictions: []const f32, targets: []const f32) f32 {
        var total: f32 = 0;
        for (predictions, targets) |p, t| {
            const diff = p - t;
            total += diff * diff;
        }

        self.last_loss = switch (self.reduction) {
            .none => total,
            .mean => total / @as(f32, @floatFromInt(predictions.len)),
            .sum => total,
        };

        return self.last_loss;
    }

    /// Backward pass: d_pred = 2 * (pred - target) / n
    pub fn backward(
        self: *const MSELoss,
        predictions: []const f32,
        targets: []const f32,
        d_predictions: []f32,
    ) void {
        const n = @as(f32, @floatFromInt(predictions.len));
        const scale: f32 = switch (self.reduction) {
            .none => 2.0,
            .mean => 2.0 / n,
            .sum => 2.0,
        };

        for (predictions, targets, 0..) |p, t, i| {
            d_predictions[i] = scale * (p - t);
        }
    }
};

/// Focal Loss for handling class imbalance.
/// FL(p) = -alpha * (1-p)^gamma * log(p)
pub const FocalLoss = struct {
    alpha: f32,
    gamma: f32,
    reduction: CrossEntropyLoss.ReductionType,

    pub fn init(alpha: f32, gamma: f32) FocalLoss {
        return .{
            .alpha = alpha,
            .gamma = gamma,
            .reduction = .mean,
        };
    }

    pub fn forward(
        self: *const FocalLoss,
        probs: []const f32,
        targets: []const u32,
        vocab_size: u32,
    ) f32 {
        const batch_size = targets.len;
        var total_loss: f32 = 0;

        for (0..batch_size) |i| {
            const target = targets[i];
            const prob_offset = i * vocab_size;
            const p = probs[prob_offset + target];

            // Focal loss: -alpha * (1-p)^gamma * log(p)
            const focal_weight = std.math.pow(f32, 1.0 - p, self.gamma);
            total_loss += -self.alpha * focal_weight * safeLog(p);
        }

        return switch (self.reduction) {
            .none => total_loss,
            .mean => total_loss / @as(f32, @floatFromInt(batch_size)),
            .sum => total_loss,
        };
    }
};

/// KL Divergence loss.
pub fn klDivergence(p: []const f32, q: []const f32) f32 {
    var total: f32 = 0;
    for (p, q) |p_val, q_val| {
        if (p_val > 0) {
            total += p_val * (safeLog(p_val) - safeLog(q_val));
        }
    }
    return total;
}

test "cross entropy loss basic" {
    const allocator = std.testing.allocator;

    var loss = CrossEntropyLoss.init(allocator, 4);
    defer loss.deinit();

    // Logits that should predict class 0
    const logits = [_]f32{
        10.0, 0.0, 0.0, 0.0, // Sample 0: strongly predicts class 0
        0.0, 10.0, 0.0, 0.0, // Sample 1: strongly predicts class 1
    };
    const targets = [_]u32{ 0, 1 };

    const loss_val = try loss.forward(&logits, &targets);

    // Loss should be very low since predictions match targets
    try std.testing.expect(loss_val < 0.1);
    try std.testing.expect(loss.getPerplexity() < 1.2);
}

test "cross entropy loss gradient" {
    const allocator = std.testing.allocator;

    var loss = CrossEntropyLoss.init(allocator, 3);
    defer loss.deinit();

    const logits = [_]f32{
        1.0, 2.0, 3.0, // Sample 0
    };
    const targets = [_]u32{2};

    _ = try loss.forward(&logits, &targets);

    var d_logits: [3]f32 = undefined;
    loss.backward(&d_logits);

    // Gradient should be softmax - one_hot
    // softmax([1,2,3]) ≈ [0.09, 0.24, 0.67]
    // target = 2, so gradient ≈ [0.09, 0.24, -0.33]
    try std.testing.expectApproxEqAbs(@as(f32, 0.09), d_logits[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.24), d_logits[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.33), d_logits[2], 0.01);
}

test "cross entropy with label smoothing" {
    const allocator = std.testing.allocator;

    var loss = CrossEntropyLoss.init(allocator, 3);
    defer loss.deinit();
    loss.setLabelSmoothing(0.1);

    const logits = [_]f32{
        1.0, 2.0, 3.0,
    };
    const targets = [_]u32{2};

    const loss_val = try loss.forward(&logits, &targets);

    // With label smoothing, loss should be slightly higher
    try std.testing.expect(loss_val > 0);
}

test "mse loss" {
    var loss = MSELoss.init();

    const predictions = [_]f32{ 1.0, 2.0, 3.0 };
    const targets = [_]f32{ 1.5, 2.0, 2.5 };

    const loss_val = loss.forward(&predictions, &targets);

    // MSE = ((0.5)^2 + 0 + (0.5)^2) / 3 = 0.5 / 3 ≈ 0.167
    try std.testing.expectApproxEqAbs(@as(f32, 0.167), loss_val, 0.01);

    var d_pred: [3]f32 = undefined;
    loss.backward(&predictions, &targets, &d_pred);

    // d_pred = 2 * (pred - target) / n
    try std.testing.expectApproxEqAbs(@as(f32, -0.333), d_pred[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), d_pred[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.333), d_pred[2], 0.01);
}

test "perplexity" {
    // Loss of 0 should give perplexity of 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), perplexity(0.0), 0.001);

    // Loss of 1 should give perplexity of e ≈ 2.718
    try std.testing.expectApproxEqAbs(@as(f32, 2.718), perplexity(1.0), 0.01);
}
