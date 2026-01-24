//! Token sampling strategies for text generation.
//!
//! Implements various sampling methods:
//! - Greedy (argmax)
//! - Temperature scaling
//! - Top-k filtering
//! - Top-p (nucleus) sampling
//! - Repetition penalty

const std = @import("std");
const activations = @import("../ops/activations.zig");

/// Sampler configuration.
pub const SamplerConfig = struct {
    /// Temperature for sampling (0 = greedy, 1 = default)
    temperature: f32 = 1.0,
    /// Top-k filtering (0 = disabled)
    top_k: u32 = 40,
    /// Top-p nucleus sampling threshold
    top_p: f32 = 0.9,
    /// Tail-free sampling z-parameter (1.0 = disabled)
    tfs_z: f32 = 1.0,
    /// Mirostat mode (0 = disabled, 1 = v1, 2 = v2)
    mirostat: u8 = 0,
    /// Mirostat target entropy (tau)
    mirostat_tau: f32 = 5.0,
    /// Mirostat learning rate (eta)
    mirostat_eta: f32 = 0.1,
    /// Repetition penalty (1.0 = disabled)
    repetition_penalty: f32 = 1.1,
    /// Number of recent tokens to consider for repetition penalty
    repetition_window: u32 = 64,
    /// Random seed (0 = use system time)
    seed: u64 = 0,
};

/// Token sampler with various strategies.
pub const Sampler = struct {
    config: SamplerConfig,
    rng: std.Random.DefaultPrng,
    recent_tokens: std.ArrayListUnmanaged(u32),
    allocator: std.mem.Allocator,
    /// Mirostat mu (dynamic surprise state)
    mirostat_mu: f32,

    pub fn init(allocator: std.mem.Allocator, config: SamplerConfig) Sampler {
        // Use a simple pseudo-random seed if not specified
        const seed = if (config.seed == 0) @as(u64, 0x853c49e6748fea9b) else config.seed;

        return .{
            .config = config,
            .rng = std.Random.DefaultPrng.init(seed),
            .recent_tokens = std.ArrayListUnmanaged(u32).empty,
            .allocator = allocator,
            .mirostat_mu = config.mirostat_tau * 2.0, // Initialize mu to 2*tau
        };
    }

    pub fn deinit(self: *Sampler) void {
        self.recent_tokens.deinit(self.allocator);
        self.* = undefined;
    }

    /// Sample a token from logits.
    pub fn sample(self: *Sampler, logits: []f32) u32 {
        // Apply repetition penalty
        if (self.config.repetition_penalty != 1.0) {
            self.applyRepetitionPenalty(logits);
        }

        // Greedy sampling
        if (self.config.temperature <= 0) {
            const token = self.argmax(logits);
            self.trackToken(token);
            return token;
        }

        // Mirostat sampling (bypasses other methods)
        if (self.config.mirostat > 0) {
            const token = self.sampleMirostat(logits);
            self.trackToken(token);
            return token;
        }

        // Temperature scaling
        if (self.config.temperature != 1.0) {
            self.applyTemperature(logits);
        }

        // Apply tail-free sampling before softmax
        if (self.config.tfs_z < 1.0) {
            self.applyTailFree(logits);
        }

        // Convert to probabilities
        activations.softmaxInPlace(logits);

        // Apply top-k and top-p filtering
        var token: u32 = undefined;
        if (self.config.top_k > 0 or self.config.top_p < 1.0) {
            token = self.sampleTopKTopP(logits);
        } else {
            token = self.sampleCategorical(logits);
        }

        // Track for repetition penalty
        self.trackToken(token);

        return token;
    }

    fn applyTemperature(self: *Sampler, logits: []f32) void {
        const inv_temp = 1.0 / self.config.temperature;
        for (logits) |*l| {
            l.* *= inv_temp;
        }
    }

    fn applyRepetitionPenalty(self: *Sampler, logits: []f32) void {
        for (self.recent_tokens.items) |token| {
            if (token < logits.len) {
                if (logits[token] > 0) {
                    logits[token] /= self.config.repetition_penalty;
                } else {
                    logits[token] *= self.config.repetition_penalty;
                }
            }
        }
    }

    fn argmax(_: *Sampler, logits: []const f32) u32 {
        var max_idx: u32 = 0;
        var max_val: f32 = logits[0];
        for (logits[1..], 1..) |v, i| {
            if (v > max_val) {
                max_val = v;
                max_idx = @intCast(i);
            }
        }
        return max_idx;
    }

    fn sampleCategorical(self: *Sampler, probs: []const f32) u32 {
        const r = self.rng.random().float(f32);
        var cum: f32 = 0;
        for (probs, 0..) |p, i| {
            cum += p;
            if (r < cum) {
                return @intCast(i);
            }
        }
        return @intCast(probs.len - 1);
    }

    fn sampleTopKTopP(self: *Sampler, probs: []f32) u32 {
        // Create indices sorted by probability
        var indices = self.allocator.alloc(u32, probs.len) catch return self.argmax(probs);
        defer self.allocator.free(indices);

        for (0..probs.len) |i| {
            indices[i] = @intCast(i);
        }

        // Sort by probability (descending)
        std.mem.sort(u32, indices, probs, struct {
            fn lessThan(p: []f32, a: u32, b: u32) bool {
                return p[a] > p[b];
            }
        }.lessThan);

        // Apply top-k
        var k = probs.len;
        if (self.config.top_k > 0 and self.config.top_k < probs.len) {
            k = self.config.top_k;
        }

        // Apply top-p
        var cum_prob: f32 = 0;
        var cutoff: usize = k;
        for (0..k) |i| {
            cum_prob += probs[indices[i]];
            if (cum_prob >= self.config.top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Renormalize
        var total: f32 = 0;
        for (0..cutoff) |i| {
            total += probs[indices[i]];
        }

        // Sample from filtered distribution
        const r = self.rng.random().float(f32) * total;
        cum_prob = 0;
        for (0..cutoff) |i| {
            cum_prob += probs[indices[i]];
            if (r < cum_prob) {
                return indices[i];
            }
        }

        return indices[0];
    }

    /// Apply tail-free sampling (TFS) to logits.
    /// TFS looks at the second derivative of the sorted probability distribution
    /// to identify the "tail" and removes low-probability tokens.
    fn applyTailFree(self: *Sampler, logits: []f32) void {
        if (logits.len < 2) return;

        // Create sorted indices
        const indices = self.allocator.alloc(u32, logits.len) catch return;
        defer self.allocator.free(indices);

        for (0..logits.len) |i| {
            indices[i] = @intCast(i);
        }

        // Sort by logit value (descending)
        std.mem.sort(u32, indices, logits, struct {
            fn lessThan(l: []f32, a: u32, b: u32) bool {
                return l[a] > l[b];
            }
        }.lessThan);

        // Convert to probabilities temporarily for TFS calculation
        const probs = self.allocator.alloc(f32, logits.len) catch return;
        defer self.allocator.free(probs);

        // Softmax for probability calculation
        const max_logit: f32 = logits[indices[0]];
        var sum_exp: f32 = 0;
        for (indices) |idx| {
            const exp_val = @exp(logits[idx] - max_logit);
            probs[idx] = exp_val;
            sum_exp += exp_val;
        }
        for (probs) |*p| {
            p.* /= sum_exp;
        }

        // Calculate first derivatives
        const first_derivs = self.allocator.alloc(f32, logits.len - 1) catch return;
        defer self.allocator.free(first_derivs);

        for (0..logits.len - 1) |i| {
            first_derivs[i] = probs[indices[i]] - probs[indices[i + 1]];
        }

        // Calculate second derivatives
        const second_derivs = self.allocator.alloc(f32, logits.len - 2) catch return;
        defer self.allocator.free(second_derivs);

        for (0..logits.len - 2) |i| {
            second_derivs[i] = first_derivs[i] - first_derivs[i + 1];
        }

        // Normalize second derivatives (absolute values)
        var sum_abs: f32 = 0;
        for (second_derivs) |d| {
            sum_abs += @abs(d);
        }

        if (sum_abs > 0) {
            for (second_derivs) |*d| {
                d.* = @abs(d.*) / sum_abs;
            }
        }

        // Find cutoff where cumulative sum exceeds z threshold
        var cum_sum: f32 = 0;
        var cutoff: usize = logits.len;
        for (second_derivs, 0..) |d, i| {
            cum_sum += d;
            if (cum_sum > self.config.tfs_z) {
                cutoff = i + 2; // +2 because second derivs are offset by 2
                break;
            }
        }

        // Zero out tokens beyond cutoff
        for (cutoff..logits.len) |i| {
            logits[indices[i]] = -std.math.inf(f32);
        }
    }

    /// Sample using Mirostat algorithm for perplexity control.
    /// Mirostat dynamically adjusts the sampling to maintain target entropy.
    fn sampleMirostat(self: *Sampler, logits: []f32) u32 {
        // Temperature scaling
        if (self.config.temperature != 1.0 and self.config.temperature > 0) {
            const inv_temp = 1.0 / self.config.temperature;
            for (logits) |*l| {
                l.* *= inv_temp;
            }
        }

        // Create sorted indices
        const indices = self.allocator.alloc(u32, logits.len) catch return self.argmax(logits);
        defer self.allocator.free(indices);

        for (0..logits.len) |i| {
            indices[i] = @intCast(i);
        }

        // Sort by logit value (descending)
        std.mem.sort(u32, indices, logits, struct {
            fn lessThan(l: []f32, a: u32, b: u32) bool {
                return l[a] > l[b];
            }
        }.lessThan);

        // Convert to probabilities
        const probs = self.allocator.alloc(f32, logits.len) catch return self.argmax(logits);
        defer self.allocator.free(probs);

        const max_logit: f32 = logits[indices[0]];
        var sum_exp: f32 = 0;
        for (indices) |idx| {
            const exp_val = @exp(logits[idx] - max_logit);
            probs[idx] = exp_val;
            sum_exp += exp_val;
        }
        for (probs) |*p| {
            p.* /= sum_exp;
        }

        if (self.config.mirostat == 1) {
            // Mirostat v1: Uses target surprise (s) and learning rate
            return self.sampleMirostatV1(indices, probs);
        } else {
            // Mirostat v2: Simpler truncation-based approach
            return self.sampleMirostatV2(indices, probs);
        }
    }

    fn sampleMirostatV1(self: *Sampler, indices: []const u32, probs: []f32) u32 {
        const n = probs.len;
        const tau = self.config.mirostat_tau;
        const eta = self.config.mirostat_eta;

        // Calculate target k based on current mu
        // k = (r^mu - 1) / (r - 1) where r is vocab size / sum of top probs
        const mu = self.mirostat_mu;
        const n_float: f32 = @floatFromInt(n);
        var k: usize = @intFromFloat(@max(1.0, @min(n_float, @ceil(std.math.pow(f32, 2.0, mu)))));
        k = @min(k, n);

        // Truncate to top-k and renormalize
        var sum: f32 = 0;
        for (0..k) |i| {
            sum += probs[indices[i]];
        }

        // Sample from truncated distribution
        const r = self.rng.random().float(f32) * sum;
        var cum: f32 = 0;
        var selected_idx: u32 = indices[0];
        var selected_prob: f32 = probs[indices[0]];

        for (0..k) |i| {
            cum += probs[indices[i]];
            if (r < cum) {
                selected_idx = indices[i];
                selected_prob = probs[indices[i]] / sum;
                break;
            }
        }

        // Calculate surprise and update mu
        const surprise = -@log2(@max(selected_prob, 1e-10));
        const error_val = surprise - tau;
        self.mirostat_mu = @max(0.0, mu - eta * error_val);

        return selected_idx;
    }

    fn sampleMirostatV2(self: *Sampler, indices: []const u32, probs: []f32) u32 {
        const tau = self.config.mirostat_tau;
        const eta = self.config.mirostat_eta;
        const mu = self.mirostat_mu;

        // Find truncation point where surprise exceeds mu
        var truncation: usize = probs.len;
        for (indices, 0..) |idx, i| {
            const p = probs[idx];
            if (p > 0) {
                const surprise = -@log2(p);
                if (surprise > mu) {
                    truncation = @max(1, i);
                    break;
                }
            }
        }

        // Renormalize truncated distribution
        var sum: f32 = 0;
        for (0..truncation) |i| {
            sum += probs[indices[i]];
        }

        // Sample from truncated distribution
        const r = self.rng.random().float(f32) * sum;
        var cum: f32 = 0;
        var selected_idx: u32 = indices[0];
        var selected_prob: f32 = probs[indices[0]];

        for (0..truncation) |i| {
            cum += probs[indices[i]];
            if (r < cum) {
                selected_idx = indices[i];
                selected_prob = probs[indices[i]] / sum;
                break;
            }
        }

        // Calculate surprise and update mu
        const surprise = -@log2(@max(selected_prob, 1e-10));
        const error_val = surprise - tau;
        self.mirostat_mu = mu - eta * error_val;

        return selected_idx;
    }

    fn trackToken(self: *Sampler, token: u32) void {
        self.recent_tokens.append(self.allocator, token) catch return;

        // Keep only recent window
        if (self.recent_tokens.items.len > self.config.repetition_window) {
            _ = self.recent_tokens.orderedRemove(0);
        }
    }

    /// Reset sampler state.
    pub fn reset(self: *Sampler) void {
        self.recent_tokens.clearRetainingCapacity();
        self.mirostat_mu = self.config.mirostat_tau * 2.0;
    }
};

/// Top-k and Top-p filtering helper.
pub const TopKTopP = struct {
    /// Apply top-k filtering to logits.
    pub fn topK(logits: []f32, k: u32, allocator: std.mem.Allocator) void {
        if (k == 0 or k >= logits.len) return;

        // Find k-th largest value
        const indices = allocator.alloc(u32, logits.len) catch return;
        defer allocator.free(indices);

        for (0..logits.len) |i| {
            indices[i] = @intCast(i);
        }

        // Partial sort to find k-th element
        std.mem.sort(u32, indices, logits, struct {
            fn lessThan(l: []f32, a: u32, b: u32) bool {
                return l[a] > l[b];
            }
        }.lessThan);

        const threshold = logits[indices[k - 1]];

        // Zero out values below threshold
        for (logits) |*l| {
            if (l.* < threshold) {
                l.* = -std.math.inf(f32);
            }
        }
    }

    /// Apply top-p (nucleus) filtering to probabilities.
    pub fn topP(probs: []f32, p: f32, allocator: std.mem.Allocator) void {
        if (p >= 1.0) return;

        const indices = allocator.alloc(u32, probs.len) catch return;
        defer allocator.free(indices);

        for (0..probs.len) |i| {
            indices[i] = @intCast(i);
        }

        std.mem.sort(u32, indices, probs, struct {
            fn lessThan(prob: []f32, a: u32, b: u32) bool {
                return prob[a] > prob[b];
            }
        }.lessThan);

        var cum: f32 = 0;
        for (indices) |idx| {
            cum += probs[idx];
            if (cum > p) {
                probs[idx] = 0;
            }
        }
    }
};

test "sampler greedy" {
    const allocator = std.testing.allocator;

    var sampler_inst = Sampler.init(allocator, .{ .temperature = 0 });
    defer sampler_inst.deinit();

    var logits = [_]f32{ 1.0, 3.0, 2.0, 0.5 };
    const token = sampler_inst.sample(&logits);

    try std.testing.expectEqual(@as(u32, 1), token); // Index of max value (3.0)
}

test "sampler temperature" {
    const allocator = std.testing.allocator;

    var sampler_inst = Sampler.init(allocator, .{
        .temperature = 1.0,
        .top_k = 0,
        .top_p = 1.0,
        .seed = 42,
    });
    defer sampler_inst.deinit();

    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // Sample multiple times and verify distribution makes sense
    var counts = [_]u32{ 0, 0, 0, 0 };
    for (0..100) |_| {
        // Reset logits each time (softmax modifies in place)
        logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const token = sampler_inst.sample(&logits);
        counts[token] += 1;
    }

    // Higher logits should be sampled more often
    try std.testing.expect(counts[3] > counts[0]);
}

test "repetition penalty" {
    const allocator = std.testing.allocator;

    var sampler_inst = Sampler.init(allocator, .{
        .temperature = 0,
        .repetition_penalty = 2.0,
    });
    defer sampler_inst.deinit();

    // First sample should be token 1 (highest logit)
    var logits = [_]f32{ 1.0, 3.0, 2.0, 2.9 };
    var token = sampler_inst.sample(&logits);
    try std.testing.expectEqual(@as(u32, 1), token);

    // After sampling token 1, its logit should be penalized
    logits = [_]f32{ 1.0, 3.0, 2.0, 2.9 };
    token = sampler_inst.sample(&logits);
    // Token 1's logit (3.0) should now be 1.5, so token 3 (2.9) should be chosen
    try std.testing.expectEqual(@as(u32, 3), token);
}

test "tail-free sampling" {
    const allocator = std.testing.allocator;

    var sampler_inst = Sampler.init(allocator, .{
        .temperature = 1.0,
        .tfs_z = 0.5, // Aggressive tail filtering
        .top_k = 0,
        .top_p = 1.0,
        .seed = 42,
    });
    defer sampler_inst.deinit();

    // With TFS z=0.5, tokens in the tail should be filtered out
    var logits = [_]f32{ 5.0, 4.0, 3.0, 0.1, 0.01, 0.001 };
    const token = sampler_inst.sample(&logits);

    // Should sample from top tokens, not the tail
    try std.testing.expect(token <= 2);
}

test "mirostat v2 sampling" {
    const allocator = std.testing.allocator;

    var sampler_inst = Sampler.init(allocator, .{
        .mirostat = 2,
        .mirostat_tau = 5.0,
        .mirostat_eta = 0.1,
        .seed = 42,
    });
    defer sampler_inst.deinit();

    // Mirostat should produce valid tokens
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const token = sampler_inst.sample(&logits);

    // Token should be valid
    try std.testing.expect(token < 5);

    // Mu should have been updated
    try std.testing.expect(sampler_inst.mirostat_mu != sampler_inst.config.mirostat_tau * 2.0);
}

test "mirostat v1 sampling" {
    const allocator = std.testing.allocator;

    var sampler_inst = Sampler.init(allocator, .{
        .mirostat = 1,
        .mirostat_tau = 3.0,
        .mirostat_eta = 0.2,
        .seed = 123,
    });
    defer sampler_inst.deinit();

    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const token = sampler_inst.sample(&logits);

    // Token should be valid
    try std.testing.expect(token < 5);
}

test "sampler reset clears mirostat state" {
    const allocator = std.testing.allocator;

    var sampler_inst = Sampler.init(allocator, .{
        .mirostat = 2,
        .mirostat_tau = 5.0,
    });
    defer sampler_inst.deinit();

    // Sample to modify mu
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    _ = sampler_inst.sample(&logits);

    // Reset should restore mu to initial value
    sampler_inst.reset();
    try std.testing.expectApproxEqAbs(
        @as(f32, 10.0), // 2 * tau
        sampler_inst.mirostat_mu,
        0.001,
    );
}
