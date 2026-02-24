//! Mixed precision training utilities.
//!
//! Provides FP16 forward / FP32 gradient training with:
//! - Loss scaling to prevent underflow
//! - Dynamic loss scale adjustment
//! - FP16/FP32 conversions
//! - Gradient scaling and unscaling

const std = @import("std");

/// Mixed precision training configuration.
pub const MixedPrecisionConfig = struct {
    /// Enable mixed precision
    enabled: bool = false,
    /// Initial loss scale (power of 2)
    initial_scale: f32 = 65536.0,
    /// Growth factor when scaling up
    growth_factor: f32 = 2.0,
    /// Backoff factor when overflow detected
    backoff_factor: f32 = 0.5,
    /// Minimum loss scale
    min_scale: f32 = 1.0,
    /// Maximum loss scale
    max_scale: f32 = 2147483648.0,
    /// Steps before attempting to increase scale
    growth_interval: u32 = 2000,
};

/// Loss scaler for mixed precision training.
/// Handles dynamic loss scaling to prevent gradient underflow.
pub const LossScaler = struct {
    allocator: std.mem.Allocator,
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    min_scale: f32,
    max_scale: f32,
    growth_interval: u32,
    /// Steps since last overflow
    steps_since_overflow: u32,
    /// Total overflow events
    overflow_count: u64,
    /// Whether gradients are currently scaled
    gradients_scaled: bool,

    pub fn init(allocator: std.mem.Allocator, config: MixedPrecisionConfig) LossScaler {
        return .{
            .allocator = allocator,
            .scale = config.initial_scale,
            .growth_factor = config.growth_factor,
            .backoff_factor = config.backoff_factor,
            .min_scale = config.min_scale,
            .max_scale = config.max_scale,
            .growth_interval = config.growth_interval,
            .steps_since_overflow = 0,
            .overflow_count = 0,
            .gradients_scaled = false,
        };
    }

    pub fn deinit(self: *LossScaler) void {
        self.* = undefined;
    }

    /// Get current loss scale.
    pub fn getScale(self: *const LossScaler) f32 {
        return self.scale;
    }

    /// Scale loss before backward pass.
    pub fn scaleLoss(self: *LossScaler, loss: f32) f32 {
        self.gradients_scaled = true;
        return loss * self.scale;
    }

    /// Unscale gradients after backward pass.
    /// Returns true if gradients are valid (no overflow/underflow).
    pub fn unscaleGradients(self: *LossScaler, gradients: []f32) bool {
        if (!self.gradients_scaled) return true;

        const inv_scale = 1.0 / self.scale;
        var has_overflow = false;

        for (gradients) |*g| {
            const scaled_g = g.* * inv_scale;

            // Check for overflow/inf/nan
            if (!std.math.isFinite(scaled_g)) {
                has_overflow = true;
                break;
            }
            g.* = scaled_g;
        }

        self.gradients_scaled = false;

        if (has_overflow) {
            // Zero out gradients on overflow
            @memset(gradients, 0);
        }

        return !has_overflow;
    }

    /// Update scale based on gradient validity.
    /// Call after unscaleGradients.
    pub fn update(self: *LossScaler, gradients_valid: bool) void {
        if (!gradients_valid) {
            // Overflow detected - reduce scale
            self.scale = @max(self.scale * self.backoff_factor, self.min_scale);
            self.steps_since_overflow = 0;
            self.overflow_count += 1;
        } else {
            // No overflow - maybe increase scale
            self.steps_since_overflow += 1;
            if (self.steps_since_overflow >= self.growth_interval) {
                self.scale = @min(self.scale * self.growth_factor, self.max_scale);
                self.steps_since_overflow = 0;
            }
        }
    }

    /// Get statistics.
    pub fn getStats(self: *const LossScaler) ScalerStats {
        return .{
            .current_scale = self.scale,
            .overflow_count = self.overflow_count,
            .steps_since_overflow = self.steps_since_overflow,
        };
    }

    pub const ScalerStats = struct {
        current_scale: f32,
        overflow_count: u64,
        steps_since_overflow: u32,
    };
};

/// Convert FP32 tensor to FP16.
pub fn fp32ToFp16(src: []const f32, dst: []f16) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |s, *d| {
        d.* = @floatCast(s);
    }
}

/// Convert FP16 tensor to FP32.
pub fn fp16ToFp32(src: []const f16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |s, *d| {
        d.* = @floatCast(s);
    }
}

/// FP16 weight copy with master weights in FP32.
/// Used for training where weights are stored in FP16 for forward
/// but master copy kept in FP32 for optimizer updates.
pub const MasterWeights = struct {
    allocator: std.mem.Allocator,
    /// Master weights in FP32 (used by optimizer)
    master: []f32,
    /// Working weights in FP16 (used by forward pass)
    working: []f16,

    pub fn init(allocator: std.mem.Allocator, size: usize) !MasterWeights {
        const master = try allocator.alloc(f32, size);
        errdefer allocator.free(master);
        const working = try allocator.alloc(f16, size);

        return .{
            .allocator = allocator,
            .master = master,
            .working = working,
        };
    }

    pub fn deinit(self: *MasterWeights) void {
        self.allocator.free(self.working);
        self.allocator.free(self.master);
        self.* = undefined;
    }

    /// Copy master weights to working (FP32 -> FP16).
    pub fn copyToWorking(self: *MasterWeights) void {
        fp32ToFp16(self.master, self.working);
    }

    /// Update master weights (optimizer step was done on master).
    pub fn afterOptimizerStep(self: *MasterWeights) void {
        // Master already updated by optimizer
        // Just sync to working
        self.copyToWorking();
    }

    /// Initialize from existing FP32 weights.
    pub fn initializeFrom(self: *MasterWeights, weights: []const f32) void {
        @memcpy(self.master, weights);
        self.copyToWorking();
    }
};

/// Mixed precision context for a training step.
pub const MixedPrecisionContext = struct {
    allocator: std.mem.Allocator,
    config: MixedPrecisionConfig,
    scaler: LossScaler,
    enabled: bool,

    pub fn init(allocator: std.mem.Allocator, config: MixedPrecisionConfig) MixedPrecisionContext {
        return .{
            .allocator = allocator,
            .config = config,
            .scaler = LossScaler.init(allocator, config),
            .enabled = config.enabled,
        };
    }

    pub fn deinit(self: *MixedPrecisionContext) void {
        self.scaler.deinit();
        self.* = undefined;
    }

    /// Prepare loss for backward pass.
    pub fn prepareLoss(self: *MixedPrecisionContext, loss: f32) f32 {
        if (!self.enabled) return loss;
        return self.scaler.scaleLoss(loss);
    }

    /// Process gradients after backward pass.
    /// Returns true if gradients are valid and optimizer step should proceed.
    pub fn processGradients(self: *MixedPrecisionContext, gradients: []f32) bool {
        if (!self.enabled) return true;

        const valid = self.scaler.unscaleGradients(gradients);
        self.scaler.update(valid);
        return valid;
    }

    /// Check if step should be skipped (due to overflow).
    pub fn shouldSkipStep(self: *const MixedPrecisionContext, gradients_valid: bool) bool {
        if (!self.enabled) return false;
        return !gradients_valid;
    }

    /// Get scaler statistics.
    pub fn getStats(self: *const MixedPrecisionContext) LossScaler.ScalerStats {
        return self.scaler.getStats();
    }
};

/// Check if tensor contains any non-finite values.
pub fn hasNonFinite(data: []const f32) bool {
    for (data) |v| {
        if (!std.math.isFinite(v)) return true;
    }
    return false;
}

/// Clamp tensor values to prevent overflow.
pub fn clampToFinite(data: []f32, max_abs: f32) void {
    for (data) |*v| {
        if (!std.math.isFinite(v.*)) {
            v.* = 0;
        } else if (v.* > max_abs) {
            v.* = max_abs;
        } else if (v.* < -max_abs) {
            v.* = -max_abs;
        }
    }
}

test "loss scaler basic" {
    var scaler = LossScaler.init(std.testing.allocator, .{
        .enabled = true,
        .initial_scale = 1024.0,
    });
    defer scaler.deinit();

    // Scale loss
    const scaled = scaler.scaleLoss(1.0);
    try std.testing.expectEqual(@as(f32, 1024.0), scaled);

    // Unscale gradients
    var grads = [_]f32{ 1024.0, 2048.0, -512.0 };
    const valid = scaler.unscaleGradients(&grads);
    try std.testing.expect(valid);
    try std.testing.expectEqual(@as(f32, 1.0), grads[0]);
    try std.testing.expectEqual(@as(f32, 2.0), grads[1]);
    try std.testing.expectEqual(@as(f32, -0.5), grads[2]);
}

test "loss scaler overflow handling" {
    var scaler = LossScaler.init(std.testing.allocator, .{
        .enabled = true,
        .initial_scale = 1024.0,
        .backoff_factor = 0.5,
    });
    defer scaler.deinit();

    _ = scaler.scaleLoss(1.0);

    // Create overflow
    var grads = [_]f32{ std.math.inf(f32), 1.0, 2.0 };
    const valid = scaler.unscaleGradients(&grads);
    try std.testing.expect(!valid);

    // Gradients should be zeroed
    for (grads) |g| {
        try std.testing.expectEqual(@as(f32, 0), g);
    }

    // Scale should be reduced
    scaler.update(valid);
    try std.testing.expectEqual(@as(f32, 512.0), scaler.getScale());
}

test "fp16 conversion" {
    var src = [_]f32{ 1.0, -2.5, 0.0, 100.0 };
    var fp16 = [_]f16{ 0, 0, 0, 0 };
    var dst = [_]f32{ 0, 0, 0, 0 };

    fp32ToFp16(&src, &fp16);
    fp16ToFp32(&fp16, &dst);

    // Values should roundtrip (with FP16 precision loss)
    for (src, dst) |s, d| {
        try std.testing.expectApproxEqAbs(s, d, 0.01);
    }
}

test "master weights" {
    const allocator = std.testing.allocator;

    var mw = try MasterWeights.init(allocator, 4);
    defer mw.deinit();

    // Initialize from FP32
    const weights = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    mw.initializeFrom(&weights);

    // Check working copy
    for (mw.working, weights) |w, expected| {
        try std.testing.expectApproxEqAbs(expected, @as(f32, @floatCast(w)), 0.01);
    }

    // Simulate optimizer update on master
    mw.master[0] = 1.5;
    mw.afterOptimizerStep();

    // Working should be updated
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), @as(f32, @floatCast(mw.working[0])), 0.01);
}

test "mixed precision context" {
    const allocator = std.testing.allocator;

    var ctx = MixedPrecisionContext.init(allocator, .{
        .enabled = true,
        .initial_scale = 256.0,
    });
    defer ctx.deinit();

    // Prepare loss
    const loss = ctx.prepareLoss(0.5);
    try std.testing.expectEqual(@as(f32, 128.0), loss);

    // Process gradients
    var grads = [_]f32{ 256.0, 512.0 };
    const valid = ctx.processGradients(&grads);
    try std.testing.expect(valid);
    try std.testing.expectEqual(@as(f32, 1.0), grads[0]);
    try std.testing.expectEqual(@as(f32, 2.0), grads[1]);
}

test {
    std.testing.refAllDecls(@This());
}
