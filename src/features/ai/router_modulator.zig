const std = @import("std");
const identity = @import("identity.zig");
const weights = @import("router_weights.zig");

pub const ProfileWeights = weights.ProfileWeights;

/// Adaptive EMA Modulator that smooths routing weights over time
/// and persists them to a WDBX store.
pub const AdaptiveModulator = struct {
    w_ema: ProfileWeights,
    alpha: f32,
    update_count: u32,

    const STORE_KEY = "modulator:weights";
    const DEFAULT_ALPHA: f32 = 0.3;

    pub fn init() AdaptiveModulator {
        return .{
            .w_ema = ProfileWeights{
                .w_abbey = identity.DEFAULT_ABBEY_WEIGHT,
                .w_aviva = identity.DEFAULT_AVIVA_WEIGHT,
                .w_abi = identity.DEFAULT_ABI_WEIGHT,
            },
            .alpha = DEFAULT_ALPHA,
            .update_count = 0,
        };
    }

    pub fn initWithAlpha(alpha: f32) AdaptiveModulator {
        return .{
            .w_ema = ProfileWeights{
                .w_abbey = identity.DEFAULT_ABBEY_WEIGHT,
                .w_aviva = identity.DEFAULT_AVIVA_WEIGHT,
                .w_abi = identity.DEFAULT_ABI_WEIGHT,
            },
            .alpha = alpha,
            .update_count = 0,
        };
    }

    /// Update the EMA weights with new observed sentiment weights.
    /// new_ema = alpha * observed + (1 - alpha) * old_ema
    pub fn update(self: *AdaptiveModulator, observed: ProfileWeights) void {
        const a = self.alpha;
        const b = 1.0 - a;
        self.w_ema.w_abbey = a * observed.w_abbey + b * self.w_ema.w_abbey;
        self.w_ema.w_aviva = a * observed.w_aviva + b * self.w_ema.w_aviva;
        self.w_ema.w_abi = a * observed.w_abi + b * self.w_ema.w_abi;
        self.w_ema.normalize();
        self.update_count +|= 1;
    }

    /// Get the current smoothed weights.
    pub fn weights(self: *const AdaptiveModulator) ProfileWeights {
        return self.w_ema;
    }

    /// Serialize weights to a string for WDBX persistence.
    pub fn serialize(self: *const AdaptiveModulator, allocator: std.mem.Allocator) ![]u8 {
        return try std.fmt.allocPrint(
            allocator,
            "{d:.6},{d:.6},{d:.6},{d},{d:.6}",
            .{ self.w_ema.w_abbey, self.w_ema.w_aviva, self.w_ema.w_abi, self.update_count, self.alpha },
        );
    }

    /// Deserialize weights from a stored string. Falls back to
    /// `AdaptiveModulator.init()` defaults if the persisted state fails
    /// validation (malformed fields, non-finite/negative weights, invalid
    /// totals/counts, or alpha outside `[0,1]`).
    pub fn deserialize(data: []const u8) AdaptiveModulator {
        return deserializeValidated(data) orelse AdaptiveModulator.init();
    }

    /// Parses and validates the CSV-encoded persisted state. Returns `null`
    /// (rather than a partially-defaulted value) if any field is malformed,
    /// a weight is non-finite/negative, the weight total is invalid, alpha is
    /// outside `[0,1]`, or the field count doesn't match exactly.
    fn deserializeValidated(data: []const u8) ?AdaptiveModulator {
        var it = std.mem.splitScalar(u8, data, ',');
        const abbey_text = it.next() orelse return null;
        const aviva_text = it.next() orelse return null;
        const abi_text = it.next() orelse return null;
        const update_count_text = it.next() orelse return null;
        const alpha_text = it.next() orelse return null;
        if (it.next() != null) return null;

        const abbey_weight = std.fmt.parseFloat(f32, abbey_text) catch return null;
        const aviva_weight = std.fmt.parseFloat(f32, aviva_text) catch return null;
        const abi_weight = std.fmt.parseFloat(f32, abi_text) catch return null;
        const update_count = std.fmt.parseInt(u32, update_count_text, 10) catch return null;
        const alpha = std.fmt.parseFloat(f32, alpha_text) catch return null;

        if (!std.math.isFinite(abbey_weight) or
            !std.math.isFinite(aviva_weight) or
            !std.math.isFinite(abi_weight) or
            abbey_weight < 0 or
            aviva_weight < 0 or
            abi_weight < 0)
        {
            return null;
        }

        const total = abbey_weight + aviva_weight + abi_weight;
        if (!std.math.isFinite(total) or total <= 0) return null;
        if (!std.math.isFinite(alpha) or alpha < 0 or alpha > 1) return null;

        var mod = AdaptiveModulator{
            .w_ema = .{
                .w_abbey = abbey_weight,
                .w_aviva = aviva_weight,
                .w_abi = abi_weight,
            },
            .alpha = alpha,
            .update_count = update_count,
        };
        mod.w_ema.normalize();
        return mod;
    }

    /// Load weights from a WDBX store. Returns default if key is missing.
    pub fn loadWeights(store: anytype) AdaptiveModulator {
        const val = store.get(STORE_KEY) orelse return AdaptiveModulator.init();
        return AdaptiveModulator.deserialize(val);
    }

    /// Save current weights to a WDBX store.
    pub fn saveWeights(self: *const AdaptiveModulator, allocator: std.mem.Allocator, store: anytype) !void {
        const serialized = try self.serialize(allocator);
        defer allocator.free(serialized);
        try store.store(STORE_KEY, serialized);
    }
};

test "AdaptiveModulator EMA smoothing" {
    var mod = AdaptiveModulator.initWithAlpha(0.5);
    const observed = ProfileWeights{ .w_abbey = 1.0, .w_aviva = 0.0, .w_abi = 0.0 };
    mod.update(observed);

    // After one update with alpha=0.5, abbey should be higher than initial
    try std.testing.expect(mod.w_ema.w_abbey > 0.5);
    try std.testing.expectEqual(@as(u32, 1), mod.update_count);

    // Weights should still be normalized
    const total = mod.w_ema.w_abbey + mod.w_ema.w_aviva + mod.w_ema.w_abi;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.01);
}

test "AdaptiveModulator serialize/deserialize roundtrip" {
    var mod = AdaptiveModulator.initWithAlpha(0.25);
    const observed = ProfileWeights{ .w_abbey = 0.8, .w_aviva = 0.1, .w_abi = 0.1 };
    mod.update(observed);

    const allocator = std.testing.allocator;
    const serialized = try mod.serialize(allocator);
    defer allocator.free(serialized);

    const restored = AdaptiveModulator.deserialize(serialized);
    try std.testing.expectApproxEqAbs(mod.w_ema.w_abbey, restored.w_ema.w_abbey, 0.001);
    try std.testing.expectApproxEqAbs(mod.w_ema.w_aviva, restored.w_ema.w_aviva, 0.001);
    try std.testing.expectApproxEqAbs(mod.w_ema.w_abi, restored.w_ema.w_abi, 0.001);
    try std.testing.expectEqual(mod.update_count, restored.update_count);
    try std.testing.expectApproxEqAbs(mod.alpha, restored.alpha, 0.001);
}

test "AdaptiveModulator default deserialization on missing key" {
    const mod = AdaptiveModulator.deserialize("");
    try std.testing.expectApproxEqAbs(identity.DEFAULT_ABBEY_WEIGHT, mod.w_ema.w_abbey, 0.01);
    try std.testing.expectEqual(@as(u32, 0), mod.update_count);
}

test "AdaptiveModulator rejects invalid persisted state deterministically" {
    const invalid_states = [_][]const u8{
        "nan,0.3,0.4,1,0.3",
        "inf,0.3,0.4,1,0.3",
        "-inf,0.3,0.4,1,0.3",
        "-0.1,0.3,0.8,1,0.3",
        "0,0,0,1,0.3",
        "3.4028235e38,3.4028235e38,3.4028235e38,1,0.3",
        "0.3,0.3,0.4,1,nan",
        "0.3,0.3,0.4,1,inf",
        "0.3,0.3,0.4,1,-0.1",
        "0.3,0.3,0.4,1,1.1",
        "malformed,0.3,0.4,1,0.3",
        "0.3,,0.4,1,0.3",
        "0.3,0.3,0.4,not-a-count,0.3",
        "0.3,0.3,0.4,4294967296,0.3",
        "0.3,0.3,0.4,1",
        "0.3,0.3,0.4,1,0.3,",
        "0.3,0.3,0.4,1,0.3,extra",
    };

    for (invalid_states) |state| {
        const restored = AdaptiveModulator.deserialize(state);
        try std.testing.expectApproxEqAbs(identity.DEFAULT_ABBEY_WEIGHT, restored.w_ema.w_abbey, 0.0001);
        try std.testing.expectApproxEqAbs(identity.DEFAULT_AVIVA_WEIGHT, restored.w_ema.w_aviva, 0.0001);
        try std.testing.expectApproxEqAbs(identity.DEFAULT_ABI_WEIGHT, restored.w_ema.w_abi, 0.0001);
        try std.testing.expectEqual(@as(u32, 0), restored.update_count);
        try std.testing.expectApproxEqAbs(@as(f32, 0.3), restored.alpha, 0.0001);
    }
}

test "AdaptiveModulator normalizes valid persisted weights" {
    const restored = AdaptiveModulator.deserialize("2,3,5,42,0.75");
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), restored.w_ema.w_abbey, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), restored.w_ema.w_aviva, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), restored.w_ema.w_abi, 0.0001);
    try std.testing.expectEqual(@as(u32, 42), restored.update_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), restored.alpha, 0.0001);
}

test "AdaptiveModulator accepts persisted alpha boundaries" {
    const zero_alpha = AdaptiveModulator.deserialize("2,3,5,7,0");
    try std.testing.expectEqual(@as(u32, 7), zero_alpha.update_count);
    try std.testing.expectEqual(@as(f32, 0), zero_alpha.alpha);

    const one_alpha = AdaptiveModulator.deserialize("2,3,5,9,1");
    try std.testing.expectEqual(@as(u32, 9), one_alpha.update_count);
    try std.testing.expectEqual(@as(f32, 1), one_alpha.alpha);
}

test "AdaptiveModulator saturates a persisted maximum update count" {
    var restored = AdaptiveModulator.deserialize("0.2,0.3,0.5,4294967295,0.5");
    const observed = ProfileWeights{ .w_abbey = 1.0, .w_aviva = 0.0, .w_abi = 0.0 };
    restored.update(observed);

    try std.testing.expectEqual(std.math.maxInt(u32), restored.update_count);
    const total = restored.w_ema.w_abbey + restored.w_ema.w_aviva + restored.w_ema.w_abi;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.0001);
}

test {
    std.testing.refAllDecls(@This());
}
