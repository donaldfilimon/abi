//! Adaptive Persona Modulation
//!
//! Implements ML-hybrid persona routing refinement:
//! - Exponential moving average (EMA) preference tracking per user
//! - Weighted interaction history for persona selection bias
//! - Confidence calibration across routing decisions
//! - Feedback-aware scoring adjustments
//!
//! The modulator sits between the AbiRouter (rule-based) and the final
//! routing decision, adjusting scores based on learned user preferences.

const std = @import("std");
const types = @import("types.zig");

/// Configuration for the adaptive modulation system.
pub const ModulationConfig = struct {
    /// EMA decay factor (0.0 = no history, 1.0 = infinite memory).
    ema_alpha: f32 = 0.3,
    /// Weight given to learned preferences vs rule-based routing (0-1).
    preference_weight: f32 = 0.2,
    /// Minimum interactions before preferences influence routing.
    min_interactions: u32 = 5,
    /// Maximum sessions to track (LRU eviction beyond this).
    max_sessions: u32 = 10_000,
    /// Whether to apply confidence calibration.
    enable_calibration: bool = true,
};

/// Per-persona preference score with EMA tracking.
pub const PersonaPreference = struct {
    /// EMA score for this persona (0.0 - 1.0).
    score: f32,
    /// Total interactions routed to this persona.
    interaction_count: u32,
    /// Total positive outcomes (user satisfaction).
    positive_count: u32,
    /// Last interaction timestamp.
    last_interaction: i64,
};

/// A user's preference profile across all personas.
pub const UserProfile = struct {
    abbey: PersonaPreference,
    aviva: PersonaPreference,
    abi: PersonaPreference,
    total_interactions: u32,

    pub fn getPreference(self: *const UserProfile, persona: types.PersonaType) PersonaPreference {
        return switch (persona) {
            .abbey => self.abbey,
            .aviva => self.aviva,
            .abi => self.abi,
            else => .{ .score = 0.5, .interaction_count = 0, .positive_count = 0, .last_interaction = 0 },
        };
    }

    pub fn setPreference(self: *UserProfile, persona: types.PersonaType, pref: PersonaPreference) void {
        switch (persona) {
            .abbey => self.abbey = pref,
            .aviva => self.aviva = pref,
            .abi => self.abi = pref,
            else => {},
        }
    }
};

/// The result of adaptive modulation on a routing decision.
pub const ModulationResult = struct {
    /// The original routing score (from rules engine).
    original_score: f32,
    /// The modulated score (after preference adjustment).
    modulated_score: f32,
    /// The preference bias applied.
    preference_bias: f32,
    /// The calibration adjustment applied.
    calibration_adjustment: f32,
    /// Whether modulation was active (enough history).
    modulation_active: bool,
};

/// Adaptive Persona Modulator.
pub const AdaptiveModulator = struct {
    allocator: std.mem.Allocator,
    config: ModulationConfig,
    /// User profiles indexed by session/user ID.
    profiles: std.StringHashMapUnmanaged(UserProfile),
    mutex: std.Thread.Mutex,
    /// Running calibration statistics.
    calibration_hits: u64,
    calibration_total: u64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, cfg: ModulationConfig) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
            .config = cfg,
            .profiles = .{},
            .mutex = .{},
            .calibration_hits = 0,
            .calibration_total = 0,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        var it = self.profiles.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.profiles.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Modulate a routing score based on user preference history.
    pub fn modulate(
        self: *Self,
        session_id: []const u8,
        persona: types.PersonaType,
        original_score: f32,
    ) ModulationResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const profile = self.profiles.get(session_id) orelse {
            return .{
                .original_score = original_score,
                .modulated_score = original_score,
                .preference_bias = 0.0,
                .calibration_adjustment = 0.0,
                .modulation_active = false,
            };
        };

        // Check if we have enough history
        if (profile.total_interactions < self.config.min_interactions) {
            return .{
                .original_score = original_score,
                .modulated_score = original_score,
                .preference_bias = 0.0,
                .calibration_adjustment = 0.0,
                .modulation_active = false,
            };
        }

        const pref = profile.getPreference(persona);

        // Compute preference bias
        const preference_bias = (pref.score - 0.5) * self.config.preference_weight;

        // Compute calibration adjustment
        var calibration_adj: f32 = 0.0;
        if (self.config.enable_calibration and self.calibration_total > 0) {
            const hit_rate = @as(f32, @floatFromInt(self.calibration_hits)) /
                @as(f32, @floatFromInt(self.calibration_total));
            // Adjust scores toward calibrated hit rate
            calibration_adj = (hit_rate - 0.5) * 0.1;
        }

        // Blend: modulated = original + preference_bias + calibration
        const modulated = std.math.clamp(original_score + preference_bias + calibration_adj, 0.0, 1.0);

        return .{
            .original_score = original_score,
            .modulated_score = modulated,
            .preference_bias = preference_bias,
            .calibration_adjustment = calibration_adj,
            .modulation_active = true,
        };
    }

    /// Record a user interaction outcome for preference learning.
    pub fn recordInteraction(
        self: *Self,
        session_id: []const u8,
        persona: types.PersonaType,
        was_positive: bool,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = std.time.timestamp();

        // Get or create profile
        const entry = self.profiles.getEntry(session_id);
        var profile: UserProfile = undefined;
        if (entry) |e| {
            profile = e.value_ptr.*;
        } else {
            profile = .{
                .abbey = defaultPreference(),
                .aviva = defaultPreference(),
                .abi = defaultPreference(),
                .total_interactions = 0,
            };
        }

        // Update the specific persona's EMA score
        var pref = profile.getPreference(persona);
        const outcome: f32 = if (was_positive) 1.0 else 0.0;
        pref.score = self.config.ema_alpha * outcome + (1.0 - self.config.ema_alpha) * pref.score;
        pref.interaction_count += 1;
        if (was_positive) pref.positive_count += 1;
        pref.last_interaction = now;
        profile.setPreference(persona, pref);
        profile.total_interactions += 1;

        // Update calibration stats
        self.calibration_total += 1;
        if (was_positive) self.calibration_hits += 1;

        if (entry) |e| {
            e.value_ptr.* = profile;
        } else {
            const owned_id = try self.allocator.dupe(u8, session_id);
            try self.profiles.put(self.allocator, owned_id, profile);
        }
    }

    /// Get a user's preference profile (null if no history).
    pub fn getProfile(self: *Self, session_id: []const u8) ?UserProfile {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.profiles.get(session_id);
    }

    /// Get the current calibration accuracy.
    pub fn calibrationAccuracy(self: *Self) f32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.calibration_total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.calibration_hits)) /
            @as(f32, @floatFromInt(self.calibration_total));
    }

    fn defaultPreference() PersonaPreference {
        return .{
            .score = 0.5,
            .interaction_count = 0,
            .positive_count = 0,
            .last_interaction = 0,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "AdaptiveModulator init and deinit" {
    const allocator = std.testing.allocator;
    var modulator = try AdaptiveModulator.init(allocator, .{});
    defer modulator.deinit();

    try std.testing.expect(modulator.calibrationAccuracy() == 0.0);
}

test "AdaptiveModulator no-history passthrough" {
    const allocator = std.testing.allocator;
    var modulator = try AdaptiveModulator.init(allocator, .{});
    defer modulator.deinit();

    const result = modulator.modulate("new-session", .abbey, 0.8);
    try std.testing.expect(!result.modulation_active);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), result.modulated_score, 0.01);
}

test "AdaptiveModulator learns preferences" {
    const allocator = std.testing.allocator;
    var modulator = try AdaptiveModulator.init(allocator, .{
        .min_interactions = 2,
        .ema_alpha = 0.5,
        .preference_weight = 0.3,
    });
    defer modulator.deinit();

    // Record multiple positive interactions with Abbey
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        try modulator.recordInteraction("user-1", .abbey, true);
    }

    const profile = modulator.getProfile("user-1");
    try std.testing.expect(profile != null);
    try std.testing.expect(profile.?.abbey.score > 0.5); // Should be biased positive
    try std.testing.expect(profile.?.total_interactions == 5);

    // Modulation should now be active
    const result = modulator.modulate("user-1", .abbey, 0.7);
    try std.testing.expect(result.modulation_active);
    try std.testing.expect(result.preference_bias > 0.0);
}

test "AdaptiveModulator calibration" {
    const allocator = std.testing.allocator;
    var modulator = try AdaptiveModulator.init(allocator, .{});
    defer modulator.deinit();

    try modulator.recordInteraction("u1", .abbey, true);
    try modulator.recordInteraction("u1", .abbey, true);
    try modulator.recordInteraction("u1", .abbey, false);

    const accuracy = modulator.calibrationAccuracy();
    try std.testing.expectApproxEqAbs(@as(f32, 0.6667), accuracy, 0.01);
}

test {
    std.testing.refAllDecls(@This());
}
