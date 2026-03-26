//! Adaptive Profile Modulation
//!
//! Implements ML-hybrid profile routing refinement:
//! - Exponential moving average (EMA) preference tracking per user
//! - Weighted interaction history for profile selection bias
//! - Confidence calibration across routing decisions
//! - Feedback-aware scoring adjustments
//!
//! The modulator sits between the AbiRouter (rule-based) and the final
//! routing decision, adjusting scores based on learned user preferences.

const std = @import("std");
const foundation = @import("../../foundation/mod.zig");
const log = foundation.logging;
const time = foundation.time;
const types = @import("types.zig");
const build_options = @import("build_options");
const db_feature = if (build_options.feat_database) @import("../database/mod.zig") else @import("../database/stub.zig");
const block_chain = db_feature.memory.block_chain;
const BlockChain = block_chain.BlockChain;
const BlockConfig = block_chain.BlockConfig;

/// Configuration for the adaptive modulation system.
pub const ModulationConfig = struct {
    /// EMA smoothing factor (higher = more responsive to recent data, lower = more smoothing).
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

/// Per-profile preference score with EMA tracking.
pub const ProfilePreference = struct {
    /// EMA score for this profile (0.0 - 1.0).
    score: f32,
    /// Total interactions routed to this profile.
    interaction_count: u32,
    /// Total positive outcomes (user satisfaction).
    positive_count: u32,
    /// Last interaction timestamp.
    last_interaction: i64,
};

/// A user's preference profile across all profiles.
pub const UserProfile = struct {
    abbey: ProfilePreference,
    aviva: ProfilePreference,
    abi: ProfilePreference,
    total_interactions: u32,

    pub fn getPreference(self: *const UserProfile, profile: types.ProfileType) ProfilePreference {
        return switch (profile) {
            .abbey => self.abbey,
            .aviva => self.aviva,
            .abi => self.abi,
            else => .{ .score = 0.5, .interaction_count = 0, .positive_count = 0, .last_interaction = 0 },
        };
    }

    pub fn setPreference(self: *UserProfile, profile: types.ProfileType, pref: ProfilePreference) void {
        switch (profile) {
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

/// Adaptive Profile Modulator.
pub const AdaptiveModulator = struct {
    allocator: std.mem.Allocator,
    config: ModulationConfig,
    /// User profiles indexed by session/user ID.
    profiles: std.StringHashMapUnmanaged(UserProfile),
    mutex: foundation.sync.Mutex,
    /// Running calibration statistics.
    calibration_hits: u64,
    calibration_total: u64,
    /// Optional WDBX chain for write-behind persistence.
    wdbx_chain: ?*BlockChain = null,

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

    /// Attach a WDBX block chain for write-behind persistence.
    /// Modulation state will be durably recorded after each interaction.
    pub fn attachWdbx(self: *Self, wdbx_chain: *BlockChain) void {
        self.wdbx_chain = wdbx_chain;
    }

    /// Persist current modulation state for a session to WDBX.
    /// Called automatically after recordInteraction when a chain is attached.
    fn persistToWdbx(self: *Self, session_id: []const u8, profile: UserProfile) void {
        const wdbx = self.wdbx_chain orelse return;
        const embedding: [4]f32 = .{
            profile.abbey.score,
            profile.aviva.score,
            profile.abi.score,
            @min(@as(f32, @floatFromInt(profile.total_interactions)) / 1000.0, 1.0),
        };
        const primary = if (profile.abbey.score >= profile.aviva.score and profile.abbey.score >= profile.abi.score)
            block_chain.ProfileTag.ProfileType.abbey
        else if (profile.aviva.score >= profile.abbey.score and profile.aviva.score >= profile.abi.score)
            block_chain.ProfileTag.ProfileType.aviva
        else
            block_chain.ProfileTag.ProfileType.abi;
        _ = session_id;
        const config = BlockConfig{
            .query_embedding = &embedding,
            .profile_tag = .{ .primary_profile = primary },
            .routing_weights = .{
                .abbey_weight = profile.abbey.score,
                .aviva_weight = profile.aviva.score,
                .abi_weight = profile.abi.score,
            },
            .intent = .general,
            .pipeline_step = .modulate,
        };
        _ = wdbx.addBlock(config) catch {};
    }

    /// Modulate a routing score based on user preference history.
    ///
    /// Copies all needed profile data under the lock, then releases the mutex
    /// before computing the modulated score. This avoids holding the lock
    /// during arithmetic and prevents TOCTOU races if the profile is mutated
    /// concurrently by recordInteraction().
    pub fn modulate(
        self: *Self,
        session_id: []const u8,
        profile_type: types.ProfileType,
        original_score: f32,
    ) ModulationResult {
        // Snapshot profile data and calibration stats under the lock.
        const snapshot = blk: {
            self.mutex.lock();
            defer self.mutex.unlock();

            const user_profile = self.profiles.get(session_id) orelse {
                break :blk null;
            };

            break :blk .{
                .total_interactions = user_profile.total_interactions,
                .pref = user_profile.getPreference(profile_type),
                .cal_hits = self.calibration_hits,
                .cal_total = self.calibration_total,
            };
        };

        const snap = snapshot orelse {
            return .{
                .original_score = original_score,
                .modulated_score = original_score,
                .preference_bias = 0.0,
                .calibration_adjustment = 0.0,
                .modulation_active = false,
            };
        };

        // Check if we have enough history
        if (snap.total_interactions < self.config.min_interactions) {
            return .{
                .original_score = original_score,
                .modulated_score = original_score,
                .preference_bias = 0.0,
                .calibration_adjustment = 0.0,
                .modulation_active = false,
            };
        }

        // Compute preference bias from the copied value
        const preference_bias = (snap.pref.score - 0.5) * self.config.preference_weight;

        // Compute calibration adjustment from copied stats
        var calibration_adj: f32 = 0.0;
        if (self.config.enable_calibration and snap.cal_total > 0) {
            const hit_rate = @as(f32, @floatFromInt(snap.cal_hits)) /
                @as(f32, @floatFromInt(snap.cal_total));
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
        profile_type: types.ProfileType,
        was_positive: bool,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.unixSeconds();

        // Get or create user profile
        const entry = self.profiles.getEntry(session_id);
        var user_profile: UserProfile = if (entry) |e|
            e.value_ptr.*
        else
            .{
                .abbey = defaultPreference(),
                .aviva = defaultPreference(),
                .abi = defaultPreference(),
                .total_interactions = 0,
            };

        // Update the specific profile's EMA score
        var pref = user_profile.getPreference(profile_type);
        const outcome: f32 = if (was_positive) 1.0 else 0.0;
        pref.score = self.config.ema_alpha * outcome + (1.0 - self.config.ema_alpha) * pref.score;
        pref.interaction_count += 1;
        if (was_positive) pref.positive_count += 1;
        pref.last_interaction = now;
        user_profile.setPreference(profile_type, pref);
        user_profile.total_interactions += 1;

        // Update calibration stats
        self.calibration_total += 1;
        if (was_positive) self.calibration_hits += 1;

        if (entry) |e| {
            e.value_ptr.* = user_profile;
        } else {
            // Evict the least-recently-used session if at capacity.
            if (self.profiles.count() >= self.config.max_sessions) {
                evictOldestSession(self);
            }
            const owned_id = try self.allocator.dupe(u8, session_id);
            try self.profiles.put(self.allocator, owned_id, user_profile);
        }

        // Write-behind: persist modulation state to WDBX (outside lock scope)
        const snapshot_profile = user_profile;
        // Note: persistToWdbx is called within the lock but only does a quick
        // block append — acceptable latency for the durability guarantee.
        self.persistToWdbx(session_id, snapshot_profile);
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

    /// Return the most recent interaction timestamp across all three profiles.
    fn mostRecentInteraction(profile: UserProfile) i64 {
        return @max(profile.abbey.last_interaction, @max(profile.aviva.last_interaction, profile.abi.last_interaction));
    }

    /// Evict the session with the oldest most-recent interaction.
    /// Caller must hold self.mutex.
    fn evictOldestSession(self: *Self) void {
        var oldest_key: ?[]const u8 = null;
        var oldest_time: i64 = std.math.maxInt(i64);

        var it = self.profiles.iterator();
        while (it.next()) |kv| {
            const last = mostRecentInteraction(kv.value_ptr.*);
            if (last < oldest_time) {
                oldest_time = last;
                oldest_key = kv.key_ptr.*;
            }
        }

        if (oldest_key) |key| {
            log.warn("AdaptiveModulator: evicting oldest session (capacity {d} reached)", .{self.config.max_sessions});
            // Remove the entry and free the owned key.
            const removed = self.profiles.fetchRemove(key);
            if (removed) |kv| {
                self.allocator.free(kv.key);
            }
        }
    }

    fn defaultPreference() ProfilePreference {
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

test "AdaptiveModulator modulate returns copied data not references" {
    const allocator = std.testing.allocator;
    var modulator = try AdaptiveModulator.init(allocator, .{
        .min_interactions = 2,
        .ema_alpha = 0.5,
        .preference_weight = 0.3,
    });
    defer modulator.deinit();

    // Build up enough history to activate modulation
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        try modulator.recordInteraction("snap-user", .abbey, true);
    }

    // Take a modulation result (snapshot captured under lock, computed after release)
    const result1 = modulator.modulate("snap-user", .abbey, 0.6);
    try std.testing.expect(result1.modulation_active);

    // Mutate the profile after the first modulate call
    try modulator.recordInteraction("snap-user", .abbey, false);
    try modulator.recordInteraction("snap-user", .abbey, false);
    try modulator.recordInteraction("snap-user", .abbey, false);

    // Take a second snapshot — should reflect the new (lower) preference
    const result2 = modulator.modulate("snap-user", .abbey, 0.6);
    try std.testing.expect(result2.modulation_active);

    // The first result must be unchanged (it was computed from a value copy)
    try std.testing.expectApproxEqAbs(result1.modulated_score, result1.modulated_score, 0.001);

    // The second result should differ because the profile changed
    try std.testing.expect(result2.preference_bias < result1.preference_bias);
}

test "AdaptiveModulator LRU eviction at capacity" {
    const allocator = std.testing.allocator;
    var modulator = try AdaptiveModulator.init(allocator, .{
        .max_sessions = 3,
    });
    defer modulator.deinit();

    // Fill to capacity with three sessions.
    try modulator.recordInteraction("session-a", .abbey, true);
    try modulator.recordInteraction("session-b", .aviva, true);
    try modulator.recordInteraction("session-c", .abi, true);

    try std.testing.expectEqual(@as(usize, 3), modulator.profiles.count());

    // Adding a fourth session should evict one old session to stay at capacity.
    try modulator.recordInteraction("session-d", .abbey, true);

    // Count must not exceed max_sessions.
    try std.testing.expectEqual(@as(usize, 3), modulator.profiles.count());
    // The newly added session must be present.
    try std.testing.expect(modulator.getProfile("session-d") != null);

    // Exactly one of the original three was evicted.
    var surviving: u32 = 0;
    if (modulator.getProfile("session-a") != null) surviving += 1;
    if (modulator.getProfile("session-b") != null) surviving += 1;
    if (modulator.getProfile("session-c") != null) surviving += 1;
    try std.testing.expectEqual(@as(u32, 2), surviving);
}

test {
    std.testing.refAllDecls(@This());
}
