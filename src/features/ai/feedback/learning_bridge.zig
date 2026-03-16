//! Learning Bridge — Connects feedback quality monitoring to self-learning retraining.
//!
//! Monitors per-persona satisfaction ratings from the FeedbackSystem and
//! triggers the SelfLearningSystem when quality drops below a configurable
//! threshold. This closes the feedback loop: poor ratings automatically
//! initiate a retraining cycle.
//!
//! Integration:
//! ```
//!  FeedbackSystem ──analyzePersona()──► LearningBridge ──update()──► SelfLearningSystem
//!                                           │
//!                                     checks threshold,
//!                                     respects interval
//! ```

const std = @import("std");
const time = @import("../../../services/shared/mod.zig").time;
const collector_mod = @import("collector.zig");
const feedback_mod = @import("mod.zig");
const self_learning_mod = @import("self_learning");

const PersonaRef = collector_mod.PersonaRef;
const Trend = @import("analyzer.zig").Trend;
const FeedbackSystem = feedback_mod.FeedbackSystem;
const SelfLearningSystem = self_learning_mod.SelfLearningSystem;

/// All PersonaRef variants for iteration.
const all_personas = [_]PersonaRef{ .abbey, .aviva, .abi, .ralph, .other };

/// Configuration for the learning bridge.
pub const BridgeConfig = struct {
    /// Average rating below this triggers retraining.
    quality_threshold: f32 = 3.0,
    /// Minimum feedback entries before evaluating a persona.
    min_entries: usize = 10,
    /// Minimum seconds between automatic checks.
    check_interval_seconds: u64 = 3600,
    /// Whether to automatically retrain when quality drops.
    auto_retrain: bool = true,
};

/// Per-persona quality health report.
pub const PersonaHealth = struct {
    persona: PersonaRef,
    average_rating: f32,
    entry_count: usize,
    needs_retrain: bool,
    trend: Trend,
};

/// Bridges the FeedbackSystem to the SelfLearningSystem.
///
/// Periodically evaluates persona satisfaction via the feedback analyzer
/// and triggers a learning update when any persona falls below the
/// configured quality threshold.
pub const LearningBridge = struct {
    feedback: *FeedbackSystem,
    learning: *SelfLearningSystem,
    config: BridgeConfig,
    last_check_timestamp: i64,

    const Self = @This();

    /// Create a new LearningBridge. Does not take ownership of subsystems.
    pub fn init(
        feedback: *FeedbackSystem,
        learning: *SelfLearningSystem,
        config: BridgeConfig,
    ) LearningBridge {
        return .{
            .feedback = feedback,
            .learning = learning,
            .config = config,
            .last_check_timestamp = 0,
        };
    }

    /// No-op — the bridge does not own either subsystem.
    pub fn deinit(self: *LearningBridge) void {
        _ = self;
    }

    /// Check all personas and trigger retraining if any falls below threshold.
    ///
    /// Respects `check_interval_seconds`: returns `false` immediately if called
    /// before the interval has elapsed (unless this is the first check).
    /// Returns `true` if a retrain was triggered, `false` otherwise.
    pub fn checkAndTrigger(self: *Self) !bool {
        const now = time.unixSeconds();

        // Respect check interval (skip if called too soon, unless first check)
        if (self.last_check_timestamp != 0) {
            const elapsed: u64 = @intCast(@max(0, now - self.last_check_timestamp));
            if (elapsed < self.config.check_interval_seconds) {
                return false;
            }
        }

        self.last_check_timestamp = now;

        if (!self.config.auto_retrain) {
            return false;
        }

        // Evaluate each persona
        for (all_personas) |persona| {
            const stats = try self.feedback.analyzePersona(persona);

            if (stats.total_entries >= self.config.min_entries and
                stats.average_rating < self.config.quality_threshold)
            {
                try self.learning.update();
                return true;
            }
        }

        return false;
    }

    /// Return a health report for a single persona.
    pub fn getPersonaHealth(self: *Self, persona: PersonaRef) !PersonaHealth {
        const stats = try self.feedback.analyzePersona(persona);

        return .{
            .persona = persona,
            .average_rating = stats.average_rating,
            .entry_count = stats.total_entries,
            .needs_retrain = stats.total_entries >= self.config.min_entries and
                stats.average_rating < self.config.quality_threshold,
            .trend = stats.trend,
        };
    }

    /// Unconditionally trigger a learning update, bypassing threshold and interval checks.
    pub fn forceRetrain(self: *Self) !void {
        try self.learning.update();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "BridgeConfig defaults" {
    const cfg = BridgeConfig{};
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), cfg.quality_threshold, 0.001);
    try std.testing.expect(cfg.min_entries == 10);
    try std.testing.expect(cfg.check_interval_seconds == 3600);
    try std.testing.expect(cfg.auto_retrain);
}

test {
    std.testing.refAllDecls(@This());
}
