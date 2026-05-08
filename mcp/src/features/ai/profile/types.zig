//! Shared types for the multi-profile orchestration layer.

const std = @import("std");

/// Identifies which profile is active or targeted.
pub const ProfileId = enum {
    abbey,
    aviva,
    abi,

    pub fn name(self: ProfileId) []const u8 {
        return switch (self) {
            .abbey => "Abbey",
            .aviva => "Aviva",
            .abi => "Abi",
        };
    }

    /// Hex color code per ABBEY-SPEC profile definitions.
    pub fn colorCode(self: ProfileId) []const u8 {
        return switch (self) {
            .abbey => "#00B3A1", // Teal
            .aviva => "#7B4FFF", // Purple
            .abi => "#FF8C42", // Orange
        };
    }

    /// Human-readable role description per spec.
    pub fn role(self: ProfileId) []const u8 {
        return switch (self) {
            .abbey => "Empathetic Polymath",
            .aviva => "Direct Expert",
            .abi => "Adaptive Moderator",
        };
    }

    /// Tone contract from the Abbey/Aviva/Abi spec.
    pub fn tone(self: ProfileId) []const u8 {
        return switch (self) {
            .abbey => "Warm, calm, collaborative, and technically capable",
            .aviva => "Sharp, efficient, factual, and concise",
            .abi => "Policy-aware, transparent, adaptive, and moderating",
        };
    }

    /// Primary behavior contract for routing and response shaping.
    pub fn behaviorContract(self: ProfileId) []const u8 {
        return switch (self) {
            .abbey => "Acknowledge emotion when present, explain clearly, and keep progress actionable",
            .aviva => "Answer directly with minimal framing, high factual density, and execution-ready code",
            .abi => "Analyze intent, policy, sentiment, and context before selecting or blending profiles",
        };
    }

    /// Default generation settings used by profile-aware callers.
    pub fn generationDefaults(self: ProfileId) GenerationDefaults {
        return switch (self) {
            .abbey => .{ .temperature = 0.6, .max_tokens = 1024, .include_empathy = true, .include_policy_rationale = false },
            .aviva => .{ .temperature = 0.2, .max_tokens = 768, .include_empathy = false, .include_policy_rationale = false },
            .abi => .{ .temperature = 0.3, .max_tokens = 896, .include_empathy = false, .include_policy_rationale = true },
        };
    }
};

pub const GenerationDefaults = struct {
    temperature: f32,
    max_tokens: u32,
    include_empathy: bool,
    include_policy_rationale: bool,
};

/// Lifecycle state of a profile instance.
pub const ProfileState = enum {
    uninitialized,
    idle,
    active,
    suspended,
    failed,
};

/// Routing strategy for multi-profile request handling.
pub const RoutingStrategy = enum {
    /// Route to a single best-fit profile.
    single,
    /// Execute multiple profiles in parallel, pick best.
    parallel,
    /// Execute all profiles and merge via consensus.
    consensus,
};

/// Weighted routing decision produced by the router.
pub const RoutingDecision = struct {
    /// Primary profile to handle the request.
    primary: ProfileId,
    /// Normalized weights for each profile (sum to 1.0).
    weights: Weights,
    /// Execution strategy.
    strategy: RoutingStrategy,
    /// Router confidence in this decision (0.0–1.0).
    confidence: f32,
    /// Human-readable reason for the routing choice.
    reason: []const u8,

    pub const Weights = struct {
        abbey: f32 = 0.0,
        aviva: f32 = 0.0,
        abi: f32 = 0.0,

        pub fn forProfile(self: Weights, id: ProfileId) f32 {
            return switch (id) {
                .abbey => self.abbey,
                .aviva => self.aviva,
                .abi => self.abi,
            };
        }

        pub fn normalize(self: *Weights) void {
            const total = self.abbey + self.aviva + self.abi;
            if (total > 0.0) {
                self.abbey /= total;
                self.aviva /= total;
                self.abi /= total;
            } else {
                self.abbey = 0.5;
                self.aviva = 0.3;
                self.abi = 0.2;
            }
        }

        pub fn primary(self: Weights) ProfileId {
            if (self.abbey >= self.aviva and self.abbey >= self.abi) return .abbey;
            if (self.aviva >= self.abi) return .aviva;
            return .abi;
        }

        pub fn sum(self: Weights) f32 {
            return self.abbey + self.aviva + self.abi;
        }
    };
};

/// Spec routing thresholds:
/// - primary > 0.9: single profile
/// - primary in [0.5, 0.9]: parallel/blended execution
/// - no clear primary: consensus
pub fn strategyFromWeights(weights: RoutingDecision.Weights) RoutingStrategy {
    const primary_weight = weights.forProfile(weights.primary());
    if (primary_weight > 0.9) return .single;
    if (primary_weight >= 0.5) return .parallel;
    return .consensus;
}

/// A response produced by a profile.
///
/// MEMORY OWNERSHIP: `content` is ALWAYS owned by this struct.
/// The creator MUST have allocated it with the same `allocator` provided here.
/// Call `deinit()` to free the content when done.
pub const ProfileResponse = struct {
    profile: ProfileId,
    content: []const u8,
    confidence: f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ProfileResponse) void {
        self.allocator.free(self.content);
    }
};

/// Message types for inter-profile communication.
pub const MessageKind = enum {
    /// Request another profile's input.
    request,
    /// Respond to a request.
    response,
    /// Offer an unsolicited opinion.
    opinion,
    /// Veto a proposed action (Abi compliance).
    veto,
};

/// A message passed between profiles on the collaboration bus.
pub const ProfileMessage = struct {
    from: ProfileId,
    to: ?ProfileId, // null = broadcast
    kind: MessageKind,
    payload: []const u8,
    confidence: f32,
    timestamp: i64,
};

/// Configuration for multi-profile routing behavior.
pub const RoutingConfig = struct {
    /// Default execution strategy.
    default_strategy: RoutingStrategy = .single,
    /// Minimum confidence to accept a routing decision without fallback.
    confidence_threshold: f32 = 0.7,
    /// Timeout for parallel execution (ms).
    parallel_timeout_ms: u32 = 5000,
    /// Whether to adapt weights based on feedback.
    enable_weight_learning: bool = false,
};

/// Errors specific to the profile orchestration layer.
pub const ProfileError = error{
    ProfileNotInitialized,
    ProfileSuspended,
    ProfileFailed,
    RoutingFailed,
    BusOverflow,
    Timeout,
    AllProfilesFailed,
    OutOfMemory,
};

test {
    std.testing.refAllDecls(@This());
}
