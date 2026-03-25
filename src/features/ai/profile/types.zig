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
            }
        }
    };
};

/// A response produced by a profile.
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
