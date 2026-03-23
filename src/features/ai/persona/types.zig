//! Shared types for the multi-persona orchestration layer.

const std = @import("std");

/// Identifies which persona is active or targeted.
pub const PersonaId = enum {
    abbey,
    aviva,
    abi,

    pub fn name(self: PersonaId) []const u8 {
        return switch (self) {
            .abbey => "Abbey",
            .aviva => "Aviva",
            .abi => "Abi",
        };
    }

    /// Hex color code per ABBEY-SPEC persona definitions.
    pub fn colorCode(self: PersonaId) []const u8 {
        return switch (self) {
            .abbey => "#00B3A1", // Teal
            .aviva => "#7B4FFF", // Purple
            .abi => "#FF8C42", // Orange
        };
    }

    /// Human-readable role description per spec.
    pub fn role(self: PersonaId) []const u8 {
        return switch (self) {
            .abbey => "Empathetic Polymath",
            .aviva => "Direct Expert",
            .abi => "Adaptive Moderator",
        };
    }
};

/// Lifecycle state of a persona instance.
pub const PersonaState = enum {
    uninitialized,
    idle,
    active,
    suspended,
    failed,
};

/// Routing strategy for multi-persona request handling.
pub const RoutingStrategy = enum {
    /// Route to a single best-fit persona.
    single,
    /// Execute multiple personas in parallel, pick best.
    parallel,
    /// Execute all personas and merge via consensus.
    consensus,
};

/// Weighted routing decision produced by the router.
pub const RoutingDecision = struct {
    /// Primary persona to handle the request.
    primary: PersonaId,
    /// Normalized weights for each persona (sum to 1.0).
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

        pub fn forPersona(self: Weights, id: PersonaId) f32 {
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

/// A response produced by a persona.
pub const PersonaResponse = struct {
    persona: PersonaId,
    content: []const u8,
    confidence: f32,
    reasoning: ?[]const u8 = null,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PersonaResponse) void {
        self.allocator.free(self.content);
        if (self.reasoning) |r| self.allocator.free(r);
    }
};

/// Message types for inter-persona communication.
pub const MessageKind = enum {
    /// Request another persona's input.
    request,
    /// Respond to a request.
    response,
    /// Offer an unsolicited opinion.
    opinion,
    /// Veto a proposed action (Abi compliance).
    veto,
};

/// A message passed between personas on the collaboration bus.
pub const PersonaMessage = struct {
    from: PersonaId,
    to: ?PersonaId, // null = broadcast
    kind: MessageKind,
    payload: []const u8,
    confidence: f32,
    timestamp: i64,
};

/// Configuration for multi-persona routing behavior.
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

/// Errors specific to the persona orchestration layer.
pub const PersonaError = error{
    PersonaNotInitialized,
    PersonaSuspended,
    PersonaFailed,
    RoutingFailed,
    BusOverflow,
    Timeout,
    AllPersonasFailed,
    OutOfMemory,
};

test {
    std.testing.refAllDecls(@This());
}
