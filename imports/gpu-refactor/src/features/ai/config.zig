//! Configuration schema for the Multi-Profile AI Assistant system.
//! Defines settings for the routing layer and individual profile behaviors.

const std = @import("std");
const types = @import("types.zig");
const core_config = @import("core/config.zig");

// Re-export ConfigBuilder from core config
pub const ConfigBuilder = core_config.ConfigBuilder;

/// Global configuration for the Multi-Profile system.
pub const MultiProfileConfig = struct {
    /// The fallback profile used when routing fails or is disabled.
    default_profile: types.ProfileType = .abbey,
    /// Whether to use the Abi router for dynamic profile selection.
    enable_dynamic_routing: bool = true,
    /// Minimum confidence threshold for routing decisions (0.0 - 1.0).
    routing_confidence_threshold: f32 = 0.6,

    /// Configuration for the Abi moderation and routing layer.
    abi: AbiConfig = .{},
    /// Configuration for the Abbey empathetic polymath profile.
    abbey: AbbeyConfig = .{},
    /// Configuration for the Aviva direct expert profile.
    aviva: AvivaConfig = .{},
    /// Configuration for profile load balancing and resilience.
    load_balancing: LoadBalancingConfig = .{},
    /// Configuration for the profile registry and lifecycle.
    registry: RegistryConfig = .{},
};

/// Settings for the Abi router and moderator.
pub const AbiConfig = struct {
    /// Enable sentiment and emotion detection for routing.
    enable_sentiment_analysis: bool = true,
    /// Enable content safety and policy enforcement.
    enable_policy_checking: bool = true,
    /// Detect and handle sensitive or controversial topics.
    sensitive_topic_detection: bool = true,
    /// Level of strictness for the content filter.
    content_filter_level: FilterLevel = .moderate,
    /// Target latency budget for routing decisions.
    max_routing_latency_ms: u64 = 50,

    pub const FilterLevel = enum {
        low,
        moderate,
        strict,
    };
};

/// Settings for the Abbey profile.
pub const AbbeyConfig = struct {
    /// Target empathy level (0.0 - 1.0).
    empathy_level: f32 = 0.8,
    /// Preferred technical depth of explanations (0.0 - 1.0).
    technical_depth: f32 = 0.7,
    /// Whether to include visible reasoning chains in responses.
    include_reasoning: bool = true,
    /// Maximum steps allowed in a reasoning chain.
    max_reasoning_steps: u32 = 5,
    /// Adapt tone and temperature based on user emotions.
    emotion_adaptation: bool = true,

    // ── Behavioral refinements per unified spec §4-§7 ────────────────

    /// User skill level for tone calibration (spec §5.1).
    /// beginner → explanatory, example-driven
    /// intermediate → structured with reasoning
    /// advanced → concise with depth and edge cases
    skill_level: SkillLevel = .intermediate,

    /// Detail level for the adaptation layer (spec §7).
    detail_level: DetailLevel = .medium,

    /// Enable structured debugging flow (spec §6):
    /// restate → context → triage → root cause → fix → validate
    structured_debugging: bool = true,

    /// Enable structured explanation flow (spec §6):
    /// intuition → example → formal model → extension
    structured_explanations: bool = true,

    /// User skill level for adaptive tone calibration.
    pub const SkillLevel = enum {
        /// Explanatory, example-driven responses.
        beginner,
        /// Structured responses with reasoning.
        intermediate,
        /// Concise with depth and edge cases.
        advanced,

        /// Get temperature adjustment for this skill level.
        pub fn getTemperature(self: SkillLevel) f32 {
            return switch (self) {
                .beginner => 0.8, // More creative, exploratory
                .intermediate => 0.7, // Balanced
                .advanced => 0.5, // Precise, focused
            };
        }

        /// Get max response length multiplier.
        pub fn getVerbosityMultiplier(self: SkillLevel) f32 {
            return switch (self) {
                .beginner => 1.5, // More detailed
                .intermediate => 1.0, // Standard
                .advanced => 0.7, // Concise
            };
        }
    };

    /// Detail level for response adaptation.
    pub const DetailLevel = enum {
        low, // Quick answers, minimal explanation
        medium, // Standard depth with reasoning
        high, // Exhaustive coverage with edge cases
    };
};

/// Settings for the Aviva profile.
pub const AvivaConfig = struct {
    /// Level of directness and brevity (0.0 - 1.0).
    directness_level: f32 = 0.9,
    /// Whether to include standard AI disclaimers or hedges.
    include_disclaimers: bool = false,
    /// Include comments and documentation in generated code.
    include_code_comments: bool = true,
    /// Run an internal fact-check pass before responding.
    verify_facts: bool = true,
    /// Maximum allowed response length in tokens.
    max_response_length: u32 = 4096,
    /// Whether to cite sources in responses.
    cite_sources: bool = false,
    /// Whether to skip preamble/boilerplate in code generation.
    skip_preamble: bool = false,
};

/// Settings for profile scaling and resilience.
pub const LoadBalancingConfig = struct {
    /// Strategy used for selecting between multiple instances.
    strategy: LoadBalancerStrategy = .health_weighted,
    /// Enable circuit breaker to prevent cascading failures.
    enable_circuit_breaker: bool = true,
    /// Number of failures before tripping the circuit breaker.
    circuit_breaker_threshold: u32 = 5,
    /// How long to wait before attempting recovery (ms).
    circuit_breaker_timeout_ms: u64 = 30000,
    /// Maximum concurrent requests per profile (0 = unlimited).
    max_concurrent_requests: u32 = 0,

    pub const LoadBalancerStrategy = enum {
        round_robin,
        least_busy,
        health_weighted,
    };
};

/// Settings for the profile registry and service discovery.
pub const RegistryConfig = struct {
    /// Maximum concurrent requests per profile.
    max_concurrent_requests: u32 = 100,
    /// Request timeout in milliseconds.
    timeout_ms: u64 = 30000,
    /// Default priority for registered profiles.
    default_priority: u8 = 5,
};

test {
    std.testing.refAllDecls(@This());
}
