//! Configuration schema for the Multi-Persona AI Assistant system.
//! Defines settings for the routing layer and individual persona behaviors.

const std = @import("std");
const types = @import("types.zig");

/// Global configuration for the Multi-Persona system.
pub const MultiPersonaConfig = struct {
    /// The fallback persona used when routing fails or is disabled.
    default_persona: types.PersonaType = .abbey,
    /// Whether to use the Abi router for dynamic persona selection.
    enable_dynamic_routing: bool = true,
    /// Minimum confidence threshold for routing decisions (0.0 - 1.0).
    routing_confidence_threshold: f32 = 0.6,

    /// Configuration for the Abi moderation and routing layer.
    abi: AbiConfig = .{},
    /// Configuration for the Abbey empathetic polymath persona.
    abbey: AbbeyConfig = .{},
    /// Configuration for the Aviva direct expert persona.
    aviva: AvivaConfig = .{},
    /// Configuration for persona load balancing and resilience.
    load_balancing: LoadBalancingConfig = .{},
    /// Configuration for the persona registry and lifecycle.
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

/// Settings for the Abbey persona.
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
};

/// Settings for the Aviva persona.
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

/// Settings for persona scaling and resilience.
pub const LoadBalancingConfig = struct {
    /// Strategy used for selecting between multiple instances.
    strategy: LoadBalancerStrategy = .health_weighted,
    /// Enable circuit breaker to prevent cascading failures.
    enable_circuit_breaker: bool = true,
    /// Number of failures before tripping the circuit breaker.
    circuit_breaker_threshold: u32 = 5,
    /// How long to wait before attempting recovery (ms).
    circuit_breaker_timeout_ms: u64 = 30000,
    /// Maximum concurrent requests per persona (0 = unlimited).
    max_concurrent_requests: u32 = 0,

    pub const LoadBalancerStrategy = enum {
        round_robin,
        least_busy,
        health_weighted,
    };
};

/// Settings for the persona registry and service discovery.
pub const RegistryConfig = struct {
    /// Maximum concurrent requests per persona.
    max_concurrent_requests: u32 = 100,
    /// Request timeout in milliseconds.
    timeout_ms: u64 = 30000,
    /// Default priority for registered personas.
    default_priority: u8 = 5,
};
