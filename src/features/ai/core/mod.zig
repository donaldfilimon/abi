//! AI Core Module
//!
//! Provides fundamental types, configuration, and interfaces for the AI system.
//! These core components are shared across LLMs, agents, and other AI services.

const types = @import("types.zig");
const config = @import("config.zig");

// ============================================================================
// Core Types
// ============================================================================

pub const InstanceId = types.InstanceId;
pub const SessionId = types.SessionId;
pub const ConfidenceLevel = types.ConfidenceLevel;
pub const Confidence = types.Confidence;
pub const EmotionType = types.EmotionType;
pub const EmotionalState = types.EmotionalState;
pub const Role = types.Role;
pub const Message = types.Message;
pub const TrustLevel = types.TrustLevel;
pub const Relationship = types.Relationship;
pub const Topic = types.Topic;
pub const Response = types.Response;
pub const AbbeyError = types.AbbeyError;

// ============================================================================
// Configuration
// ============================================================================

pub const AbbeyConfig = config.AbbeyConfig;
pub const BehaviorConfig = config.BehaviorConfig;
pub const MemoryConfig = config.MemoryConfig;
pub const ReasoningConfig = config.ReasoningConfig;
pub const EmotionConfig = config.EmotionConfig;
pub const LearningConfig = config.LearningConfig;
pub const LLMConfig = config.LLMConfig;
pub const ServerConfig = config.ServerConfig;
pub const DiscordConfig = config.DiscordConfig;
pub const ConfigBuilder = config.ConfigBuilder;

// ============================================================================
// Utility Functions
// ============================================================================

pub const getTimestampNs = types.getTimestampNs;
pub const getTimestampMs = types.getTimestampMs;
pub const getTimestampSec = types.getTimestampSec;
pub const loadConfigFromEnvironment = config.loadFromEnvironment;

test {
    @import("std").testing.refAllDecls(@This());
}
