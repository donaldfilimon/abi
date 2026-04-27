//! Core types for the Multi-Profile AI Assistant system.
//! Standardizes requests, responses, and routing metadata across all profiles.

const std = @import("std");
const core_types = @import("core/types.zig");

// Re-export core types used by abbey, emotion, and other sub-modules
pub const EmotionType = core_types.EmotionType;
pub const EmotionalState = core_types.EmotionalState;
pub const getTimestampNs = core_types.getTimestampNs;
pub const getTimestampMs = core_types.getTimestampMs;
pub const getTimestampSec = core_types.getTimestampSec;
pub const ConfidenceLevel = core_types.ConfidenceLevel;
pub const Confidence = core_types.Confidence;
pub const InstanceId = core_types.InstanceId;
pub const SessionId = core_types.SessionId;
pub const Role = core_types.Role;
pub const TrustLevel = core_types.TrustLevel;
pub const Relationship = core_types.Relationship;
pub const Message = core_types.Message;
pub const Topic = core_types.Topic;
pub const Response = core_types.Response;
pub const AbbeyError = core_types.AbbeyError;

/// AI feature errors visible at the framework level.
pub const FrameworkError = error{
    AiDisabled,
    LlmDisabled,
    EmbeddingsDisabled,
    AgentsDisabled,
    ReasoningDisabled,
    TrainingDisabled,
    ModelNotFound,
    InferenceFailed,
} || AbbeyError;

/// Available profile types in the system.
pub const ProfileType = enum {
    /// General-purpose helpful assistant.
    assistant,
    /// Code-focused programming specialist.
    coder,
    /// Creative writing and content generation specialist.
    writer,
    /// Data analysis and research specialist.
    analyst,
    /// Friendly conversational companion.
    companion,
    /// Documentation specialist.
    docs,
    /// Code and logic reviewer.
    reviewer,
    /// Minimal, direct response model.
    minimal,
    /// Empathetic polymath (Emotional Intelligence + Technical Depth).
    abbey,
    /// Direct expert (Concise + Factual).
    aviva,
    /// Adaptive moderator and router.
    abi,
    /// Iterative agent loop specialist.
    ralph,
    /// Locally-trained assistant based on gpt-oss.
    ava,
};

/// Standardized request structure for all profiles.
pub const ProfileRequest = struct {
    /// The user input text.
    content: []const u8,
    /// Optional session identifier.
    session_id: ?[]const u8 = null,
    /// Optional user identifier.
    user_id: ?[]const u8 = null,
    /// Optional forced profile selection (bypasses routing).
    preferred_profile: ?ProfileType = null,
    /// Optional system instruction/context for the profile.
    system_instruction: ?[]const u8 = null,
    /// Optional maximum tokens for the response.
    max_tokens: ?u32 = null,
    /// Optional temperature override for generation.
    temperature: ?f32 = null,
    /// Current emotional context of the conversation.
    emotional_context: core_types.EmotionalState = .{},
    /// Additional metadata for the request.
    metadata: std.StringHashMapUnmanaged([]const u8) = .empty,

    pub fn deinit(self: *ProfileRequest, allocator: std.mem.Allocator) void {
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit(allocator);
    }
};

/// Ordered list of all profile types for enumeration and UI.
pub const ALL_PROFILE_TYPES = blk: {
    const fields = std.meta.fields(ProfileType);
    var values: [fields.len]ProfileType = undefined;
    for (fields, 0..) |field, i| {
        values[i] = @enumFromInt(field.value);
    }
    break :blk values;
};

/// Returns a slice of all known profile types.
pub fn allProfileTypes() []const ProfileType {
    return &ALL_PROFILE_TYPES;
}

/// Standardized response structure from any profile.
pub const ProfileResponse = struct {
    /// Generated response text.
    content: []const u8,
    /// The profile that generated this response.
    profile: ProfileType,
    /// Confidence score for the response (0.0 - 1.0).
    confidence: f32,
    /// Suggested emotional tone for the response.
    emotional_tone: ?core_types.EmotionType = null,
    /// Steps taken during reasoning (if enabled).
    reasoning_chain: ?[]const ReasoningStep = null,
    /// Code blocks extracted or generated.
    code_blocks: ?[]const CodeBlock = null,
    /// Source references used for grounding.
    references: ?[]const Source = null,
    /// Time taken to generate the response in milliseconds.
    generation_time_ms: u64 = 0,

    pub fn deinit(self: *ProfileResponse, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
        if (self.reasoning_chain) |chain| {
            for (chain) |step| step.deinit(allocator);
            allocator.free(chain);
        }
        if (self.code_blocks) |blocks| {
            for (blocks) |block| block.deinit(allocator);
            allocator.free(blocks);
        }
        if (self.references) |sources| {
            for (sources) |source| source.deinit(allocator);
            allocator.free(sources);
        }
    }
};

/// Decision metadata from the Abi router.
pub const RoutingDecision = struct {
    /// The profile selected for the request.
    selected_profile: ProfileType,
    /// Confidence in this routing choice (0.0 - 1.0).
    confidence: f32,
    /// Detected emotional state from input.
    emotional_context: core_types.EmotionalState,
    /// Safety and policy flags.
    policy_flags: PolicyFlags = .{},
    /// Human-readable reason for the routing choice.
    routing_reason: []const u8,

    pub fn deinit(self: *RoutingDecision, allocator: std.mem.Allocator) void {
        allocator.free(self.routing_reason);
    }
};

/// Safety and policy compliance flags.
pub const PolicyFlags = struct {
    is_safe: bool = true,
    requires_moderation: bool = false,
    sensitive_topic: bool = false,
    pii_detected: bool = false,
    violation_details: ?[]const u8 = null,

    pub fn deinit(self: *PolicyFlags, allocator: std.mem.Allocator) void {
        if (self.violation_details) |details| {
            allocator.free(details);
        }
    }
};

/// A single step in a profile's reasoning process.
pub const ReasoningStep = struct {
    title: []const u8,
    explanation: []const u8,
    confidence: f32 = 1.0,

    pub fn deinit(self: *const ReasoningStep, allocator: std.mem.Allocator) void {
        allocator.free(self.title);
        allocator.free(self.explanation);
    }
};

/// A block of code in a response.
pub const CodeBlock = struct {
    language: []const u8,
    code: []const u8,
    explanation: ?[]const u8 = null,

    pub fn deinit(self: *const CodeBlock, allocator: std.mem.Allocator) void {
        allocator.free(self.language);
        allocator.free(self.code);
        if (self.explanation) |exp| allocator.free(exp);
    }
};

/// A source reference for grounding facts.
pub const Source = struct {
    title: []const u8,
    url: ?[]const u8 = null,
    snippet: ?[]const u8 = null,
    confidence: f32 = 1.0,

    pub fn deinit(self: *const Source, allocator: std.mem.Allocator) void {
        allocator.free(self.title);
        if (self.url) |u| allocator.free(u);
        if (self.snippet) |s| allocator.free(s);
    }
};

/// Common interface that all profiles must implement.
pub const ProfileInterface = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        process: *const fn (ctx: *anyopaque, request: ProfileRequest) anyerror!ProfileResponse,
        getName: *const fn (ctx: *anyopaque) []const u8,
        getType: *const fn (ctx: *anyopaque) ProfileType,
    };

    pub fn process(self: ProfileInterface, request: ProfileRequest) !ProfileResponse {
        return self.vtable.process(self.ptr, request);
    }

    pub fn getName(self: ProfileInterface) []const u8 {
        return self.vtable.getName(self.ptr);
    }

    pub fn getType(self: ProfileInterface) ProfileType {
        return self.vtable.getType(self.ptr);
    }
};

test {
    std.testing.refAllDecls(@This());
}
