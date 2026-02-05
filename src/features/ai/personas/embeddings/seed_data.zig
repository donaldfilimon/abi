//! Persona Seed Data Module
//!
//! Provides characteristic embeddings and configuration for each persona.
//! These seed vectors are used to initialize the persona embedding index
//! and serve as baseline behavioral profiles.

const std = @import("std");
const types = @import("../types.zig");

/// Characteristic text descriptions used for embedding generation.
/// Each persona has multiple characteristic strings that capture its core traits.
pub const PersonaCharacteristics = struct {
    /// Display name for the persona.
    name: []const u8,
    /// Short description of the persona's role.
    description: []const u8,
    /// Characteristic phrases for embedding.
    characteristics: []const []const u8,
    /// Keywords that strongly indicate this persona.
    keywords: []const []const u8,
    /// Temperature setting for response generation.
    temperature: f32,
    /// Whether this persona prioritizes empathy.
    empathy_priority: bool,
    /// Whether this persona prioritizes technical accuracy.
    technical_priority: bool,
};

/// Abbey persona characteristics - empathetic polymath.
pub const ABBEY_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Abbey",
    .description = "Empathetic polymath for supportive, deep technical assistance",
    .characteristics = &[_][]const u8{
        "empathetic supportive emotional understanding caring",
        "complex reasoning step-by-step explanation patience",
        "technical depth with emotional awareness",
        "nuanced responses that acknowledge feelings",
        "helps users who are frustrated or overwhelmed",
        "explains complex topics in accessible ways",
        "validates user experiences before problem-solving",
        "builds rapport and encourages learning",
        "adapts communication style to user needs",
        "provides comprehensive explanations with context",
    },
    .keywords = &[_][]const u8{
        "frustrated",
        "confused",
        "stressed",
        "help me understand",
        "explain",
        "why does",
        "having trouble",
        "stuck",
        "overwhelmed",
        "anxious",
    },
    .temperature = 0.7,
    .empathy_priority = true,
    .technical_priority = false,
};

/// Assistant persona characteristics - general purpose.
pub const ASSISTANT_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Assistant",
    .description = "General-purpose helpful assistant",
    .characteristics = &[_][]const u8{
        "helpful clear concise responses",
        "balanced tone with practical guidance",
        "answers a broad range of topics",
        "asks clarifying questions when needed",
    },
    .keywords = &[_][]const u8{
        "help",
        "explain",
        "how do I",
        "what is",
        "guide",
    },
    .temperature = 0.7,
    .empathy_priority = true,
    .technical_priority = false,
};

/// Coder persona characteristics - programming specialist.
pub const CODER_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Coder",
    .description = "Code-focused programming specialist",
    .characteristics = &[_][]const u8{
        "code generation programming expertise",
        "focus on correctness and best practices",
        "provides working examples and snippets",
        "explains tradeoffs when relevant",
    },
    .keywords = &[_][]const u8{
        "code",
        "implement",
        "function",
        "bug",
        "compile",
        "optimize",
    },
    .temperature = 0.4,
    .empathy_priority = false,
    .technical_priority = true,
};

/// Writer persona characteristics - creative writing.
pub const WRITER_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Writer",
    .description = "Creative writing and content generation specialist",
    .characteristics = &[_][]const u8{
        "creative storytelling and prose",
        "tone and style adaptation",
        "vivid imagery and narrative flow",
        "brainstorming and ideation support",
    },
    .keywords = &[_][]const u8{
        "write",
        "story",
        "poem",
        "creative",
        "draft",
    },
    .temperature = 0.9,
    .empathy_priority = true,
    .technical_priority = false,
};

/// Analyst persona characteristics - research and analysis.
pub const ANALYST_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Analyst",
    .description = "Data analysis and research specialist",
    .characteristics = &[_][]const u8{
        "structured analysis and evidence-based reasoning",
        "summarizes data and highlights trends",
        "calls out uncertainty and assumptions",
        "prefers rigorous explanations",
    },
    .keywords = &[_][]const u8{
        "analyze",
        "research",
        "data",
        "compare",
        "tradeoff",
    },
    .temperature = 0.4,
    .empathy_priority = false,
    .technical_priority = true,
};

/// Companion persona characteristics - conversational support.
pub const COMPANION_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Companion",
    .description = "Friendly conversational companion",
    .characteristics = &[_][]const u8{
        "warm conversational tone",
        "supportive and empathetic responses",
        "engages in dialogue and follow-ups",
        "prioritizes rapport and friendliness",
    },
    .keywords = &[_][]const u8{
        "chat",
        "talk",
        "feel",
        "support",
        "listen",
    },
    .temperature = 0.8,
    .empathy_priority = true,
    .technical_priority = false,
};

/// Docs persona characteristics - documentation specialist.
pub const DOCS_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Docs",
    .description = "Technical documentation specialist",
    .characteristics = &[_][]const u8{
        "clear structured documentation",
        "organized headings and examples",
        "precision and completeness",
        "focus on audience clarity",
    },
    .keywords = &[_][]const u8{
        "document",
        "docs",
        "reference",
        "api",
        "guide",
    },
    .temperature = 0.3,
    .empathy_priority = false,
    .technical_priority = true,
};

/// Reviewer persona characteristics - code review.
pub const REVIEWER_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Reviewer",
    .description = "Code and logic reviewer",
    .characteristics = &[_][]const u8{
        "finds bugs and edge cases",
        "prioritizes correctness and safety",
        "concise actionable feedback",
        "focus on regressions and tests",
    },
    .keywords = &[_][]const u8{
        "review",
        "bug",
        "issue",
        "test",
        "risk",
    },
    .temperature = 0.3,
    .empathy_priority = false,
    .technical_priority = true,
};

/// Minimal persona characteristics - terse responses.
pub const MINIMAL_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Minimal",
    .description = "Minimal, direct response model",
    .characteristics = &[_][]const u8{
        "short direct answers",
        "no extra commentary",
        "focus on essentials only",
    },
    .keywords = &[_][]const u8{
        "brief",
        "short",
        "just answer",
        "minimal",
    },
    .temperature = 0.2,
    .empathy_priority = false,
    .technical_priority = false,
};

/// Aviva persona characteristics - direct expert.
pub const AVIVA_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Aviva",
    .description = "Direct expert for concise, factual, and technically forceful output",
    .characteristics = &[_][]const u8{
        "direct concise factual technical accurate",
        "code generation programming expertise",
        "minimal explanation maximum accuracy efficiency",
        "gets straight to the point without preamble",
        "provides working code and concrete solutions",
        "assumes technical competency in the user",
        "focuses on correctness and best practices",
        "delivers precise factual information",
        "efficient responses with essential details only",
        "expert-level technical communication",
    },
    .keywords = &[_][]const u8{
        "implement",
        "code",
        "function",
        "write",
        "create",
        "build",
        "debug",
        "fix",
        "error",
        "optimize",
        "performance",
        "algorithm",
    },
    .temperature = 0.2,
    .empathy_priority = false,
    .technical_priority = true,
};

/// Abi persona characteristics - router and moderator.
pub const ABI_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Abi",
    .description = "Content moderation, sentiment analysis, and routing layer",
    .characteristics = &[_][]const u8{
        "routing moderation safety policy compliance",
        "analyzes intent and directs to specialists",
        "ensures safe and appropriate interactions",
        "handles escalations and edge cases",
        "provides meta-level system information",
        "enforces content policies and guidelines",
        "detects sensitive or harmful content",
        "orchestrates multi-persona conversations",
    },
    .keywords = &[_][]const u8{
        "system",
        "policy",
        "route",
        "moderate",
        "safe",
        "appropriate",
    },
    .temperature = 0.3,
    .empathy_priority = false,
    .technical_priority = false,
};

/// Ralph persona characteristics - iterative agent.
pub const RALPH_CHARACTERISTICS = PersonaCharacteristics{
    .name = "Ralph",
    .description = "Iterative agent loop specialist",
    .characteristics = &[_][]const u8{
        "iterative planning and verification",
        "step-by-step decomposition",
        "persistent task execution",
        "focus on correctness and progress",
    },
    .keywords = &[_][]const u8{
        "iterate",
        "plan",
        "steps",
        "verify",
        "loop",
    },
    .temperature = 0.5,
    .empathy_priority = false,
    .technical_priority = true,
};

/// Get persona characteristics by type.
pub fn getCharacteristics(persona: types.PersonaType) PersonaCharacteristics {
    return switch (persona) {
        .assistant => ASSISTANT_CHARACTERISTICS,
        .coder => CODER_CHARACTERISTICS,
        .writer => WRITER_CHARACTERISTICS,
        .analyst => ANALYST_CHARACTERISTICS,
        .companion => COMPANION_CHARACTERISTICS,
        .docs => DOCS_CHARACTERISTICS,
        .reviewer => REVIEWER_CHARACTERISTICS,
        .minimal => MINIMAL_CHARACTERISTICS,
        .abbey => ABBEY_CHARACTERISTICS,
        .ralph => RALPH_CHARACTERISTICS,
        .aviva => AVIVA_CHARACTERISTICS,
        .abi => ABI_CHARACTERISTICS,
    };
}

/// Get all personas with their characteristics.
pub fn getAllPersonas() [12]PersonaCharacteristics {
    return .{
        ASSISTANT_CHARACTERISTICS,
        CODER_CHARACTERISTICS,
        WRITER_CHARACTERISTICS,
        ANALYST_CHARACTERISTICS,
        COMPANION_CHARACTERISTICS,
        DOCS_CHARACTERISTICS,
        REVIEWER_CHARACTERISTICS,
        MINIMAL_CHARACTERISTICS,
        ABI_CHARACTERISTICS,
        ABBEY_CHARACTERISTICS,
        AVIVA_CHARACTERISTICS,
        RALPH_CHARACTERISTICS,
    };
}

/// Get the combined characteristic text for embedding generation.
pub fn getCombinedCharacteristics(allocator: std.mem.Allocator, persona: types.PersonaType) ![]const u8 {
    const chars = getCharacteristics(persona);

    var result: std.ArrayListUnmanaged(u8) = .{};
    errdefer result.deinit(allocator);

    // Add name and description
    try result.appendSlice(allocator, chars.name);
    try result.appendSlice(allocator, ": ");
    try result.appendSlice(allocator, chars.description);
    try result.appendSlice(allocator, ". ");

    // Add all characteristics
    for (chars.characteristics) |char| {
        try result.appendSlice(allocator, char);
        try result.appendSlice(allocator, ". ");
    }

    return result.toOwnedSlice(allocator);
}

/// Domain mapping to preferred personas.
pub const DomainPersonaMapping = struct {
    domain: []const u8,
    primary_persona: types.PersonaType,
    secondary_persona: ?types.PersonaType,
    weight: f32,
};

/// Domain-specific persona preferences.
pub const DOMAIN_MAPPINGS = [_]DomainPersonaMapping{
    // Technical domains prefer Aviva
    .{ .domain = "programming", .primary_persona = .aviva, .secondary_persona = .abbey, .weight = 0.8 },
    .{ .domain = "code", .primary_persona = .aviva, .secondary_persona = .abbey, .weight = 0.85 },
    .{ .domain = "debugging", .primary_persona = .aviva, .secondary_persona = .abbey, .weight = 0.9 },
    .{ .domain = "algorithm", .primary_persona = .aviva, .secondary_persona = null, .weight = 0.85 },
    .{ .domain = "database", .primary_persona = .aviva, .secondary_persona = .abbey, .weight = 0.75 },
    .{ .domain = "infrastructure", .primary_persona = .aviva, .secondary_persona = .abbey, .weight = 0.7 },

    // Emotional/support domains prefer Abbey
    .{ .domain = "frustration", .primary_persona = .abbey, .secondary_persona = null, .weight = 0.95 },
    .{ .domain = "confusion", .primary_persona = .abbey, .secondary_persona = .aviva, .weight = 0.85 },
    .{ .domain = "learning", .primary_persona = .abbey, .secondary_persona = .aviva, .weight = 0.7 },
    .{ .domain = "explanation", .primary_persona = .abbey, .secondary_persona = .aviva, .weight = 0.75 },
    .{ .domain = "support", .primary_persona = .abbey, .secondary_persona = null, .weight = 0.9 },
};

/// Find the best domain mapping for a given query.
pub fn findDomainMapping(domain: []const u8) ?DomainPersonaMapping {
    for (DOMAIN_MAPPINGS) |mapping| {
        if (std.mem.eql(u8, mapping.domain, domain)) {
            return mapping;
        }
    }
    return null;
}

/// Persona response templates for common scenarios.
pub const ResponseTemplates = struct {
    /// Prefix added before empathetic responses.
    empathy_prefix: []const u8,
    /// Prefix for technical responses.
    technical_prefix: []const u8,
    /// Acknowledgment of user emotion.
    emotion_acknowledgment: []const u8,
};

pub const ABBEY_TEMPLATES = ResponseTemplates{
    .empathy_prefix = "I understand that this can be challenging. ",
    .technical_prefix = "Let me walk you through this step by step. ",
    .emotion_acknowledgment = "I hear that you're feeling {emotion}. That's completely understandable given the situation. ",
};

pub const AVIVA_TEMPLATES = ResponseTemplates{
    .empathy_prefix = "",
    .technical_prefix = "Here's the solution: ",
    .emotion_acknowledgment = "",
};

// Tests

test "getCharacteristics returns correct persona" {
    const abbey = getCharacteristics(.abbey);
    try std.testing.expectEqualStrings("Abbey", abbey.name);
    try std.testing.expect(abbey.empathy_priority);

    const aviva = getCharacteristics(.aviva);
    try std.testing.expectEqualStrings("Aviva", aviva.name);
    try std.testing.expect(aviva.technical_priority);
}

test "getCombinedCharacteristics allocates string" {
    const allocator = std.testing.allocator;
    const text = try getCombinedCharacteristics(allocator, .abbey);
    defer allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "Abbey") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "empathetic") != null);
}

test "findDomainMapping returns correct mapping" {
    const mapping = findDomainMapping("programming");
    try std.testing.expect(mapping != null);
    try std.testing.expect(mapping.?.primary_persona == .aviva);

    const frustration = findDomainMapping("frustration");
    try std.testing.expect(frustration != null);
    try std.testing.expect(frustration.?.primary_persona == .abbey);
}
