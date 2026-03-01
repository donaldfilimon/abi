//! Legal Industry Persona Templates
//!
//! Pre-configured persona settings for legal services:
//! - Client Interaction (Abbey): Client communication, case updates
//! - Legal Research (Aviva): Case law analysis, precedent search
//! - Compliance (Abi): Privilege detection, confidentiality enforcement

const std = @import("std");
const persona_config = @import("../config.zig");

/// Legal-specific routing context.
pub const LegalRoutingContext = struct {
    /// Keywords triggering research routing to Aviva.
    research_keywords: []const []const u8 = &default_research_keywords,
    /// Keywords triggering privilege/confidentiality routing to Abi.
    privilege_keywords: []const []const u8 = &default_privilege_keywords,
    /// Whether to enforce attorney-client privilege detection.
    enforce_privilege: bool = true,
    /// Whether to require citation formatting in research responses.
    require_citations: bool = true,
};

const default_research_keywords = [_][]const u8{
    "case law",     "precedent",      "statute",     "regulation",
    "ruling",       "opinion",        "brief",       "motion",
    "jurisdiction", "appeal",         "tort",        "liability",
    "contract law", "constitutional", "due process", "discovery",
};

const default_privilege_keywords = [_][]const u8{
    "privileged",   "confidential", "attorney-client",
    "work product", "settlement",   "mediation",
    "sealed",       "in camera",    "protective order",
};

/// Create a legal-tuned MultiPersonaConfig.
pub fn legalConfig() persona_config.MultiPersonaConfig {
    return .{
        .default_persona = .abbey,
        .enable_dynamic_routing = true,
        .routing_confidence_threshold = 0.5,
        .abbey = .{
            .empathy_level = 0.7,
            .technical_depth = 0.8, // High depth for legal discussions
            .include_reasoning = true,
            .emotion_adaptation = true,
        },
        .aviva = .{
            .directness_level = 0.85,
            .verify_facts = true,
            .cite_sources = true, // Citations are critical in legal
            .include_code_comments = false,
            .include_disclaimers = true, // Legal disclaimers
        },
        .abi = .{
            .enable_sentiment_analysis = true,
            .enable_policy_checking = true,
            .sensitive_topic_detection = true,
            .content_filter_level = .strict,
        },
    };
}

/// Classify legal query intent.
pub const LegalIntent = enum {
    client_communication,
    case_research,
    document_drafting,
    compliance_inquiry,
    privilege_matter,
    billing_admin,
    general_legal,

    pub fn fromContent(content: []const u8) LegalIntent {
        if (containsAny(content, &.{ "privileged", "confidential", "attorney-client", "work product" }))
            return .privilege_matter;
        if (containsAny(content, &.{ "case law", "precedent", "statute", "ruling", "jurisdiction" }))
            return .case_research;
        if (containsAny(content, &.{ "draft", "contract", "agreement", "clause", "template" }))
            return .document_drafting;
        if (containsAny(content, &.{ "compliance", "regulation", "filing", "disclosure" }))
            return .compliance_inquiry;
        if (containsAny(content, &.{ "billing", "retainer", "invoice", "hours" }))
            return .billing_admin;
        if (containsAny(content, &.{ "client", "meeting", "update", "consultation" }))
            return .client_communication;

        return .general_legal;
    }
};

fn containsAny(content: []const u8, keywords: []const []const u8) bool {
    for (keywords) |kw| {
        if (indexOfCaseInsensitive(content, kw) != null) return true;
    }
    return false;
}

fn indexOfCaseInsensitive(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len > haystack.len) return null;
    var i: usize = 0;
    outer: while (i <= haystack.len - needle.len) : (i += 1) {
        for (needle, 0..) |nc, j| {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(nc)) continue :outer;
        }
        return i;
    }
    return null;
}

// ============================================================================
// Tests
// ============================================================================

test "legalConfig creates valid config" {
    const cfg = legalConfig();
    try std.testing.expect(cfg.default_persona == .abbey);
    try std.testing.expect(cfg.aviva.cite_sources);
    try std.testing.expect(cfg.abbey.technical_depth >= 0.8);
}

test "LegalIntent classification" {
    try std.testing.expect(LegalIntent.fromContent("This is attorney-client privileged information") == .privilege_matter);
    try std.testing.expect(LegalIntent.fromContent("Find case law precedent for contract disputes") == .case_research);
    try std.testing.expect(LegalIntent.fromContent("Draft a non-disclosure agreement") == .document_drafting);
    try std.testing.expect(LegalIntent.fromContent("What are the filing requirements?") == .compliance_inquiry);
}

test {
    std.testing.refAllDecls(@This());
}
