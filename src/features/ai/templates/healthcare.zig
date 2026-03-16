//! Healthcare Industry Persona Templates
//!
//! Pre-configured persona settings for healthcare environments:
//! - Medical Assistant (Abbey): Patient inquiries, appointment scheduling
//! - Research Support (Aviva): Data analysis, literature reviews
//! - HIPAA Compliance (Abi): PHI monitoring, regulatory adherence

const std = @import("std");
const persona_config = @import("../config.zig");

/// Healthcare-specific routing rules for the Abi router.
pub const HealthcareRoutingContext = struct {
    /// Keywords that trigger medical assistant routing.
    medical_keywords: []const []const u8 = &default_medical_keywords,
    /// Keywords that trigger research support routing.
    research_keywords: []const []const u8 = &default_research_keywords,
    /// Whether to enforce HIPAA-strict routing.
    hipaa_strict: bool = true,
    /// Require de-identification before research persona access.
    require_de_identification: bool = true,
};

const default_medical_keywords = [_][]const u8{
    "symptom",   "appointment", "medication", "prescription",
    "diagnosis", "treatment",   "checkup",    "referral",
    "insurance", "billing",     "lab result", "test result",
    "follow-up", "side effect", "dosage",     "allergies",
};

const default_research_keywords = [_][]const u8{
    "study",      "research",   "clinical trial",    "meta-analysis",
    "evidence",   "literature", "systematic review", "cohort",
    "randomized", "placebo",    "peer-reviewed",     "publication",
};

/// Create a healthcare-tuned MultiPersonaConfig.
pub fn healthcareConfig() persona_config.MultiPersonaConfig {
    return .{
        .default_persona = .abbey,
        .enable_dynamic_routing = true,
        .routing_confidence_threshold = 0.5,
        .abbey = .{
            .empathy_level = 0.9, // High empathy for patient interactions
            .technical_depth = 0.5, // Accessible explanations
            .include_reasoning = true,
            .emotion_adaptation = true,
        },
        .aviva = .{
            .directness_level = 0.8,
            .verify_facts = true,
            .cite_sources = true, // Important for medical research
            .include_code_comments = false,
        },
        .abi = .{
            .enable_sentiment_analysis = true,
            .enable_policy_checking = true,
            .sensitive_topic_detection = true,
            .content_filter_level = .strict, // Strict for healthcare
        },
    };
}

/// Classify healthcare query intent.
pub const HealthcareIntent = enum {
    patient_inquiry,
    appointment_management,
    medication_question,
    research_request,
    billing_insurance,
    emergency_triage,
    general_health,
    unknown,

    pub fn fromContent(content: []const u8) HealthcareIntent {
        const lower_buf: [512]u8 = undefined;
        _ = lower_buf;
        var has_emergency = false;
        var has_appointment = false;
        var has_medication = false;
        var has_research = false;
        var has_billing = false;

        // Simple keyword matching
        if (containsAny(content, &.{ "emergency", "urgent", "911", "severe pain", "chest pain" }))
            has_emergency = true;
        if (containsAny(content, &.{ "appointment", "schedule", "reschedule", "cancel appointment" }))
            has_appointment = true;
        if (containsAny(content, &.{ "medication", "prescription", "dosage", "refill", "side effect" }))
            has_medication = true;
        if (containsAny(content, &.{ "research", "study", "clinical trial", "evidence", "literature" }))
            has_research = true;
        if (containsAny(content, &.{ "billing", "insurance", "copay", "claim", "coverage" }))
            has_billing = true;

        if (has_emergency) return .emergency_triage;
        if (has_research) return .research_request;
        if (has_appointment) return .appointment_management;
        if (has_medication) return .medication_question;
        if (has_billing) return .billing_insurance;

        return .general_health;
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

test "healthcareConfig creates valid config" {
    const cfg = healthcareConfig();
    try std.testing.expect(cfg.default_persona == .abbey);
    try std.testing.expect(cfg.abbey.empathy_level >= 0.9);
    try std.testing.expect(cfg.abi.content_filter_level == .strict);
}

test "HealthcareIntent classification" {
    try std.testing.expect(HealthcareIntent.fromContent("I need to schedule an appointment") == .appointment_management);
    try std.testing.expect(HealthcareIntent.fromContent("What are the side effects of my medication?") == .medication_question);
    try std.testing.expect(HealthcareIntent.fromContent("Find clinical trial results for treatment X") == .research_request);
    try std.testing.expect(HealthcareIntent.fromContent("I have severe chest pain") == .emergency_triage);
}

test {
    std.testing.refAllDecls(@This());
}
