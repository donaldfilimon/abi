//! Finance Industry Persona Templates
//!
//! Pre-configured persona settings for financial services:
//! - Customer Service (Abbey): Account inquiries, product recommendations
//! - Fraud Detection (Aviva): Transaction analysis, anomaly detection
//! - Regulatory Compliance (Abi): BSA/AML monitoring, financial regulations

const std = @import("std");
const persona_config = @import("../config.zig");

/// Finance-specific routing context.
pub const FinanceRoutingContext = struct {
    /// Keywords triggering fraud detection routing.
    fraud_keywords: []const []const u8 = &default_fraud_keywords,
    /// Keywords triggering compliance routing.
    compliance_keywords: []const []const u8 = &default_compliance_keywords,
    /// Whether to always route transaction queries to Aviva.
    route_transactions_to_aviva: bool = true,
    /// Whether to add financial disclaimers to responses.
    require_disclaimers: bool = true,
};

const default_fraud_keywords = [_][]const u8{
    "fraud",            "suspicious", "unauthorized", "stolen",
    "identity theft",   "phishing",   "scam",         "compromised",
    "unusual activity", "dispute",    "chargeback",   "breach",
};

const default_compliance_keywords = [_][]const u8{
    "regulation", "compliance", "audit",      "reporting",
    "AML",        "BSA",        "KYC",        "sanctions",
    "FINRA",      "SEC",        "disclosure", "fiduciary",
};

/// Create a finance-tuned MultiPersonaConfig.
pub fn financeConfig() persona_config.MultiPersonaConfig {
    return .{
        .default_persona = .abbey,
        .enable_dynamic_routing = true,
        .routing_confidence_threshold = 0.55,
        .abbey = .{
            .empathy_level = 0.7,
            .technical_depth = 0.6,
            .include_reasoning = true,
            .emotion_adaptation = true,
        },
        .aviva = .{
            .directness_level = 0.95, // Very direct for financial data
            .verify_facts = true,
            .cite_sources = true,
            .include_disclaimers = true, // Financial disclaimers required
            .include_code_comments = false,
        },
        .abi = .{
            .enable_sentiment_analysis = true,
            .enable_policy_checking = true,
            .sensitive_topic_detection = true,
            .content_filter_level = .strict,
        },
    };
}

/// Classify finance query intent.
pub const FinanceIntent = enum {
    account_inquiry,
    transaction_query,
    fraud_report,
    investment_advice,
    loan_mortgage,
    compliance_question,
    product_recommendation,
    general_banking,

    pub fn fromContent(content: []const u8) FinanceIntent {
        if (containsAny(content, &.{ "fraud", "suspicious", "unauthorized", "stolen", "scam" }))
            return .fraud_report;
        if (containsAny(content, &.{ "compliance", "regulation", "audit", "AML", "BSA", "KYC" }))
            return .compliance_question;
        if (containsAny(content, &.{ "invest", "portfolio", "stock", "bond", "dividend", "return" }))
            return .investment_advice;
        if (containsAny(content, &.{ "loan", "mortgage", "interest rate", "refinance", "credit" }))
            return .loan_mortgage;
        if (containsAny(content, &.{ "transaction", "transfer", "payment", "wire", "ACH" }))
            return .transaction_query;
        if (containsAny(content, &.{ "account", "balance", "statement", "checking", "savings" }))
            return .account_inquiry;

        return .general_banking;
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

test "financeConfig creates valid config" {
    const cfg = financeConfig();
    try std.testing.expect(cfg.default_persona == .abbey);
    try std.testing.expect(cfg.aviva.include_disclaimers);
    try std.testing.expect(cfg.abi.content_filter_level == .strict);
}

test "FinanceIntent classification" {
    try std.testing.expect(FinanceIntent.fromContent("I see unauthorized transactions on my account") == .fraud_report);
    try std.testing.expect(FinanceIntent.fromContent("What are the AML compliance requirements?") == .compliance_question);
    try std.testing.expect(FinanceIntent.fromContent("I want to invest in index funds") == .investment_advice);
    try std.testing.expect(FinanceIntent.fromContent("Transfer $500 to my savings") == .transaction_query);
}

test {
    std.testing.refAllDecls(@This());
}
