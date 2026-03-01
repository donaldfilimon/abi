//! Industry Persona Templates Module
//!
//! Provides pre-configured persona settings for specific industries:
//! - Healthcare: HIPAA-compliant, patient-focused configurations
//! - Finance: BSA/AML-aware, fraud detection configurations
//! - Legal: Privilege-aware, citation-focused configurations
//!
//! Each template produces a `MultiPersonaConfig` tuned for its industry.

const std = @import("std");
const persona_config = @import("../config.zig");

pub const healthcare = @import("healthcare.zig");
pub const finance = @import("finance.zig");
pub const legal = @import("legal.zig");

// Re-export key types
pub const HealthcareRoutingContext = healthcare.HealthcareRoutingContext;
pub const HealthcareIntent = healthcare.HealthcareIntent;
pub const FinanceRoutingContext = finance.FinanceRoutingContext;
pub const FinanceIntent = finance.FinanceIntent;
pub const LegalRoutingContext = legal.LegalRoutingContext;
pub const LegalIntent = legal.LegalIntent;

/// Supported industry templates.
pub const Industry = enum {
    healthcare,
    finance,
    legal,
    custom,
};

/// An industry template encapsulating a complete persona configuration.
pub const IndustryTemplate = struct {
    industry: Industry,
    name: []const u8,
    description: []const u8,
    config: persona_config.MultiPersonaConfig,

    /// List all available industry templates.
    pub fn listAll() [3]IndustryTemplate {
        return .{
            .{
                .industry = .healthcare,
                .name = "Healthcare",
                .description = "HIPAA-compliant patient interaction, medical research, and clinical compliance",
                .config = healthcare.healthcareConfig(),
            },
            .{
                .industry = .finance,
                .name = "Finance",
                .description = "BSA/AML-aware customer service, fraud detection, and regulatory compliance",
                .config = finance.financeConfig(),
            },
            .{
                .industry = .legal,
                .name = "Legal",
                .description = "Privilege-aware client interaction, legal research, and confidentiality enforcement",
                .config = legal.legalConfig(),
            },
        };
    }

    /// Get a template by industry.
    pub fn getTemplate(industry: Industry) ?IndustryTemplate {
        const all = listAll();
        for (&all) |*tmpl| {
            if (tmpl.industry == industry) return tmpl.*;
        }
        return null;
    }
};

/// Apply an industry template, returning the configured MultiPersonaConfig.
pub fn applyTemplate(industry: Industry) ?persona_config.MultiPersonaConfig {
    const template = IndustryTemplate.getTemplate(industry) orelse return null;
    return template.config;
}

// ============================================================================
// Tests
// ============================================================================

test "IndustryTemplate listAll" {
    const all = IndustryTemplate.listAll();
    try std.testing.expect(all.len == 3);
    try std.testing.expect(all[0].industry == .healthcare);
    try std.testing.expect(all[1].industry == .finance);
    try std.testing.expect(all[2].industry == .legal);
}

test "IndustryTemplate getTemplate" {
    const tmpl = IndustryTemplate.getTemplate(.healthcare);
    try std.testing.expect(tmpl != null);
    try std.testing.expectEqualStrings("Healthcare", tmpl.?.name);
}

test "applyTemplate returns config" {
    const cfg = applyTemplate(.finance);
    try std.testing.expect(cfg != null);
    try std.testing.expect(cfg.?.aviva.include_disclaimers);
}

test "applyTemplate custom returns null" {
    const cfg = applyTemplate(.custom);
    try std.testing.expect(cfg == null);
}

test {
    std.testing.refAllDecls(@This());
}
