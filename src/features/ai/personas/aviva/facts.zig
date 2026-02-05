//! Aviva Fact Checker Module
//!
//! Provides fact verification and confidence scoring for Aviva's responses.
//! Ensures accuracy and helps identify claims that need qualification.
//!
//! Features:
//! - Claim extraction and analysis
//! - Confidence scoring for facts
//! - Uncertainty detection
//! - Qualification suggestions

const std = @import("std");
const classifier = @import("classifier.zig");
const knowledge = @import("knowledge.zig");

/// A factual claim extracted from content.
pub const Claim = struct {
    /// The claim text.
    text: []const u8,
    /// Type of claim.
    claim_type: ClaimType,
    /// Confidence in this claim (0.0 - 1.0).
    confidence: f32,
    /// Whether this needs verification.
    needs_verification: bool,
    /// Supporting evidence if available.
    evidence: ?[]const u8 = null,
    /// Suggested qualification if uncertain.
    qualification: ?[]const u8 = null,
};

/// Types of factual claims.
pub const ClaimType = enum {
    /// Definitional claim ("X is Y").
    definition,
    /// Numerical claim (statistics, measurements).
    numerical,
    /// Temporal claim (dates, timelines).
    temporal,
    /// Comparison claim ("X is better/faster than Y").
    comparison,
    /// Causal claim ("X causes Y").
    causal,
    /// Best practice or recommendation.
    recommendation,
    /// Procedural claim (how to do something).
    procedural,
    /// Attribution claim ("According to X").
    attribution,
    /// General statement.
    general,

    pub fn getDefaultConfidence(self: ClaimType) f32 {
        return switch (self) {
            .definition => 0.85,
            .numerical => 0.7, // Numbers can become outdated
            .temporal => 0.75,
            .comparison => 0.7,
            .causal => 0.65,
            .recommendation => 0.75,
            .procedural => 0.8,
            .attribution => 0.6, // Needs verification
            .general => 0.75,
        };
    }
};

/// Result of fact checking.
pub const FactCheckResult = struct {
    allocator: std.mem.Allocator,
    /// Extracted claims.
    claims: std.ArrayListUnmanaged(Claim),
    /// Overall confidence score.
    overall_confidence: f32,
    /// Number of claims needing verification.
    verification_needed_count: usize,
    /// Suggested response qualifications.
    qualifications: std.ArrayListUnmanaged([]const u8),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .claims = .{},
            .overall_confidence = 1.0,
            .verification_needed_count = 0,
            .qualifications = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.claims.deinit(self.allocator);
        self.qualifications.deinit(self.allocator);
    }

    /// Get high-confidence claims.
    pub fn getHighConfidence(self: *const Self, threshold: f32) []const Claim {
        var count: usize = 0;
        for (self.claims.items) |claim| {
            if (claim.confidence >= threshold) count += 1;
        }
        // Return slice of high-confidence claims
        var result_count: usize = 0;
        for (self.claims.items) |claim| {
            if (claim.confidence >= threshold) result_count += 1 else break;
        }
        return self.claims.items[0..result_count];
    }
};

/// Patterns that indicate uncertainty.
const UNCERTAINTY_PATTERNS = [_][]const u8{
    "might",
    "may",
    "could",
    "possibly",
    "perhaps",
    "probably",
    "likely",
    "unlikely",
    "seems",
    "appears",
    "suggests",
    "indicates",
    "approximately",
    "roughly",
    "around",
    "about",
    "estimated",
    "believed",
    "thought to",
    "reportedly",
};

/// Patterns indicating high confidence.
const CERTAINTY_PATTERNS = [_][]const u8{
    "always",
    "never",
    "definitely",
    "certainly",
    "guaranteed",
    "proven",
    "confirmed",
    "established",
    "verified",
    "documented",
    "officially",
};

/// Claim indicator patterns.
const ClaimPattern = struct {
    patterns: []const []const u8,
    claim_type: ClaimType,
};

const CLAIM_PATTERNS = [_]ClaimPattern{
    .{
        .patterns = &[_][]const u8{ " is a ", " is an ", " are ", " was ", " were ", " means ", "defined as" },
        .claim_type = .definition,
    },
    .{
        .patterns = &[_][]const u8{ "percent", "%", "million", "billion", "thousand", "average", "rate" },
        .claim_type = .numerical,
    },
    .{
        .patterns = &[_][]const u8{ "in 19", "in 20", "since", "until", "before", "after", "during", "year" },
        .claim_type = .temporal,
    },
    .{
        .patterns = &[_][]const u8{ "better than", "faster than", "slower than", "more than", "less than", "compared to" },
        .claim_type = .comparison,
    },
    .{
        .patterns = &[_][]const u8{ "causes", "leads to", "results in", "because", "therefore", "thus" },
        .claim_type = .causal,
    },
    .{
        .patterns = &[_][]const u8{ "should", "recommended", "best practice", "prefer", "avoid", "use" },
        .claim_type = .recommendation,
    },
    .{
        .patterns = &[_][]const u8{ "first", "then", "next", "finally", "step", "to do" },
        .claim_type = .procedural,
    },
    .{
        .patterns = &[_][]const u8{ "according to", "says", "states", "reports", "claims" },
        .claim_type = .attribution,
    },
};

/// Configuration for the fact checker.
pub const FactCheckerConfig = struct {
    /// Minimum confidence to accept without qualification.
    min_unqualified_confidence: f32 = 0.8,
    /// Whether to add qualifications automatically.
    auto_qualify: bool = true,
    /// Whether to extract individual claims.
    extract_claims: bool = true,
    /// Maximum claims to extract.
    max_claims: usize = 10,
};

/// Fact checker for Aviva.
pub const FactChecker = struct {
    allocator: std.mem.Allocator,
    config: FactCheckerConfig,

    const Self = @This();

    /// Initialize the fact checker.
    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithConfig(allocator, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: FactCheckerConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Check facts in content.
    pub fn check(self: *Self, content: []const u8) !FactCheckResult {
        var result = FactCheckResult.init(self.allocator);
        errdefer result.deinit();

        // Convert to lowercase for matching
        var lower_buf: [8192]u8 = undefined;
        const content_lower = self.toLowerBounded(content, &lower_buf);

        // Extract claims if enabled
        if (self.config.extract_claims) {
            try self.extractClaims(content, content_lower, &result);
        }

        // Calculate overall confidence
        if (result.claims.items.len > 0) {
            var total_conf: f32 = 0;
            for (result.claims.items) |claim| {
                total_conf += claim.confidence;
                if (claim.needs_verification) {
                    result.verification_needed_count += 1;
                }
            }
            result.overall_confidence = total_conf / @as(f32, @floatFromInt(result.claims.items.len));
        }

        // Generate qualifications if needed
        if (self.config.auto_qualify) {
            try self.generateQualifications(&result);
        }

        return result;
    }

    /// Extract claims from content.
    fn extractClaims(
        self: *Self,
        content: []const u8,
        content_lower: []const u8,
        result: *FactCheckResult,
    ) !void {
        // Split into sentences (simple heuristic)
        var sentences = std.mem.splitAny(u8, content, ".!?");

        while (sentences.next()) |sentence| {
            if (result.claims.items.len >= self.config.max_claims) break;

            const trimmed = std.mem.trim(u8, sentence, " \t\n\r");
            if (trimmed.len < 10) continue; // Skip very short sentences

            // Determine claim type
            const claim_type = self.detectClaimType(trimmed, content_lower);

            // Calculate confidence
            var confidence = claim_type.getDefaultConfidence();

            // Adjust for uncertainty markers
            confidence = self.adjustConfidenceForMarkers(trimmed, confidence);

            // Determine if verification needed
            const needs_verification = confidence < self.config.min_unqualified_confidence;

            try result.claims.append(result.allocator, .{
                .text = trimmed,
                .claim_type = claim_type,
                .confidence = confidence,
                .needs_verification = needs_verification,
                .qualification = if (needs_verification) self.suggestQualification(claim_type) else null,
            });
        }
    }

    /// Detect the type of claim.
    fn detectClaimType(self: *const Self, sentence: []const u8, content_lower: []const u8) ClaimType {
        _ = self;
        _ = content_lower;

        // Check patterns
        for (CLAIM_PATTERNS) |pattern| {
            for (pattern.patterns) |p| {
                if (std.mem.indexOf(u8, sentence, p) != null) {
                    return pattern.claim_type;
                }
            }
        }

        return .general;
    }

    /// Adjust confidence based on uncertainty/certainty markers.
    fn adjustConfidenceForMarkers(self: *const Self, text: []const u8, base_confidence: f32) f32 {
        _ = self;
        var confidence = base_confidence;

        // Check for uncertainty markers
        for (UNCERTAINTY_PATTERNS) |pattern| {
            if (std.mem.indexOf(u8, text, pattern) != null) {
                confidence *= 0.9; // Reduce confidence
                break;
            }
        }

        // Check for certainty markers
        for (CERTAINTY_PATTERNS) |pattern| {
            if (std.mem.indexOf(u8, text, pattern) != null) {
                // Be careful with absolute claims
                if (std.mem.indexOf(u8, text, "always") != null or
                    std.mem.indexOf(u8, text, "never") != null)
                {
                    confidence *= 0.85; // Absolute claims are risky
                } else {
                    confidence = @min(1.0, confidence * 1.05);
                }
                break;
            }
        }

        return confidence;
    }

    /// Suggest a qualification for uncertain claims.
    fn suggestQualification(self: *const Self, claim_type: ClaimType) []const u8 {
        _ = self;
        return switch (claim_type) {
            .numerical => "Note: Specific numbers may vary or be outdated.",
            .temporal => "Dates should be verified for accuracy.",
            .comparison => "Comparisons may depend on specific use cases.",
            .causal => "This causal relationship may have exceptions.",
            .attribution => "This claim should be verified with the original source.",
            .recommendation => "Best practices may vary by context.",
            else => "This claim may need verification.",
        };
    }

    /// Generate qualifications for the result.
    fn generateQualifications(self: *Self, result: *FactCheckResult) !void {
        // Add qualifications for claims that need verification
        var added_types = std.AutoHashMap(ClaimType, void).init(self.allocator);
        defer added_types.deinit();

        for (result.claims.items) |claim| {
            if (claim.needs_verification) {
                if (!added_types.contains(claim.claim_type)) {
                    if (claim.qualification) |qual| {
                        try result.qualifications.append(result.allocator, qual);
                        try added_types.put(claim.claim_type, {});
                    }
                }
            }
        }

        // Add overall qualification if confidence is low
        if (result.overall_confidence < 0.6) {
            try result.qualifications.append(result.allocator, "Some information in this response may need verification.");
        }
    }

    /// Convert to lowercase with bounded buffer.
    fn toLowerBounded(self: *const Self, text: []const u8, buf: []u8) []const u8 {
        _ = self;
        const len = @min(text.len, buf.len);
        for (text[0..len], 0..) |c, i| {
            buf[i] = std.ascii.toLower(c);
        }
        return buf[0..len];
    }

    /// Score a single statement's factual accuracy.
    pub fn scoreStatement(self: *const Self, statement: []const u8) f32 {
        var confidence: f32 = 0.75; // Base confidence

        // Adjust for markers
        confidence = self.adjustConfidenceForMarkers(statement, confidence);

        // Adjust for claim type
        const claim_type = self.detectClaimType(statement, statement);
        const type_confidence = claim_type.getDefaultConfidence();

        return (confidence + type_confidence) / 2.0;
    }
};

/// Apply qualifications to response content.
pub fn applyQualifications(
    allocator: std.mem.Allocator,
    content: []const u8,
    qualifications: []const []const u8,
) ![]const u8 {
    if (qualifications.len == 0) return try allocator.dupe(u8, content);

    var result: std.ArrayListUnmanaged(u8) = .{};
    errdefer result.deinit(allocator);

    try result.appendSlice(allocator, content);

    if (qualifications.len > 0) {
        try result.appendSlice(allocator, "\n\n---\n**Note**: ");
        for (qualifications, 0..) |qual, i| {
            if (i > 0) try result.appendSlice(allocator, " ");
            try result.appendSlice(allocator, qual);
        }
    }

    return result.toOwnedSlice(allocator);
}

// Tests

test "fact checker initialization" {
    const checker = FactChecker.init(std.testing.allocator);
    try std.testing.expectEqual(@as(f32, 0.8), checker.config.min_unqualified_confidence);
}

test "claim type default confidence" {
    try std.testing.expect(ClaimType.definition.getDefaultConfidence() > ClaimType.causal.getDefaultConfidence());
}

test "check simple facts" {
    var checker = FactChecker.init(std.testing.allocator);

    var result = try checker.check("Zig is a systems programming language. It was created by Andrew Kelley.");
    defer result.deinit();

    try std.testing.expect(result.claims.items.len >= 1);
    try std.testing.expect(result.overall_confidence > 0.5);
}

test "detect uncertainty markers" {
    const checker = FactChecker.init(std.testing.allocator);

    const certain = checker.scoreStatement("This is definitely correct.");
    const uncertain = checker.scoreStatement("This might be correct.");

    try std.testing.expect(uncertain < certain);
}

test "detect numerical claims" {
    const checker = FactChecker.init(std.testing.allocator);

    const claim_type = checker.detectClaimType("The average is 50 percent.", "the average is 50 percent.");
    try std.testing.expectEqual(ClaimType.numerical, claim_type);
}

test "detect causal claims" {
    const checker = FactChecker.init(std.testing.allocator);

    const claim_type = checker.detectClaimType("This causes memory leaks.", "this causes memory leaks.");
    try std.testing.expectEqual(ClaimType.causal, claim_type);
}

test "apply qualifications" {
    const qualifications = [_][]const u8{
        "Numbers may be outdated.",
        "Verify with official sources.",
    };

    const result = try applyQualifications(
        std.testing.allocator,
        "Some content here.",
        &qualifications,
    );
    defer std.testing.allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "Note") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "outdated") != null);
}

test "fact check result initialization" {
    var result = FactCheckResult.init(std.testing.allocator);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 0), result.claims.items.len);
    try std.testing.expectEqual(@as(f32, 1.0), result.overall_confidence);
}
