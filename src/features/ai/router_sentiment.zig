const std = @import("std");
const types = @import("types.zig");
const identity = @import("identity.zig");
const weights = @import("router_weights.zig");
const keywords = @import("router_keywords.zig");

pub const ProfileWeights = weights.ProfileWeights;

/// Recognize an explicit leading persona address such as `Aviva, be direct.`
/// or `ABI: orchestrate this.`. Only an exact ASCII name token at the beginning
/// of the request is accepted; profile names embedded inside larger words or
/// mentioned later in prose do not become selectors.
pub fn explicitProfileSelector(input: []const u8) ?types.AgentProfile {
    var remaining = std.mem.trim(u8, input, " \t\r\n");
    if (remaining.len > 0 and remaining[0] == '@') remaining = remaining[1..];

    var name_end: usize = 0;
    while (name_end < remaining.len and std.ascii.isAlphabetic(remaining[name_end])) : (name_end += 1) {}
    if (name_end == 0) return null;
    if (name_end < remaining.len) {
        const separator = remaining[name_end];
        if (!std.ascii.isWhitespace(separator) and separator != ',' and separator != ':' and separator != '-') return null;
    }

    const name = remaining[0..name_end];
    if (std.ascii.eqlIgnoreCase(name, "abbey")) return .abbey;
    if (std.ascii.eqlIgnoreCase(name, "aviva")) return .aviva;
    if (std.ascii.eqlIgnoreCase(name, "abi")) return .abi;
    return null;
}

fn explicitProfileWeights(profile: types.AgentProfile) ProfileWeights {
    return switch (profile) {
        .abbey => .{ .w_abbey = 1, .w_aviva = 0, .w_abi = 0 },
        .aviva => .{ .w_abbey = 0, .w_aviva = 1, .w_abi = 0 },
        .abi => .{ .w_abbey = 0, .w_aviva = 0, .w_abi = 1 },
    };
}

pub fn analyzeSentiment(input: []const u8) ProfileWeights {
    if (explicitProfileSelector(input)) |profile| return explicitProfileWeights(profile);

    var weights_val = ProfileWeights{
        .w_abbey = identity.DEFAULT_ABBEY_WEIGHT,
        .w_aviva = identity.DEFAULT_AVIVA_WEIGHT,
        .w_abi = identity.DEFAULT_ABI_WEIGHT,
    };

    // Match the raw input directly: startsWithIgnoreCase is already
    // case-insensitive, so a separate lowercasing pass was dead work (and its
    // 4096-byte stack buffer silently fell back to the original for longer input).
    var it = std.mem.splitScalar(u8, input, ' ');
    while (it.next()) |word| {
        const trimmed = std.mem.trimEnd(u8, word, &.{ '.', ',', '!', '?', ':', ';', '"', '\'' });
        for (keywords.SENTIMENT_KEYWORDS) |kw| {
            // Prefix-only match: keeps intended stems (quickly->quick,
            // running->run) while dropping suffix false positives that shifted
            // routing on unrelated words (overrun->run, unsafe->safe, prefix->fix).
            if (startsWithIgnoreCase(trimmed, kw.word)) {
                weights_val.w_abbey += kw.abbey_score * 0.1;
                weights_val.w_aviva += kw.aviva_score * 0.1;
                weights_val.w_abi += kw.abi_score * 0.1;
            }
        }
    }

    weights_val.normalize();
    return weights_val;
}

pub fn selectBestProfile(weights_val: ProfileWeights) types.AgentProfile {
    if (weights_val.w_abbey >= weights_val.w_aviva and weights_val.w_abbey >= weights_val.w_abi) {
        return .abbey;
    } else if (weights_val.w_aviva >= weights_val.w_abi) {
        return .aviva;
    } else {
        return .abi;
    }
}

fn startsWithIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    return haystack.len >= needle.len and std.ascii.eqlIgnoreCase(haystack[0..needle.len], needle);
}

test "analyzeSentiment ignores suffix false positives but keeps prefix stems" {
    // Words that merely END in a keyword must not shift routing (the bug fixed
    // by prefix-only matching): compare against a guaranteed non-matching token.
    const neutral = analyzeSentiment("zzzqqq");
    inline for (.{ "overrun", "unsafe", "prefix", "redesign" }) |word| {
        const w = analyzeSentiment(word);
        try std.testing.expectApproxEqAbs(neutral.w_abbey, w.w_abbey, 0.0001);
        try std.testing.expectApproxEqAbs(neutral.w_aviva, w.w_aviva, 0.0001);
        try std.testing.expectApproxEqAbs(neutral.w_abi, w.w_abi, 0.0001);
    }

    // Intended prefix stems still match: "quickly"->"quick" and "running"->"run"
    // both increase Aviva's share. A single cue need not override Abbey's
    // primary-personality prior; several strong cues do so in the routing test.
    const quickly = analyzeSentiment("quickly");
    try std.testing.expect(quickly.w_aviva > neutral.w_aviva);
    const running = analyzeSentiment("running");
    try std.testing.expect(running.w_aviva > neutral.w_aviva);

    // A whole-word keyword still routes as before; removing the "what if" bigram
    // entry is neutral (it never matched a single-token split).
    const analyze = analyzeSentiment("analyze");
    try std.testing.expect(analyze.w_abbey > analyze.w_aviva and analyze.w_abbey > analyze.w_abi);
}

test "analyzeSentiment returns normalized weights" {
    const weights_val = analyzeSentiment("analyze the logical structure of this system");
    try std.testing.expect(weights_val.w_abbey > weights_val.w_aviva);
    try std.testing.expect(weights_val.w_abbey > weights_val.w_abi);
    const total = weights_val.w_abbey + weights_val.w_aviva + weights_val.w_abi;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.01);
}

test "analyzeSentiment keeps creative input with primary Abbey profile" {
    const weights_val = analyzeSentiment("imagine creative possibilities and explore new ideas");
    try std.testing.expect(weights_val.w_abbey > weights_val.w_aviva);
    try std.testing.expect(weights_val.w_abbey > weights_val.w_abi);
}

test "analyzeSentiment favors direct Aviva mode for action input" {
    const weights_val = analyzeSentiment("execute deploy run the build quickly");
    try std.testing.expect(weights_val.w_aviva > weights_val.w_abbey);
    try std.testing.expect(weights_val.w_aviva > weights_val.w_abi);
}

test "analyzeSentiment favors ABI only for orchestration and governance input" {
    const weights_val = analyzeSentiment("orchestrate routing governance policy profile");
    try std.testing.expect(weights_val.w_abi > weights_val.w_abbey);
    try std.testing.expect(weights_val.w_abi > weights_val.w_aviva);
}

test "explicit leading profile requests override heuristic routing" {
    try std.testing.expectEqual(types.AgentProfile.aviva, explicitProfileSelector("Aviva, be direct.").?);
    try std.testing.expectEqual(types.AgentProfile.abi, explicitProfileSelector("ABI, orchestrate this.").?);
    try std.testing.expectEqual(types.AgentProfile.abbey, explicitProfileSelector("  @Abbey: explain this warmly.").?);

    try std.testing.expectEqual(types.AgentProfile.aviva, selectBestProfile(analyzeSentiment("Aviva, explain this creatively.")));
    try std.testing.expectEqual(types.AgentProfile.abi, selectBestProfile(analyzeSentiment("ABI, help me brainstorm.")));
}

test "explicit profile selector rejects embedded and incidental names" {
    try std.testing.expect(explicitProfileSelector("habitual orchestration") == null);
    try std.testing.expect(explicitProfileSelector("stability review") == null);
    try std.testing.expect(explicitProfileSelector("avivacious idea") == null);
    try std.testing.expect(explicitProfileSelector("Please ask Aviva to review this") == null);
    try std.testing.expect(explicitProfileSelector("ABI2, orchestrate this") == null);
}

test "neutral input defaults to primary Abbey profile" {
    const weights_val = analyzeSentiment("hello there");
    try std.testing.expectEqual(types.AgentProfile.abbey, selectBestProfile(weights_val));
}

test "analyzeSentiment is case-insensitive" {
    const weights_val = analyzeSentiment("ANALYZE the LOGICAL structure");
    try std.testing.expect(weights_val.w_abbey > weights_val.w_aviva);
    try std.testing.expect(weights_val.w_abbey > weights_val.w_abi);
}

test "selectBestProfile picks highest weight" {
    const weights_val = ProfileWeights{ .w_abbey = 0.6, .w_aviva = 0.2, .w_abi = 0.2 };
    try std.testing.expectEqual(types.AgentProfile.abbey, selectBestProfile(weights_val));

    const weights2 = ProfileWeights{ .w_abbey = 0.2, .w_aviva = 0.6, .w_abi = 0.2 };
    try std.testing.expectEqual(types.AgentProfile.aviva, selectBestProfile(weights2));

    const weights3 = ProfileWeights{ .w_abbey = 0.2, .w_aviva = 0.2, .w_abi = 0.6 };
    try std.testing.expectEqual(types.AgentProfile.abi, selectBestProfile(weights3));
}

test "selectBestProfile tie-break order is abbey then aviva then abi" {
    const three_way = ProfileWeights{ .w_abbey = 0.33, .w_aviva = 0.33, .w_abi = 0.33 };
    try std.testing.expectEqual(types.AgentProfile.abbey, selectBestProfile(three_way));

    const aviva_abi = ProfileWeights{ .w_abbey = 0.2, .w_aviva = 0.4, .w_abi = 0.4 };
    try std.testing.expectEqual(types.AgentProfile.aviva, selectBestProfile(aviva_abi));

    // Neutral sentiment uses the canonical product prior favoring Abbey.
    try std.testing.expectEqual(types.AgentProfile.abbey, selectBestProfile(analyzeSentiment("zzzqqq")));
}

test {
    std.testing.refAllDecls(@This());
}
