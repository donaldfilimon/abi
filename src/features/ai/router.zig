const std = @import("std");
const types = @import("types.zig");
const point_neural_net = @import("point_neural_net.zig");
const incremental = @import("incremental.zig");
const identity = @import("identity.zig");

pub const ProfileWeights = struct {
    w_abbey: f32,
    w_aviva: f32,
    w_abi: f32,

    pub fn normalize(self: *ProfileWeights) void {
        const total = self.w_abbey + self.w_aviva + self.w_abi;
        if (total > 0) {
            self.w_abbey /= total;
            self.w_aviva /= total;
            self.w_abi /= total;
        }
    }
};

pub const SentimentKeyword = struct {
    word: []const u8,
    abbey_score: f32,
    aviva_score: f32,
    abi_score: f32,
};

pub const SENTIMENT_KEYWORDS = [_]SentimentKeyword{
    // Abbey is the broad, primary personality: analytical, creative,
    // empathetic, explanatory, and collaborative cues all strengthen Abbey.
    .{ .word = "analyze", .abbey_score = 0.9, .aviva_score = 0.5, .abi_score = 0.2 },
    .{ .word = "structure", .abbey_score = 0.9, .aviva_score = 0.6, .abi_score = 0.2 },
    .{ .word = "logical", .abbey_score = 0.85, .aviva_score = 0.7, .abi_score = 0.2 },
    .{ .word = "compare", .abbey_score = 0.8, .aviva_score = 0.7, .abi_score = 0.2 },
    .{ .word = "explain", .abbey_score = 0.95, .aviva_score = 0.4, .abi_score = 0.2 },
    .{ .word = "creative", .abbey_score = 0.95, .aviva_score = 0.3, .abi_score = 0.1 },
    .{ .word = "imagine", .abbey_score = 0.95, .aviva_score = 0.3, .abi_score = 0.1 },
    .{ .word = "explore", .abbey_score = 0.9, .aviva_score = 0.4, .abi_score = 0.2 },
    .{ .word = "brainstorm", .abbey_score = 0.95, .aviva_score = 0.3, .abi_score = 0.1 },
    .{ .word = "help", .abbey_score = 0.95, .aviva_score = 0.4, .abi_score = 0.2 },
    .{ .word = "learn", .abbey_score = 0.95, .aviva_score = 0.3, .abi_score = 0.2 },
    .{ .word = "frustrated", .abbey_score = 0.95, .aviva_score = 0.2, .abi_score = 0.1 },
    // Keywords are matched per whitespace-split token (see analyzeSentiment), so
    // every entry must be a single word — multi-word phrases can never match.
    // Aviva is the direct expert mode for urgent, terse execution cues.
    .{ .word = "run", .abbey_score = 0.3, .aviva_score = 0.95, .abi_score = 0.2 },
    .{ .word = "execute", .abbey_score = 0.3, .aviva_score = 0.95, .abi_score = 0.2 },
    .{ .word = "deploy", .abbey_score = 0.3, .aviva_score = 0.95, .abi_score = 0.2 },
    .{ .word = "build", .abbey_score = 0.5, .aviva_score = 0.9, .abi_score = 0.2 },
    .{ .word = "fix", .abbey_score = 0.5, .aviva_score = 0.95, .abi_score = 0.2 },
    .{ .word = "quick", .abbey_score = 0.3, .aviva_score = 0.95, .abi_score = 0.1 },
    .{ .word = "direct", .abbey_score = 0.3, .aviva_score = 0.95, .abi_score = 0.1 },
    .{ .word = "concise", .abbey_score = 0.3, .aviva_score = 0.95, .abi_score = 0.1 },
    // ABI is selected for explicit orchestration/governance work rather than
    // ordinary user-facing execution.
    .{ .word = "orchestrate", .abbey_score = 0.2, .aviva_score = 0.2, .abi_score = 0.95 },
    .{ .word = "routing", .abbey_score = 0.2, .aviva_score = 0.2, .abi_score = 0.95 },
    .{ .word = "governance", .abbey_score = 0.3, .aviva_score = 0.3, .abi_score = 0.95 },
    .{ .word = "policy", .abbey_score = 0.3, .aviva_score = 0.4, .abi_score = 0.9 },
    .{ .word = "profile", .abbey_score = 0.3, .aviva_score = 0.3, .abi_score = 0.9 },
    .{ .word = "safe", .abbey_score = 0.8, .aviva_score = 0.5, .abi_score = 0.5 },
    .{ .word = "risk", .abbey_score = 0.8, .aviva_score = 0.6, .abi_score = 0.6 },
    .{ .word = "design", .abbey_score = 0.9, .aviva_score = 0.5, .abi_score = 0.3 },
    .{ .word = "pattern", .abbey_score = 0.85, .aviva_score = 0.5, .abi_score = 0.3 },
};

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

pub const abbey = struct {
    pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        std.log.info("Abbey processing: {s}", .{input});
        const contract = identity.profileContract(.abbey);
        return try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
            contract.response_prefix,
            input,
            contract.response_suffix,
        });
    }
};

pub const aviva = struct {
    pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        const contract = identity.profileContract(.aviva);
        return try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
            contract.response_prefix,
            input,
            contract.response_suffix,
        });
    }
};

pub const abi_profile = struct {
    pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        const contract = identity.profileContract(.abi);
        return try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
            contract.response_prefix,
            input,
            contract.response_suffix,
        });
    }
};

/// Adaptive EMA Modulator that smooths routing weights over time
/// and persists them to a WDBX store.
pub const AdaptiveModulator = struct {
    w_ema: ProfileWeights,
    alpha: f32,
    update_count: u32,

    const STORE_KEY = "modulator:weights";
    const DEFAULT_ALPHA: f32 = 0.3;

    pub fn init() AdaptiveModulator {
        return .{
            .w_ema = ProfileWeights{
                .w_abbey = identity.DEFAULT_ABBEY_WEIGHT,
                .w_aviva = identity.DEFAULT_AVIVA_WEIGHT,
                .w_abi = identity.DEFAULT_ABI_WEIGHT,
            },
            .alpha = DEFAULT_ALPHA,
            .update_count = 0,
        };
    }

    pub fn initWithAlpha(alpha: f32) AdaptiveModulator {
        return .{
            .w_ema = ProfileWeights{
                .w_abbey = identity.DEFAULT_ABBEY_WEIGHT,
                .w_aviva = identity.DEFAULT_AVIVA_WEIGHT,
                .w_abi = identity.DEFAULT_ABI_WEIGHT,
            },
            .alpha = alpha,
            .update_count = 0,
        };
    }

    /// Update the EMA weights with new observed sentiment weights.
    /// new_ema = alpha * observed + (1 - alpha) * old_ema
    pub fn update(self: *AdaptiveModulator, observed: ProfileWeights) void {
        const a = self.alpha;
        const b = 1.0 - a;
        self.w_ema.w_abbey = a * observed.w_abbey + b * self.w_ema.w_abbey;
        self.w_ema.w_aviva = a * observed.w_aviva + b * self.w_ema.w_aviva;
        self.w_ema.w_abi = a * observed.w_abi + b * self.w_ema.w_abi;
        self.w_ema.normalize();
        self.update_count +|= 1;
    }

    /// Get the current smoothed weights.
    pub fn weights(self: *const AdaptiveModulator) ProfileWeights {
        return self.w_ema;
    }

    /// Serialize weights to a string for WDBX persistence.
    pub fn serialize(self: *const AdaptiveModulator, allocator: std.mem.Allocator) ![]u8 {
        return try std.fmt.allocPrint(
            allocator,
            "{d:.6},{d:.6},{d:.6},{d},{d:.6}",
            .{ self.w_ema.w_abbey, self.w_ema.w_aviva, self.w_ema.w_abi, self.update_count, self.alpha },
        );
    }

    /// Deserialize weights from a stored string. Falls back to
    /// `AdaptiveModulator.init()` defaults if the persisted state fails
    /// validation (malformed fields, non-finite/negative weights, invalid
    /// totals/counts, or alpha outside `[0,1]`).
    pub fn deserialize(data: []const u8) AdaptiveModulator {
        return deserializeValidated(data) orelse AdaptiveModulator.init();
    }

    /// Parses and validates the CSV-encoded persisted state. Returns `null`
    /// (rather than a partially-defaulted value) if any field is malformed,
    /// a weight is non-finite/negative, the weight total is invalid, alpha is
    /// outside `[0,1]`, or the field count doesn't match exactly.
    fn deserializeValidated(data: []const u8) ?AdaptiveModulator {
        var it = std.mem.splitScalar(u8, data, ',');
        const abbey_text = it.next() orelse return null;
        const aviva_text = it.next() orelse return null;
        const abi_text = it.next() orelse return null;
        const update_count_text = it.next() orelse return null;
        const alpha_text = it.next() orelse return null;
        if (it.next() != null) return null;

        const abbey_weight = std.fmt.parseFloat(f32, abbey_text) catch return null;
        const aviva_weight = std.fmt.parseFloat(f32, aviva_text) catch return null;
        const abi_weight = std.fmt.parseFloat(f32, abi_text) catch return null;
        const update_count = std.fmt.parseInt(u32, update_count_text, 10) catch return null;
        const alpha = std.fmt.parseFloat(f32, alpha_text) catch return null;

        if (!std.math.isFinite(abbey_weight) or
            !std.math.isFinite(aviva_weight) or
            !std.math.isFinite(abi_weight) or
            abbey_weight < 0 or
            aviva_weight < 0 or
            abi_weight < 0)
        {
            return null;
        }

        const total = abbey_weight + aviva_weight + abi_weight;
        if (!std.math.isFinite(total) or total <= 0) return null;
        if (!std.math.isFinite(alpha) or alpha < 0 or alpha > 1) return null;

        var mod = AdaptiveModulator{
            .w_ema = .{
                .w_abbey = abbey_weight,
                .w_aviva = aviva_weight,
                .w_abi = abi_weight,
            },
            .alpha = alpha,
            .update_count = update_count,
        };
        mod.w_ema.normalize();
        return mod;
    }

    /// Load weights from a WDBX store. Returns default if key is missing.
    pub fn loadWeights(store: anytype) AdaptiveModulator {
        const val = store.get(STORE_KEY) orelse return AdaptiveModulator.init();
        return AdaptiveModulator.deserialize(val);
    }

    /// Save current weights to a WDBX store.
    pub fn saveWeights(self: *const AdaptiveModulator, allocator: std.mem.Allocator, store: anytype) !void {
        const serialized = try self.serialize(allocator);
        defer allocator.free(serialized);
        try store.store(STORE_KEY, serialized);
    }
};

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
        for (SENTIMENT_KEYWORDS) |kw| {
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

/// Helper function to route to the appropriate profile based on profile selector.
/// Uses the same iterative template generator as streaming completions (without
/// a callback) so one-shot and incremental paths stay string-identical.
pub fn routeToProfile(allocator: std.mem.Allocator, profile_sel: types.AgentProfile, input: []const u8) ![]u8 {
    return incremental.generateProfileIncremental(allocator, profile_sel, input, null, null);
}

pub fn routeInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const weights_val = analyzeSentiment(input);
    const profile_sel = selectBestProfile(weights_val);
    return routeToProfile(allocator, profile_sel, input);
}

/// Soul-aware routing: blends keyword-based sentiment with a
/// pre-trained 3-output PointNeuralNetwork (one output per profile:
/// abbey, aviva, abi). The network's output is softmax-normalized
/// and blended with keyword weights using `blend_alpha` (0.0 = keyword only,
/// 1.0 = neural only). Falls back to keyword-only if `net` is null
/// or doesn't have 3 outputs.
pub fn routeInputWithSoul(
    allocator: std.mem.Allocator,
    net: ?*point_neural_net.PointNeuralNetwork,
    blend_alpha: f32,
    input: []const u8,
) ![]u8 {
    if (explicitProfileSelector(input)) |profile| {
        return routeToProfile(allocator, profile, input);
    }

    const keyword_weights = analyzeSentiment(input);
    // Start from the keyword decision so a missing network or rejected output
    // shape/value preserves the documented fallback regardless of blend_alpha.
    var neural_weights = keyword_weights;

    if (net) |n| {
        if (n.layers.len > 0 and n.layers[n.layers.len - 1].output_size == 3) {
            const point = point_neural_net.Point.fromText(input);
            const output = try n.forward(&point.toArray());
            defer allocator.free(output);
            if (output.len == 3) {
                // Stable softmax: subtracting the largest finite logit avoids
                // overflow while non-finite output preserves keyword fallback.
                var logits_are_finite = true;
                var max_logit = output[0];
                for (output[1..]) |o| {
                    if (!std.math.isFinite(o)) {
                        logits_are_finite = false;
                        break;
                    }
                    max_logit = @max(max_logit, o);
                }
                if (!std.math.isFinite(max_logit)) logits_are_finite = false;

                if (logits_are_finite) {
                    var exps: [3]f32 = undefined;
                    var sum: f32 = 0;
                    for (output, 0..) |o, i| {
                        exps[i] = @exp(o - max_logit);
                        sum += exps[i];
                    }
                    if (sum > 0 and std.math.isFinite(sum)) {
                        neural_weights.w_abbey = exps[0] / sum;
                        neural_weights.w_aviva = exps[1] / sum;
                        neural_weights.w_abi = exps[2] / sum;
                    }
                }
            }
        }
    }

    const blended = blendWeights(keyword_weights, neural_weights, blend_alpha);
    const profile_sel = selectBestProfile(blended);
    return routeToProfile(allocator, profile_sel, input);
}

/// Blend two ProfileWeights with alpha (0.0 = a only, 1.0 = b only).
pub fn blendWeights(a: ProfileWeights, b: ProfileWeights, alpha: f32) ProfileWeights {
    const a_alpha = 1.0 - alpha;
    return .{
        .w_abbey = a.w_abbey * a_alpha + b.w_abbey * alpha,
        .w_aviva = a.w_aviva * a_alpha + b.w_aviva * alpha,
        .w_abi = a.w_abi * a_alpha + b.w_abi * alpha,
    };
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

test {
    std.testing.refAllDecls(@This());
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

test "explicit profile requests override one-shot and soul routing" {
    const allocator = std.testing.allocator;

    const aviva_result = try routeInput(allocator, "Aviva, be direct.");
    defer allocator.free(aviva_result);
    try std.testing.expect(std.mem.startsWith(u8, aviva_result, identity.profileContract(.aviva).response_prefix));

    const abi_result = try routeInputWithSoul(allocator, null, 1.0, "ABI, orchestrate this.");
    defer allocator.free(abi_result);
    try std.testing.expect(std.mem.startsWith(u8, abi_result, identity.profileContract(.abi).response_prefix));
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

test "routeInputWithSoul preserves keyword routing without a network" {
    const allocator = std.testing.allocator;
    const input = "analyze the logical structure";

    const expected = try routeInput(allocator, input);
    defer allocator.free(expected);
    const actual = try routeInputWithSoul(allocator, null, 1.0, input);
    defer allocator.free(actual);

    try std.testing.expectEqualStrings(expected, actual);
}

test "routeInput returns response from selected profile" {
    const allocator = std.testing.allocator;
    const result = try routeInput(allocator, "analyze the logical structure");
    defer allocator.free(result);
    try std.testing.expect(result.len > 0);
}

test "AdaptiveModulator EMA smoothing" {
    var mod = AdaptiveModulator.initWithAlpha(0.5);
    const observed = ProfileWeights{ .w_abbey = 1.0, .w_aviva = 0.0, .w_abi = 0.0 };
    mod.update(observed);

    // After one update with alpha=0.5, abbey should be higher than initial
    try std.testing.expect(mod.w_ema.w_abbey > 0.5);
    try std.testing.expectEqual(@as(u32, 1), mod.update_count);

    // Weights should still be normalized
    const total = mod.w_ema.w_abbey + mod.w_ema.w_aviva + mod.w_ema.w_abi;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.01);
}

test "AdaptiveModulator serialize/deserialize roundtrip" {
    var mod = AdaptiveModulator.initWithAlpha(0.25);
    const observed = ProfileWeights{ .w_abbey = 0.8, .w_aviva = 0.1, .w_abi = 0.1 };
    mod.update(observed);

    const allocator = std.testing.allocator;
    const serialized = try mod.serialize(allocator);
    defer allocator.free(serialized);

    const restored = AdaptiveModulator.deserialize(serialized);
    try std.testing.expectApproxEqAbs(mod.w_ema.w_abbey, restored.w_ema.w_abbey, 0.001);
    try std.testing.expectApproxEqAbs(mod.w_ema.w_aviva, restored.w_ema.w_aviva, 0.001);
    try std.testing.expectApproxEqAbs(mod.w_ema.w_abi, restored.w_ema.w_abi, 0.001);
    try std.testing.expectEqual(mod.update_count, restored.update_count);
    try std.testing.expectApproxEqAbs(mod.alpha, restored.alpha, 0.001);
}

test "AdaptiveModulator default deserialization on missing key" {
    const mod = AdaptiveModulator.deserialize("");
    try std.testing.expectApproxEqAbs(identity.DEFAULT_ABBEY_WEIGHT, mod.w_ema.w_abbey, 0.01);
    try std.testing.expectEqual(@as(u32, 0), mod.update_count);
}

test "AdaptiveModulator rejects invalid persisted state deterministically" {
    const invalid_states = [_][]const u8{
        "nan,0.3,0.4,1,0.3",
        "inf,0.3,0.4,1,0.3",
        "-inf,0.3,0.4,1,0.3",
        "-0.1,0.3,0.8,1,0.3",
        "0,0,0,1,0.3",
        "3.4028235e38,3.4028235e38,3.4028235e38,1,0.3",
        "0.3,0.3,0.4,1,nan",
        "0.3,0.3,0.4,1,inf",
        "0.3,0.3,0.4,1,-0.1",
        "0.3,0.3,0.4,1,1.1",
        "malformed,0.3,0.4,1,0.3",
        "0.3,,0.4,1,0.3",
        "0.3,0.3,0.4,not-a-count,0.3",
        "0.3,0.3,0.4,4294967296,0.3",
        "0.3,0.3,0.4,1",
        "0.3,0.3,0.4,1,0.3,",
        "0.3,0.3,0.4,1,0.3,extra",
    };

    for (invalid_states) |state| {
        const restored = AdaptiveModulator.deserialize(state);
        try std.testing.expectApproxEqAbs(identity.DEFAULT_ABBEY_WEIGHT, restored.w_ema.w_abbey, 0.0001);
        try std.testing.expectApproxEqAbs(identity.DEFAULT_AVIVA_WEIGHT, restored.w_ema.w_aviva, 0.0001);
        try std.testing.expectApproxEqAbs(identity.DEFAULT_ABI_WEIGHT, restored.w_ema.w_abi, 0.0001);
        try std.testing.expectEqual(@as(u32, 0), restored.update_count);
        try std.testing.expectApproxEqAbs(@as(f32, 0.3), restored.alpha, 0.0001);
    }
}

test "AdaptiveModulator normalizes valid persisted weights" {
    const restored = AdaptiveModulator.deserialize("2,3,5,42,0.75");
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), restored.w_ema.w_abbey, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), restored.w_ema.w_aviva, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), restored.w_ema.w_abi, 0.0001);
    try std.testing.expectEqual(@as(u32, 42), restored.update_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), restored.alpha, 0.0001);
}

test "AdaptiveModulator accepts persisted alpha boundaries" {
    const zero_alpha = AdaptiveModulator.deserialize("2,3,5,7,0");
    try std.testing.expectEqual(@as(u32, 7), zero_alpha.update_count);
    try std.testing.expectEqual(@as(f32, 0), zero_alpha.alpha);

    const one_alpha = AdaptiveModulator.deserialize("2,3,5,9,1");
    try std.testing.expectEqual(@as(u32, 9), one_alpha.update_count);
    try std.testing.expectEqual(@as(f32, 1), one_alpha.alpha);
}

test "AdaptiveModulator saturates a persisted maximum update count" {
    var restored = AdaptiveModulator.deserialize("0.2,0.3,0.5,4294967295,0.5");
    const observed = ProfileWeights{ .w_abbey = 1.0, .w_aviva = 0.0, .w_abi = 0.0 };
    restored.update(observed);

    try std.testing.expectEqual(std.math.maxInt(u32), restored.update_count);
    const total = restored.w_ema.w_abbey + restored.w_ema.w_aviva + restored.w_ema.w_abi;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.0001);
}

test "abbey processInput" {
    const allocator = std.testing.allocator;
    const result = try abbey.processInput(allocator, "test");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "Abbey:") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "warmth, creativity, and technical care") != null);
}

test "aviva processInput returns direct expert response" {
    const allocator = std.testing.allocator;
    const result = try aviva.processInput(allocator, "what is consciousness?");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "Aviva direct expert") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "concrete answer") != null);
}

test "abi_profile processInput returns orchestration response" {
    const allocator = std.testing.allocator;
    const result = try abi_profile.processInput(allocator, "deploy to production");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "ABI orchestration review") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "intent, risk, context") != null);
}
