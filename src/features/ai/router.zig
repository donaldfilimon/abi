const std = @import("std");
const types = @import("types.zig");
const point_neural_net = @import("point_neural_net.zig");

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
    .{ .word = "analyze", .abbey_score = 0.8, .aviva_score = 0.2, .abi_score = 0.3 },
    .{ .word = "structure", .abbey_score = 0.9, .aviva_score = 0.1, .abi_score = 0.2 },
    .{ .word = "logical", .abbey_score = 0.85, .aviva_score = 0.15, .abi_score = 0.2 },
    .{ .word = "compare", .abbey_score = 0.7, .aviva_score = 0.4, .abi_score = 0.3 },
    .{ .word = "explain", .abbey_score = 0.6, .aviva_score = 0.5, .abi_score = 0.3 },
    .{ .word = "creative", .abbey_score = 0.2, .aviva_score = 0.9, .abi_score = 0.1 },
    .{ .word = "imagine", .abbey_score = 0.1, .aviva_score = 0.95, .abi_score = 0.1 },
    .{ .word = "explore", .abbey_score = 0.3, .aviva_score = 0.85, .abi_score = 0.2 },
    .{ .word = "brainstorm", .abbey_score = 0.2, .aviva_score = 0.9, .abi_score = 0.15 },
    // Keywords are matched per whitespace-split token (see analyzeSentiment), so
    // every entry must be a single word — multi-word phrases can never match.
    .{ .word = "run", .abbey_score = 0.2, .aviva_score = 0.1, .abi_score = 0.9 },
    .{ .word = "execute", .abbey_score = 0.3, .aviva_score = 0.1, .abi_score = 0.95 },
    .{ .word = "deploy", .abbey_score = 0.2, .aviva_score = 0.1, .abi_score = 0.9 },
    .{ .word = "build", .abbey_score = 0.4, .aviva_score = 0.3, .abi_score = 0.8 },
    .{ .word = "fix", .abbey_score = 0.5, .aviva_score = 0.1, .abi_score = 0.85 },
    .{ .word = "quick", .abbey_score = 0.2, .aviva_score = 0.2, .abi_score = 0.8 },
    .{ .word = "safe", .abbey_score = 0.7, .aviva_score = 0.3, .abi_score = 0.4 },
    .{ .word = "risk", .abbey_score = 0.75, .aviva_score = 0.4, .abi_score = 0.5 },
    .{ .word = "design", .abbey_score = 0.5, .aviva_score = 0.7, .abi_score = 0.3 },
    .{ .word = "pattern", .abbey_score = 0.8, .aviva_score = 0.3, .abi_score = 0.2 },
};

pub const abbey = struct {
    pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        std.log.info("Abbey processing: {s}", .{input});
        return try std.fmt.allocPrint(allocator, "Abbey analyzed: {s}", .{input});
    }
};

pub const aviva = struct {
    pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        return try std.fmt.allocPrint(
            allocator,
            "Aviva creative exploration: {s}\n\nExploring multiple perspectives and creative angles for this topic...",
            .{input},
        );
    }
};

pub const abi_profile = struct {
    pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        return try std.fmt.allocPrint(
            allocator,
            "Abi action: {s}\n\nExecuting requested operation with minimal overhead.",
            .{input},
        );
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
                .w_abbey = 0.33,
                .w_aviva = 0.33,
                .w_abi = 0.34,
            },
            .alpha = DEFAULT_ALPHA,
            .update_count = 0,
        };
    }

    pub fn initWithAlpha(alpha: f32) AdaptiveModulator {
        return .{
            .w_ema = ProfileWeights{
                .w_abbey = 0.33,
                .w_aviva = 0.33,
                .w_abi = 0.34,
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
        self.update_count += 1;
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

    /// Deserialize weights from a stored string.
    pub fn deserialize(data: []const u8) AdaptiveModulator {
        var mod = AdaptiveModulator.init();
        var it = std.mem.splitScalar(u8, data, ',');
        if (it.next()) |s| mod.w_ema.w_abbey = std.fmt.parseFloat(f32, s) catch 0.33;
        if (it.next()) |s| mod.w_ema.w_aviva = std.fmt.parseFloat(f32, s) catch 0.33;
        if (it.next()) |s| mod.w_ema.w_abi = std.fmt.parseFloat(f32, s) catch 0.34;
        if (it.next()) |s| mod.update_count = std.fmt.parseInt(u32, s, 10) catch 0;
        if (it.next()) |s| mod.alpha = std.fmt.parseFloat(f32, s) catch DEFAULT_ALPHA;
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
    var weights_val = ProfileWeights{
        .w_abbey = 0.33,
        .w_aviva = 0.33,
        .w_abi = 0.34,
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

/// Helper function to route to the appropriate profile based on profile selector
pub fn routeToProfile(allocator: std.mem.Allocator, profile_sel: types.AgentProfile, input: []const u8) ![]u8 {
    return switch (profile_sel) {
        .abbey => abbey.processInput(allocator, input),
        .aviva => aviva.processInput(allocator, input),
        .abi => abi_profile.processInput(allocator, input),
    };
}

pub fn routeInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const weights_val = analyzeSentiment(input);
    const profile_sel = selectBestProfile(weights_val);
    return routeToProfile(allocator, profile_sel, input);
}

/// Route input with adaptive EMA modulation and WDBX persistence.
/// Loads modulator state from the store, blends sentiment with EMA weights,
/// routes through the selected profile, then saves updated state.
pub fn routeInputAdaptive(allocator: std.mem.Allocator, store: anytype, input: []const u8) ![]u8 {
    var mod = AdaptiveModulator.loadWeights(store);
    const observed = analyzeSentiment(input);
    mod.update(observed);
    try mod.saveWeights(allocator, store);

    const blended = mod.weights();
    const profile_sel = selectBestProfile(blended);
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
    const keyword_weights = analyzeSentiment(input);
    var neural_weights: ProfileWeights = .{
        .w_abbey = 0.33,
        .w_aviva = 0.33,
        .w_abi = 0.34,
    };

    if (net) |n| {
        if (n.layers.len > 0 and n.layers[n.layers.len - 1].output_size == 3) {
            const point = point_neural_net.Point.fromText(input);
            const output = try n.forward(&point.toArray());
            defer allocator.free(output);
            // Softmax the 3 outputs
            var exps: [3]f32 = undefined;
            var sum: f32 = 0;
            for (output, 0..) |o, i| {
                exps[i] = @exp(o);
                sum += exps[i];
            }
            if (sum > 0) {
                neural_weights.w_abbey = exps[0] / sum;
                neural_weights.w_aviva = exps[1] / sum;
                neural_weights.w_abi = exps[2] / sum;
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
    // both bias abi, the dominant weight for those keywords.
    const quickly = analyzeSentiment("quickly");
    try std.testing.expect(quickly.w_abi > quickly.w_abbey and quickly.w_abi > quickly.w_aviva);
    const running = analyzeSentiment("running");
    try std.testing.expect(running.w_abi > running.w_abbey and running.w_abi > running.w_aviva);

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

test "analyzeSentiment favors aviva for creative input" {
    const weights_val = analyzeSentiment("imagine creative possibilities and explore new ideas");
    try std.testing.expect(weights_val.w_aviva > weights_val.w_abbey);
    try std.testing.expect(weights_val.w_aviva > weights_val.w_abi);
}

test "analyzeSentiment favors abi for action input" {
    const weights_val = analyzeSentiment("execute deploy run the build quickly");
    try std.testing.expect(weights_val.w_abi > weights_val.w_abbey);
    try std.testing.expect(weights_val.w_abi > weights_val.w_aviva);
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
    try std.testing.expectApproxEqAbs(@as(f32, 0.33), mod.w_ema.w_abbey, 0.01);
    try std.testing.expectEqual(@as(u32, 0), mod.update_count);
}

test "abbey processInput" {
    const allocator = std.testing.allocator;
    const result = try abbey.processInput(allocator, "test");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "Abbey analyzed") != null);
}

test "aviva processInput returns creative response" {
    const allocator = std.testing.allocator;
    const result = try aviva.processInput(allocator, "what is consciousness?");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "Aviva creative exploration") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "creative") != null);
}

test "abi_profile processInput returns concise response" {
    const allocator = std.testing.allocator;
    const result = try abi_profile.processInput(allocator, "deploy to production");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "Abi action") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Executing") != null);
}
