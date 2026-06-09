const std = @import("std");
const types = @import("types.zig");

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
    .{ .word = "what if", .abbey_score = 0.2, .aviva_score = 0.8, .abi_score = 0.2 },
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

    var lower_buffer: [4096]u8 = undefined;
    const lower_input = toLowerSlice(input, &lower_buffer) orelse input;
    var it = std.mem.splitScalar(u8, lower_input, ' ');
    while (it.next()) |word| {
        const trimmed = std.mem.trimEnd(u8, word, &.{ '.', ',', '!', '?', ':', ';', '"', '\'' });
        for (SENTIMENT_KEYWORDS) |kw| {
            if (startsWithIgnoreCase(trimmed, kw.word) or endsWithIgnoreCase(trimmed, kw.word)) {
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

fn toLowerSlice(input: []const u8, buffer: []u8) ?[]const u8 {
    if (input.len > buffer.len) return null;
    for (input, 0..) |byte, i| {
        buffer[i] = std.ascii.toLower(byte);
    }
    return buffer[0..input.len];
}

fn startsWithIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    return haystack.len >= needle.len and std.ascii.eqlIgnoreCase(haystack[0..needle.len], needle);
}

fn endsWithIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    return haystack.len >= needle.len and std.ascii.eqlIgnoreCase(haystack[haystack.len - needle.len ..], needle);
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
