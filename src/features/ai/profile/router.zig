const std = @import("std");
const ai = @import("../mod.zig");
const abbey = @import("../abbey/mod.zig");
const aviva = @import("../aviva/mod.zig");
const abi_profile = @import("../abi_profile/mod.zig");

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

const SentimentKeyword = struct {
    word: []const u8,
    abbey_score: f32,
    aviva_score: f32,
    abi_score: f32,
};

const SENTIMENT_KEYWORDS = [_]SentimentKeyword{
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

pub fn analyzeSentiment(input: []const u8) ProfileWeights {
    var weights = ProfileWeights{
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
                weights.w_abbey += kw.abbey_score * 0.1;
                weights.w_aviva += kw.aviva_score * 0.1;
                weights.w_abi += kw.abi_score * 0.1;
            }
        }
    }

    weights.normalize();
    return weights;
}

pub fn selectBestProfile(weights: ProfileWeights) ai.AgentProfile {
    if (weights.w_abbey >= weights.w_aviva and weights.w_abbey >= weights.w_abi) {
        return .abbey;
    } else if (weights.w_aviva >= weights.w_abi) {
        return .aviva;
    } else {
        return .abi;
    }
}

pub fn routeInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const weights = analyzeSentiment(input);
    const profile = selectBestProfile(weights);

    return switch (profile) {
        .abbey => abbey.processInput(allocator, input),
        .aviva => aviva.processInput(allocator, input),
        .abi => abi_profile.processInput(allocator, input),
    };
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
    const weights = analyzeSentiment("analyze the logical structure of this system");
    try std.testing.expect(weights.w_abbey > weights.w_aviva);
    try std.testing.expect(weights.w_abbey > weights.w_abi);
    const total = weights.w_abbey + weights.w_aviva + weights.w_abi;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.01);
}

test "analyzeSentiment favors aviva for creative input" {
    const weights = analyzeSentiment("imagine creative possibilities and explore new ideas");
    try std.testing.expect(weights.w_aviva > weights.w_abbey);
    try std.testing.expect(weights.w_aviva > weights.w_abi);
}

test "analyzeSentiment favors abi for action input" {
    const weights = analyzeSentiment("execute deploy run the build quickly");
    try std.testing.expect(weights.w_abi > weights.w_abbey);
    try std.testing.expect(weights.w_abi > weights.w_aviva);
}

test "analyzeSentiment is case-insensitive" {
    const weights = analyzeSentiment("ANALYZE the LOGICAL structure");
    try std.testing.expect(weights.w_abbey > weights.w_aviva);
    try std.testing.expect(weights.w_abbey > weights.w_abi);
}

test "selectBestProfile picks highest weight" {
    const weights = ProfileWeights{ .w_abbey = 0.6, .w_aviva = 0.2, .w_abi = 0.2 };
    try std.testing.expectEqual(ai.AgentProfile.abbey, selectBestProfile(weights));

    const weights2 = ProfileWeights{ .w_abbey = 0.2, .w_aviva = 0.6, .w_abi = 0.2 };
    try std.testing.expectEqual(ai.AgentProfile.aviva, selectBestProfile(weights2));

    const weights3 = ProfileWeights{ .w_abbey = 0.2, .w_aviva = 0.2, .w_abi = 0.6 };
    try std.testing.expectEqual(ai.AgentProfile.abi, selectBestProfile(weights3));
}

test "routeInput returns response from selected profile" {
    const allocator = std.testing.allocator;
    const result = try routeInput(allocator, "analyze the logical structure");
    defer allocator.free(result);
    try std.testing.expect(result.len > 0);
}
