const std = @import("std");

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

test {
    std.testing.refAllDecls(@This());
}
