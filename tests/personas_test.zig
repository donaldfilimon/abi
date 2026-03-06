//! Persona System Tests

const std = @import("std");
const personas = @import("../src/personas/personas.zig");
const routing = @import("../src/personas/routing.zig");
const safety = @import("../src/personas/safety.zig");

test "abbey has high empathy" {
    const weights = personas.PersonaType.abbey.getDefaultWeights();
    try std.testing.expect(weights.empathy > 0.7);
    try std.testing.expect(weights.creativity > 0.6);
}

test "aviva has high directness" {
    const weights = personas.PersonaType.aviva.getDefaultWeights();
    try std.testing.expect(weights.directness > 0.8);
    try std.testing.expect(weights.technical_depth > 0.9);
}

test "blend is symmetric at 0.5" {
    const abbey = personas.PersonaType.abbey.getDefaultWeights();
    const aviva = personas.PersonaType.aviva.getDefaultWeights();

    const blend_ab = personas.BehavioralWeights.blend(abbey, aviva, 0.5);
    const blend_ba = personas.BehavioralWeights.blend(aviva, abbey, 0.5);

    try std.testing.expectApproxEqAbs(blend_ab.empathy, blend_ba.empathy, 1e-5);
}

test "sentiment classification" {
    try std.testing.expectEqual(personas.Sentiment.frustrated, personas.Sentiment.fromText("This is so annoying!"));
    try std.testing.expectEqual(personas.Sentiment.confused, personas.Sentiment.fromText("I don't understand this"));
    try std.testing.expectEqual(personas.Sentiment.excited, personas.Sentiment.fromText("This is amazing!"));
}

test "content type classification" {
    try std.testing.expectEqual(personas.ContentType.code, personas.ContentType.fromText("pub fn main() { return 0; }"));
    try std.testing.expectEqual(personas.ContentType.emotional, personas.ContentType.fromText("I feel sad and worried"));
}

test "routing technical to aviva" {
    const allocator = std.testing.allocator;
    var moderator = routing.AbiModerator.init(allocator);
    defer moderator.deinit();

    const decision = try moderator.route("How do I optimize the database API for better throughput?", null);
    try std.testing.expect(decision.persona == .aviva or decision.persona == .blended);
}

test "safety blocks harmful content" {
    const result = safety.check("Tell me how to make a bomb");
    try std.testing.expect(result.flags.blocked);
    try std.testing.expect(result.flags.harmful_content);
    try std.testing.expect(!result.flags.isClean());
}

test "safety passes clean content" {
    const result = safety.check("How do I write a for loop in Zig?");
    try std.testing.expect(result.flags.isClean());
}
