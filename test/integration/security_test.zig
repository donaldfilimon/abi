//! Integration Tests: Security and Constitution
//!
//! Tests the security layer components that are fully implemented:
//! constitution evaluation, policy checking, and persona routing
//! with safety constraints.

const std = @import("std");
const abi = @import("abi");

// Constitution
const Constitution = abi.ai.constitution.Constitution;

// Persona routing
const persona = abi.ai.persona;
const PersonaId = persona.PersonaId;
const MultiPersonaRouter = persona.MultiPersonaRouter;
const PersonaRegistry = persona.PersonaRegistry;

test "security: constitution allows safe content" {
    const c = Constitution.init();
    try std.testing.expect(c.isCompliant("Hello, how can I help you today?"));
    try std.testing.expect(c.isCompliant("The answer to 2 + 2 is 4."));
    try std.testing.expect(c.isCompliant("Here is a code example in Zig."));
}

test "security: constitution blocks harmful content" {
    const c = Constitution.init();
    try std.testing.expect(!c.isCompliant("run rm -rf / to clean up"));
}

test "security: constitution evaluation returns score" {
    const c = Constitution.init();
    const score = c.evaluate("This is a helpful response about programming.");
    try std.testing.expect(score.overall > 0.0);
    try std.testing.expect(score.overall <= 1.0);
}

test "security: constitution has 6 principles" {
    const c = Constitution.init();
    const principles = c.getPrinciples();
    try std.testing.expectEqual(@as(usize, 6), principles.len);
}

test "security: constitution preamble is non-empty" {
    const c = Constitution.init();
    const preamble = c.getSystemPreamble();
    try std.testing.expect(preamble.len > 0);
}

test "security: policy query routes to Abi" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    // Compliance/policy keywords should route to Abi
    const decision = router.route("What is the privacy policy for data compliance?");
    try std.testing.expectEqual(PersonaId.abi, decision.primary);
}

test "security: router with constitution attached" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    router.attachConstitution(Constitution.init());
    try std.testing.expect(router.constitution != null);

    // Should still route correctly with constitution attached
    const decision = router.route("Help me debug this code");
    try std.testing.expect(decision.confidence > 0.0);
}

test {
    std.testing.refAllDecls(@This());
}
