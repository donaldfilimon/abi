const std = @import("std");

test "e2e: llm pipeline (error handling)" {
    // Verify engine initialization, loading a missing model, and proper cleanup.
    const std = @import("std");
    const llm = @import("../../src/features/ai/llm.zig");
    var engine = llm.Engine.init(std.testing.allocator, .{});
    defer engine.deinit();
    // Expect ModelNotFound error when loading a non‑existent GGUF file.
    const err = engine.loadModel("nonexistent-model.gguf");
    try std.testing.expectError(error.ModelNotFound, err);
    // The engine should be usable after a failed load (no panic).
    // Ensure cleanup does not panic.
}
