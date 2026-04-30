const std = @import("std");

test "e2e: llm pipeline (skeleton)" {
    // TODO: implement full LLM pipeline test (load -> tokenize -> generate -> decode)
    // This skeleton ensures the test file compiles in CI.
    // For now, verify that the LLM module compiles and basic types are available.
    const llm = @import("../../src/features/ai/llm.zig");
    _ = llm;
}
