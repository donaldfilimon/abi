const std = @import("std");
const abi = @import("abi");

// Structural guard for the documented completion persistence contract.
// See docs/contracts/public-api.mdx (committed this turn to SCRATCH): only `completion:<query_vector_id>`
// metadata + vectors + block; exactly +1 kv entry per store_result=true call.
// No memory_record in the basic (non-SEA) path.

test "completion_kv_delta matches committed contract" {
    try std.testing.expectEqual(@as(usize, 1), abi.features.ai.completion_kv_delta);
}

test "completionMetadataKey produces documented key shape" {
    const key = try abi.features.ai.completionMetadataKey(std.testing.allocator, 123);
    defer std.testing.allocator.free(key);
    try std.testing.expect(std.mem.startsWith(u8, key, "completion:"));
    try std.testing.expect(std.mem.indexOf(u8, key, "123") != null);
}

test "basic completeWithStore does not expose memory_record symbols" {
    // Guard against re-introducing memory_record in the basic path.
    // If the completion submodule re-exports or defines it, this will fail at comptime
    // when the test module is analyzed.
    const ai = abi.features.ai;
    if (@hasDecl(ai, "completionMemoryRecordJson") or @hasDecl(ai, "completionMemoryRecordKey")) {
        @compileError("memory_record symbols must not be present for basic completion contract");
    }
    // Also assert via the module that the only documented kv writer is the metadata one.
    _ = ai.completion_kv_delta;
    _ = ai.completionMetadataKey;
}
