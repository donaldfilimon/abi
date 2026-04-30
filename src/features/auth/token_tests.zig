const std = @import("std");
const auth = @import("mod.zig");
const env_gate = @import("common");

// Token-related tests guarded behind env gate to ensure parity can run in
// environments where ABI_JWT_SECRET is not configured.
test "auth token segmentation is exactly three parts" {
    if (!env_gate.canRunAuth()) return;
    const allocator = std.testing.allocator;
    const token = try auth.createToken(allocator, "seg_check");
    defer allocator.free(token.raw);
    var count: u32 = 0;
    var it = std.mem.splitScalar(u8, token.raw, '.');
    while (it.next()) |_| count += 1;
    try std.testing.expectEqual(@as(u32, 3), count);
    // Ensure each segment is non-empty
    var it2 = std.mem.splitScalar(u8, token.raw, '.');
    while (it2.next()) |seg| {
        try std.testing.expect(seg.len > 0);
    }
}
