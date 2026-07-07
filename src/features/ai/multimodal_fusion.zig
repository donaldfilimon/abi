const std = @import("std");

/// Vision processor stub: encodes a text description into a fixed-length vector.
/// Deterministic — maps each ASCII character to a token bucket, normalizes to
/// unit length (design reference: `docs/spec/wdbx-rust-capability-extract.mdx`
/// §7.1).  In a production system this would be a real vision encoder; here it
/// produces a deterministic, testable representation.
pub const VisionProcessor = struct {
    pub const EMBEDDING_LEN: usize = 64;

    /// Encode a text description into a deterministic 64-element vector.
    /// Character buckets: each char's ASCII value maps to a bucket index, that
    /// bucket's value is incremented.  The result is L2-normalized.
    pub fn encode(allocator: std.mem.Allocator, description: []const u8) ![]f32 {
        var raw = try allocator.alloc(f32, EMBEDDING_LEN);
        @memset(raw, 0);
        for (description) |ch| {
            const idx = @as(usize, @intCast(ch)) % EMBEDDING_LEN;
            raw[idx] += 1.0;
        }
        // L2 normalize
        var norm: f32 = 0;
        for (raw) |v| norm += v * v;
        if (norm > 1e-12) {
            const inv = 1.0 / @sqrt(norm);
            for (raw) |*v| v.* *= inv;
        }
        return raw;
    }
};

/// Audio processor stub: encodes a text description into a fixed-length vector.
/// Same deterministic strategy as VisionProcessor (design reference:
/// `docs/spec/wdbx-rust-capability-extract.mdx` §7.2).
pub const AudioProcessor = struct {
    pub const EMBEDDING_LEN: usize = 32;

    pub fn encode(allocator: std.mem.Allocator, description: []const u8) ![]f32 {
        var raw = try allocator.alloc(f32, EMBEDDING_LEN);
        @memset(raw, 0);
        for (description) |ch| {
            const idx = @as(usize, @intCast(ch)) % EMBEDDING_LEN;
            raw[idx] += 1.0;
        }
        var norm: f32 = 0;
        for (raw) |v| norm += v * v;
        if (norm > 1e-12) {
            const inv = 1.0 / @sqrt(norm);
            for (raw) |*v| v.* *= inv;
        }
        return raw;
    }
};

/// IoT processor: wraps IotMonitor and produces a numeric embedding from the
/// monitor's current normalized mean reading (design reference:
/// `docs/spec/wdbx-rust-capability-extract.mdx` §6–7.3).
pub const IotProcessor = struct {
    pub const EMBEDDING_LEN: usize = 16;

    pub fn encode(allocator: std.mem.Allocator, mean_reading: f64) ![]f32 {
        var raw = try allocator.alloc(f32, EMBEDDING_LEN);
        @memset(raw, 0);
        const val = @as(f32, @floatCast(mean_reading));
        // Scatter the value across the embedding with a simple LCG-like stride
        // so the same value always produces the same vector.
        raw[0] = val;
        const step: usize = 7;
        var idx: usize = step % EMBEDDING_LEN;
        var i: usize = 0;
        while (i < EMBEDDING_LEN) : (i += 1) {
            raw[idx] += @as(f32, @floatFromInt(@as(i32, @intCast(i))));
            idx = (idx + step) % EMBEDDING_LEN;
        }
        var norm: f32 = 0;
        for (raw) |sample| norm += sample * sample;
        if (norm > 1e-12) {
            const inv = 1.0 / @sqrt(norm);
            for (raw) |*v| v.* *= inv;
        }
        return raw;
    }
};

/// Fuse multiple modality embeddings into a single output vector by
/// concatenation (design reference: `docs/spec/wdbx-rust-capability-extract.mdx`
/// §7.4).  Returns `{vision_len + audio_len + iot_len}` f32 elements.
pub fn fuse(
    allocator: std.mem.Allocator,
    vision: []const f32,
    audio: []const f32,
    iot: []const f32,
) ![]f32 {
    const total = vision.len + audio.len + iot.len;
    var out = try allocator.alloc(f32, total);
    var pos: usize = 0;
    @memcpy(out[0..vision.len], vision);
    pos += vision.len;
    @memcpy(out[pos..][0..audio.len], audio);
    pos += audio.len;
    @memcpy(out[pos..][0..iot.len], iot);
    return out;
}

test "VisionProcessor produces deterministic unit-length vectors" {
    const v1 = try VisionProcessor.encode(std.testing.allocator, "a red ball");
    defer std.testing.allocator.free(v1);
    const v2 = try VisionProcessor.encode(std.testing.allocator, "a red ball");
    defer std.testing.allocator.free(v2);
    try std.testing.expectEqualDeep(v1, v2);
    var norm: f32 = 0;
    for (v1) |x| norm += x * x;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-4);
}

test "VisionProcessor different descriptions produce different embeddings" {
    const v1 = try VisionProcessor.encode(std.testing.allocator, "cat");
    defer std.testing.allocator.free(v1);
    const v2 = try VisionProcessor.encode(std.testing.allocator, "dog");
    defer std.testing.allocator.free(v2);
    try std.testing.expect(!std.mem.eql(f32, v1, v2));
}

test "AudioProcessor produces deterministic unit-length vectors" {
    const a1 = try AudioProcessor.encode(std.testing.allocator, "loud beep");
    defer std.testing.allocator.free(a1);
    const a2 = try AudioProcessor.encode(std.testing.allocator, "loud beep");
    defer std.testing.allocator.free(a2);
    try std.testing.expectEqualDeep(a1, a2);
    var norm: f32 = 0;
    for (a1) |x| norm += x * x;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-4);
}

test "IotProcessor produces deterministic unit-length vectors" {
    const emb1 = try IotProcessor.encode(std.testing.allocator, 0.75);
    defer std.testing.allocator.free(emb1);
    const emb2 = try IotProcessor.encode(std.testing.allocator, 0.75);
    defer std.testing.allocator.free(emb2);
    try std.testing.expectEqualDeep(emb1, emb2);
    var norm: f32 = 0;
    for (emb1) |x| norm += x * x;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-4);
}

test "fuse concatenates all three embeddings" {
    const v = try VisionProcessor.encode(std.testing.allocator, "test");
    defer std.testing.allocator.free(v);
    const a = try AudioProcessor.encode(std.testing.allocator, "test");
    defer std.testing.allocator.free(a);
    const i = try IotProcessor.encode(std.testing.allocator, 0.5);
    defer std.testing.allocator.free(i);

    const fused = try fuse(std.testing.allocator, v, a, i);
    defer std.testing.allocator.free(fused);

    try std.testing.expectEqual(VisionProcessor.EMBEDDING_LEN + AudioProcessor.EMBEDDING_LEN + IotProcessor.EMBEDDING_LEN, fused.len);
    // first segment matches vision
    try std.testing.expect(std.mem.eql(f32, fused[0..v.len], v));
    // second segment matches audio
    try std.testing.expect(std.mem.eql(f32, fused[v.len..][0..a.len], a));
    // third segment matches iot
    try std.testing.expect(std.mem.eql(f32, fused[v.len + a.len ..], i));
}

test {
    std.testing.refAllDecls(@This());
}
