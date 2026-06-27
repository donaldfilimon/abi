//! Lossless order-0 entropy codec (canonical Huffman) for byte data.
//!
//! A real, self-contained entropy coder: it builds a per-input canonical Huffman
//! code from byte frequencies, bit-packs the symbols, and carries the code-length
//! table so `decode` is exact (round-trip identity). Code lengths are capped at
//! 32 bits; inputs whose optimal code would exceed that cap, or that do not
//! compress, fall back to a verbatim `stored` mode — so the codec is always
//! correct and never expands the payload beyond the original.
//!
//! This is a classical entropy coder (Huffman). It is NOT a learned, arithmetic,
//! or ANS codec, and it is order-0 (no context modeling). It complements the
//! lossy int8 `compression` codec and the learned `neural_compress` autoencoder
//! with an exact, lossless option for arbitrary bytes (snapshots, metadata).

const std = @import("std");

const SYM = 256;
/// Canonical code lengths are capped here; codes fit in a u64 with room to spare.
const MAX_LEN = 32;

pub const Mode = enum(u8) { stored, huffman };

pub const Encoded = struct {
    mode: Mode,
    /// huffman: bit-packed codes; stored: verbatim bytes. Owned.
    data: []u8,
    /// huffman: number of valid bits in `data`; stored: `data.len * 8`.
    bit_len: usize,
    original_len: usize,
    /// Canonical code length per symbol (0 = symbol absent). Unused in `stored`.
    code_lengths: [SYM]u8,

    pub fn deinit(self: *Encoded, allocator: std.mem.Allocator) void {
        if (self.data.len > 0) allocator.free(self.data);
        self.data = &.{};
    }

    /// Serialized size in bytes: the payload, plus the 256-byte code-length table
    /// in huffman mode (stored mode carries no table).
    pub fn serializedLen(self: Encoded) usize {
        return switch (self.mode) {
            .stored => self.data.len,
            .huffman => ((self.bit_len + 7) / 8) + SYM,
        };
    }

    /// original / serialized. > 1 means net compression (table overhead
    /// included). Stored mode never exceeds ~1, so the codec never expands data.
    pub fn compressionRatio(self: Encoded) f32 {
        const total = self.serializedLen();
        if (total == 0) return 0;
        return @as(f32, @floatFromInt(self.original_len)) / @as(f32, @floatFromInt(total));
    }
};

const Node = struct { freq: usize, sym: i32, left: i32, right: i32 };

/// Compute canonical code lengths per symbol from byte frequencies. Returns the
/// maximum length assigned (0 if no symbols). Uses an O(n^2) two-smallest merge —
/// trivial for a 256-symbol alphabet and far less error-prone than a heap.
fn buildLengths(allocator: std.mem.Allocator, freq: *const [SYM]usize, lengths: *[SYM]u8) !u8 {
    var nodes = std.ArrayListUnmanaged(Node).empty;
    defer nodes.deinit(allocator);
    var active = std.ArrayListUnmanaged(usize).empty;
    defer active.deinit(allocator);

    var distinct: usize = 0;
    for (0..SYM) |s| {
        if (freq[s] > 0) {
            try nodes.append(allocator, .{ .freq = freq[s], .sym = @intCast(s), .left = -1, .right = -1 });
            try active.append(allocator, nodes.items.len - 1);
            distinct += 1;
        }
    }
    if (distinct == 0) return 0;
    // A single distinct symbol still needs one bit (canonical length 1).
    if (distinct == 1) {
        for (0..SYM) |s| {
            if (freq[s] > 0) lengths[s] = 1;
        }
        return 1;
    }

    while (active.items.len > 1) {
        // Indices (within `active`) of the two smallest-frequency nodes.
        var lo: usize = 0;
        var hi: usize = 1;
        if (nodes.items[active.items[lo]].freq > nodes.items[active.items[hi]].freq) {
            const t = lo;
            lo = hi;
            hi = t;
        }
        var k: usize = 2;
        while (k < active.items.len) : (k += 1) {
            const f = nodes.items[active.items[k]].freq;
            if (f < nodes.items[active.items[lo]].freq) {
                hi = lo;
                lo = k;
            } else if (f < nodes.items[active.items[hi]].freq) {
                hi = k;
            }
        }
        const a = active.items[lo];
        const b = active.items[hi];
        try nodes.append(allocator, .{
            .freq = nodes.items[a].freq + nodes.items[b].freq,
            .sym = -1,
            .left = @intCast(a),
            .right = @intCast(b),
        });
        const parent = nodes.items.len - 1;
        // Remove the larger index first so the smaller stays valid.
        _ = active.swapRemove(@max(lo, hi));
        _ = active.swapRemove(@min(lo, hi));
        try active.append(allocator, parent);
    }

    // Assign each leaf its depth as its code length (iterative DFS).
    const root = active.items[0];
    var max_len: u8 = 0;
    const Frame = struct { idx: usize, depth: u32 };
    var stack = std.ArrayListUnmanaged(Frame).empty;
    defer stack.deinit(allocator);
    try stack.append(allocator, .{ .idx = root, .depth = 0 });
    while (stack.pop()) |fr| {
        const n = nodes.items[fr.idx];
        if (n.sym >= 0) {
            const d: u8 = if (fr.depth == 0) 1 else if (fr.depth > 255) 255 else @intCast(fr.depth);
            lengths[@intCast(n.sym)] = d;
            if (d > max_len) max_len = d;
        } else {
            if (n.left >= 0) try stack.append(allocator, .{ .idx = @intCast(n.left), .depth = fr.depth + 1 });
            if (n.right >= 0) try stack.append(allocator, .{ .idx = @intCast(n.right), .depth = fr.depth + 1 });
        }
    }
    return max_len;
}

/// Assign canonical codes from code lengths (RFC 1951 algorithm).
fn buildCanonical(lengths: *const [SYM]u8, codes: *[SYM]u64) void {
    var bl_count: [MAX_LEN + 1]usize = @splat(0);
    for (lengths) |l| {
        if (l > 0) bl_count[l] += 1;
    }
    var next_code: [MAX_LEN + 1]u64 = @splat(0);
    var code: u64 = 0;
    var bits: usize = 1;
    while (bits <= MAX_LEN) : (bits += 1) {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for (0..SYM) |s| {
        const l = lengths[s];
        if (l > 0) {
            codes[s] = next_code[l];
            next_code[l] += 1;
        }
    }
}

fn storedEncode(allocator: std.mem.Allocator, input: []const u8) !Encoded {
    return .{
        .mode = .stored,
        .data = try allocator.dupe(u8, input),
        .bit_len = input.len * 8,
        .original_len = input.len,
        .code_lengths = @splat(0),
    };
}

/// Entropy-encode `input`. Returns an owned `Encoded` (`deinit` frees it).
pub fn encode(allocator: std.mem.Allocator, input: []const u8) !Encoded {
    var lengths: [SYM]u8 = @splat(0);
    if (input.len == 0) {
        return .{ .mode = .stored, .data = &.{}, .bit_len = 0, .original_len = 0, .code_lengths = lengths };
    }

    var freq: [SYM]usize = @splat(0);
    for (input) |b| freq[b] += 1;

    const max_len = try buildLengths(allocator, &freq, &lengths);
    // Fall back to verbatim when the optimal code would overflow the length cap.
    if (max_len == 0 or max_len > MAX_LEN) return storedEncode(allocator, input);

    var codes: [SYM]u64 = @splat(0);
    buildCanonical(&lengths, &codes);

    var total_bits: usize = 0;
    for (input) |b| total_bits += lengths[b];
    const payload_bytes = (total_bits + 7) / 8;
    // No net win once the 256-byte table is counted → store instead.
    if (payload_bytes + SYM >= input.len) return storedEncode(allocator, input);

    var data = try allocator.alloc(u8, payload_bytes);
    errdefer allocator.free(data);
    @memset(data, 0);

    var bitpos: usize = 0;
    for (input) |b| {
        const l = lengths[b];
        const c = codes[b];
        var i: u8 = 0;
        while (i < l) : (i += 1) {
            const bit: u1 = @intCast((c >> @intCast(l - 1 - i)) & 1);
            if (bit == 1) data[bitpos >> 3] |= (@as(u8, 1) << @intCast(7 - (bitpos & 7)));
            bitpos += 1;
        }
    }

    return .{
        .mode = .huffman,
        .data = data,
        .bit_len = total_bits,
        .original_len = input.len,
        .code_lengths = lengths,
    };
}

/// Decode an `Encoded` back to the exact original bytes. Returns an owned slice.
pub fn decode(allocator: std.mem.Allocator, enc: Encoded) ![]u8 {
    if (enc.original_len == 0) return allocator.alloc(u8, 0);
    if (enc.mode == .stored) return allocator.dupe(u8, enc.data);

    var bl_count: [MAX_LEN + 1]usize = @splat(0);
    for (enc.code_lengths) |l| {
        if (l > 0) bl_count[l] += 1;
    }
    var first_code: [MAX_LEN + 1]u64 = @splat(0);
    {
        var code: u64 = 0;
        var bits: usize = 1;
        while (bits <= MAX_LEN) : (bits += 1) {
            code = (code + bl_count[bits - 1]) << 1;
            first_code[bits] = code;
        }
    }
    // Base offset into the symbol-order table for each length.
    var index_base: [MAX_LEN + 1]usize = @splat(0);
    {
        var acc: usize = 0;
        var l: usize = 1;
        while (l <= MAX_LEN) : (l += 1) {
            index_base[l] = acc;
            acc += bl_count[l];
        }
    }
    var total_syms: usize = 0;
    for (bl_count) |c| total_syms += c;
    var sorted = try allocator.alloc(u16, total_syms);
    defer allocator.free(sorted);
    {
        var cursor = index_base;
        for (0..SYM) |s| {
            const l = enc.code_lengths[s];
            if (l > 0) {
                sorted[cursor[l]] = @intCast(s);
                cursor[l] += 1;
            }
        }
    }

    var out = try allocator.alloc(u8, enc.original_len);
    errdefer allocator.free(out);

    var produced: usize = 0;
    var code: u64 = 0;
    var len: usize = 0;
    var bitpos: usize = 0;
    while (produced < enc.original_len) {
        if ((bitpos >> 3) >= enc.data.len) return error.CorruptEntropyStream;
        const byte = enc.data[bitpos >> 3];
        const bit: u1 = @intCast((byte >> @intCast(7 - (bitpos & 7))) & 1);
        bitpos += 1;
        code = (code << 1) | bit;
        len += 1;
        if (len > MAX_LEN) return error.CorruptEntropyStream;
        if (bl_count[len] > 0) {
            const offset = code -% first_code[len];
            if (offset < bl_count[len]) {
                out[produced] = @intCast(sorted[index_base[len] + @as(usize, @intCast(offset))]);
                produced += 1;
                code = 0;
                len = 0;
            }
        }
    }
    return out;
}

/// Convenience round-trip helper used by tests and callers: encode then decode,
/// returning the reconstructed bytes (must equal the input).
pub fn roundTrip(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var enc = try encode(allocator, input);
    defer enc.deinit(allocator);
    return decode(allocator, enc);
}

const testing = std.testing;

fn expectRoundTrip(input: []const u8) !void {
    const out = try roundTrip(testing.allocator, input);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u8, input, out);
}

test "entropy: round-trips empty, single-symbol, and two-symbol inputs" {
    try expectRoundTrip("");
    try expectRoundTrip("a");
    try expectRoundTrip("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    try expectRoundTrip("ababababababababababababababab");
}

test "entropy: round-trips skewed text and reports net compression" {
    // Highly skewed distribution -> Huffman should beat raw once the fixed
    // 256-byte code-length table is amortized across a realistic payload.
    var text: [4096]u8 = undefined;
    for (&text, 0..) |*b, i| {
        b.* = if (i < 3300) 'a' else if (i < 3900) 'e' else @as(u8, @intCast('0' + (i % 10)));
    }

    var enc = try encode(testing.allocator, &text);
    defer enc.deinit(testing.allocator);
    try testing.expectEqual(Mode.huffman, enc.mode);
    try testing.expect(enc.compressionRatio() > 1.0);

    const out = try decode(testing.allocator, enc);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u8, &text, out);
}

test "entropy: round-trips all 256 byte values" {
    var buf: [1024]u8 = undefined;
    for (0..buf.len) |i| buf[i] = @intCast(i % 256);
    try expectRoundTrip(&buf);
}

test "entropy: incompressible input still round-trips and does not expand the payload" {
    var prng = std.Random.DefaultPrng.init(0xC0FFEE123456);
    const rand = prng.random();
    var buf: [4096]u8 = undefined;
    rand.bytes(&buf);
    var enc = try encode(testing.allocator, &buf);
    defer enc.deinit(testing.allocator);
    // Random bytes don't compress; the codec must remain exact and not expand the
    // payload beyond the original (plus the small header it already accounts for).
    try testing.expect(enc.serializedLen() <= buf.len + SYM);
    const out = try decode(testing.allocator, enc);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(u8, &buf, out);
}

test "entropy: deterministic — same input yields the same encoding" {
    const input = "deterministic entropy coding test vector with repetition rrrrrr";
    var a = try encode(testing.allocator, input);
    defer a.deinit(testing.allocator);
    var b = try encode(testing.allocator, input);
    defer b.deinit(testing.allocator);
    try testing.expectEqual(a.mode, b.mode);
    try testing.expectEqual(a.bit_len, b.bit_len);
    try testing.expectEqualSlices(u8, a.data, b.data);
    try testing.expectEqualSlices(u8, &a.code_lengths, &b.code_lengths);
}

test {
    testing.refAllDecls(@This());
}
