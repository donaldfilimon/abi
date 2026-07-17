//! Demo rANS (range Asymmetric Numeral Systems) + order-1 residual coding.
//!
//! Complements Huffman in `entropy.zig` with a second entropy-coding family.
//! Order-1 here means previous-byte residual coding (`b - prev`) fed into
//! order-0 rANS — a classical context technique without a 256x256 table.
//!
//! HONEST SCOPE: reference/demo for `abi wdbx secure demo` and unit tests.
//! **Not** SOTA, **not** a production learned codec, **not** a zlib/zstd
//! replacement.

const std = @import("std");

const SYM = 256;
const L_BASE: u32 = 1 << 16;
const SCALE_BITS: u5 = 12;
const SCALE: u32 = 1 << SCALE_BITS;

pub const Mode = enum(u8) {
    stored = 0,
    rans0 = 1,
    rans1 = 2,
};

pub const Encoded = struct {
    mode: Mode,
    data: []u8,
    original_len: usize,

    pub fn deinit(self: *Encoded, allocator: std.mem.Allocator) void {
        if (self.data.len > 0) allocator.free(self.data);
        self.data = &.{};
    }

    pub fn serializedLen(self: Encoded) usize {
        return self.data.len;
    }

    pub fn compressionRatio(self: Encoded) f32 {
        if (self.data.len == 0) return 0;
        return @as(f32, @floatFromInt(self.original_len)) / @as(f32, @floatFromInt(self.data.len));
    }
};

fn normalize(freq: *[SYM]u32, total: u32) void {
    if (total == 0) {
        @memset(freq, 0);
        freq[0] = SCALE;
        return;
    }
    var sum: u32 = 0;
    var max_i: usize = 0;
    for (0..SYM) |i| {
        if (freq[i] == 0) continue;
        const scaled: u32 = @max(1, @as(u32, @intCast((@as(u64, freq[i]) * SCALE) / total)));
        freq[i] = scaled;
        sum += scaled;
        if (freq[i] >= freq[max_i]) max_i = i;
    }
    if (sum == 0) {
        freq[0] = SCALE;
        return;
    }
    if (sum > SCALE) freq[max_i] -= sum - SCALE else if (sum < SCALE) freq[max_i] += SCALE - sum;
}

fn cumulFrom(freq: *const [SYM]u32, cumul: *[SYM + 1]u32) void {
    cumul[0] = 0;
    for (0..SYM) |i| cumul[i + 1] = cumul[i] + freq[i];
}

fn symbolFrom(cumul: *const [SYM + 1]u32, slot: u32) u8 {
    var lo: usize = 0;
    var hi: usize = SYM;
    while (lo + 1 < hi) {
        const mid = (lo + hi) / 2;
        if (cumul[mid] <= slot) lo = mid else hi = mid;
    }
    return @intCast(lo);
}

const Renorm = struct {
    words: std.ArrayListUnmanaged(u16) = .empty,

    fn push(self: *Renorm, allocator: std.mem.Allocator, w: u16) !void {
        try self.words.append(allocator, w);
    }
};

fn put(x: *u32, renorm: *Renorm, allocator: std.mem.Allocator, f: u32, start: u32) !void {
    const x_max = ((L_BASE >> SCALE_BITS) << 16) * f;
    while (x.* >= x_max) {
        try renorm.push(allocator, @truncate(x.*));
        x.* >>= 16;
    }
    x.* = (x.* / f) * SCALE + start + (x.* % f);
}

fn get(x: *u32, stream: *[]const u16, f: u32, start: u32) !void {
    x.* = f * (x.* / SCALE) + (x.* % SCALE) - start;
    while (x.* < L_BASE) {
        if (stream.*.len == 0) return error.TruncatedAnsStream;
        const w = stream.*[stream.*.len - 1];
        stream.* = stream.*[0 .. stream.*.len - 1];
        x.* = (x.* << 16) | w;
    }
}

fn writeU32(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, v: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try list.appendSlice(allocator, &buf);
}

fn writeU16(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, v: u16) !void {
    var buf: [2]u8 = undefined;
    std.mem.writeInt(u16, &buf, v, .little);
    try list.appendSlice(allocator, &buf);
}

fn readU32(data: []const u8, off: *usize) !u32 {
    if (off.* + 4 > data.len) return error.TruncatedAnsStream;
    const v = std.mem.readInt(u32, data[off.*..][0..4], .little);
    off.* += 4;
    return v;
}

fn readU16(data: []const u8, off: *usize) !u16 {
    if (off.* + 2 > data.len) return error.TruncatedAnsStream;
    const v = std.mem.readInt(u16, data[off.*..][0..2], .little);
    off.* += 2;
    return v;
}

fn packFreqRow(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, freq: *const [SYM]u32) !void {
    for (freq) |f| try writeU16(list, allocator, @intCast(f));
}

fn unpackFreqRow(data: []const u8, off: *usize, freq: *[SYM]u32) !void {
    for (freq) |*f| f.* = try readU16(data, off);
}

fn buildFreq(bytes: []const u8) [SYM]u32 {
    var freq: [SYM]u32 = @splat(0);
    for (bytes) |b| freq[b] += 1;
    var total: u32 = 0;
    for (freq) |f| total += f;
    normalize(&freq, total);
    return freq;
}

fn toResiduals(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, input.len);
    var prev: u8 = 0;
    for (input, out) |b, *slot| {
        slot.* = b -% prev;
        prev = b;
    }
    return out;
}

fn fromResiduals(residuals: []const u8, out: []u8) void {
    var prev: u8 = 0;
    for (residuals, out) |r, *slot| {
        const b = prev +% r;
        slot.* = b;
        prev = b;
    }
}

fn encodeBytes(allocator: std.mem.Allocator, input: []const u8, mode: Mode) !Encoded {
    if (input.len == 0) {
        var out: std.ArrayListUnmanaged(u8) = .empty;
        try out.append(allocator, @intFromEnum(Mode.stored));
        try writeU32(&out, allocator, 0);
        return .{ .mode = .stored, .data = try out.toOwnedSlice(allocator), .original_len = 0 };
    }

    var freq = buildFreq(input);
    var cumul: [SYM + 1]u32 = undefined;
    cumulFrom(&freq, &cumul);

    var renorm: Renorm = .{};
    defer renorm.words.deinit(allocator);

    var x: u32 = L_BASE;
    var i = input.len;
    while (i > 0) {
        i -= 1;
        const s = input[i];
        const f = freq[s];
        if (f == 0) return error.ZeroFrequencySymbol;
        try put(&x, &renorm, allocator, f, cumul[s]);
    }

    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, @intFromEnum(mode));
    try writeU32(&out, allocator, @intCast(input.len));
    try writeU32(&out, allocator, x);
    try packFreqRow(&out, allocator, &freq);
    try writeU32(&out, allocator, @intCast(renorm.words.items.len));
    for (renorm.words.items) |w| try writeU16(&out, allocator, w);

    if (out.items.len >= input.len) {
        out.deinit(allocator);
        var stored: std.ArrayListUnmanaged(u8) = .empty;
        errdefer stored.deinit(allocator);
        try stored.append(allocator, @intFromEnum(Mode.stored));
        try writeU32(&stored, allocator, @intCast(input.len));
        try stored.appendSlice(allocator, input);
        return .{ .mode = .stored, .data = try stored.toOwnedSlice(allocator), .original_len = input.len };
    }

    return .{ .mode = mode, .data = try out.toOwnedSlice(allocator), .original_len = input.len };
}

fn decodeBytes(allocator: std.mem.Allocator, blob: []const u8) !struct { mode: Mode, bytes: []u8 } {
    if (blob.len < 1) return error.TruncatedAnsStream;
    const mode: Mode = @enumFromInt(blob[0]);
    var off: usize = 1;
    const original_len = try readU32(blob, &off);
    if (mode == .stored) {
        if (off + original_len > blob.len) return error.TruncatedAnsStream;
        return .{ .mode = .stored, .bytes = try allocator.dupe(u8, blob[off .. off + original_len]) };
    }

    var x = try readU32(blob, &off);
    var freq: [SYM]u32 = undefined;
    try unpackFreqRow(blob, &off, &freq);
    var cumul: [SYM + 1]u32 = undefined;
    cumulFrom(&freq, &cumul);

    const nwords = try readU32(blob, &off);
    if (off + nwords * 2 > blob.len) return error.TruncatedAnsStream;
    const words = try allocator.alloc(u16, nwords);
    defer allocator.free(words);
    for (words) |*w| w.* = try readU16(blob, &off);

    var stream: []const u16 = words;
    const out = try allocator.alloc(u8, original_len);
    errdefer allocator.free(out);

    for (out) |*slot| {
        const s = symbolFrom(&cumul, x % SCALE);
        const f = freq[s];
        if (f == 0) return error.ZeroFrequencySymbol;
        try get(&x, &stream, f, cumul[s]);
        slot.* = s;
    }
    return .{ .mode = mode, .bytes = out };
}

/// Order-0 rANS encode (self-contained blob).
pub fn encode(allocator: std.mem.Allocator, input: []const u8) !Encoded {
    return encodeBytes(allocator, input, .rans0);
}

/// Order-1 residual coding: encode (byte - previous) with order-0 rANS.
pub fn encodeOrder1(allocator: std.mem.Allocator, input: []const u8) !Encoded {
    if (input.len == 0) return encode(allocator, input);
    const residuals = try toResiduals(allocator, input);
    defer allocator.free(residuals);
    // Encode residuals; tag as rans1. If stored fallback, store original plaintext
    // under stored mode (honest smaller encoding).
    var enc = try encodeBytes(allocator, residuals, .rans1);
    if (enc.mode == .stored) {
        enc.deinit(allocator);
        return encodeBytes(allocator, input, .stored);
    }
    // Rewrite original_len to plaintext length (same as residuals.len).
    enc.original_len = input.len;
    return enc;
}

pub fn decode(allocator: std.mem.Allocator, enc: Encoded) ![]u8 {
    const decoded = try decodeBytes(allocator, enc.data);
    if (decoded.mode != .rans1) return decoded.bytes;
    // Reconstruct plaintext from residuals.
    const out = try allocator.alloc(u8, decoded.bytes.len);
    fromResiduals(decoded.bytes, out);
    allocator.free(decoded.bytes);
    return out;
}

test "ans order-0 round-trips" {
    const allocator = std.testing.allocator;
    const input = "aaaaaaaaaaabbbbbbbbbbccccccccccHELLO_WORLD!!!!";
    var enc = try encode(allocator, input);
    defer enc.deinit(allocator);
    const out = try decode(allocator, enc);
    defer allocator.free(out);
    try std.testing.expectEqualStrings(input, out);
}

test "ans order-1 round-trips structured text" {
    const allocator = std.testing.allocator;
    const input = "the the the cat sat on the mat the the cat";
    var enc = try encodeOrder1(allocator, input);
    defer enc.deinit(allocator);
    const out = try decode(allocator, enc);
    defer allocator.free(out);
    try std.testing.expectEqualStrings(input, out);
}

test "ans empty round-trips" {
    const allocator = std.testing.allocator;
    var enc = try encode(allocator, "");
    defer enc.deinit(allocator);
    const out = try decode(allocator, enc);
    defer allocator.free(out);
    try std.testing.expectEqualStrings("", out);
}

test {
    std.testing.refAllDecls(@This());
}
