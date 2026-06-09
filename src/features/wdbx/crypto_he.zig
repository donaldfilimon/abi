//! Additively homomorphic encryption (Security Layer).
//!
//! A real partially-homomorphic scheme over a prime field: ciphertexts can be
//! summed without decryption, and the sum decrypts to the sum of plaintexts.
//! It is a keyed additive masking scheme (one-time-pad in GF(p)): for a secret
//! key k and nonce r, Enc(m) = (m + PRF(k, r)) mod p, carrying r. Addition
//! merges the nonce sets and adds the masked values; decryption subtracts the
//! sum of the per-nonce masks.
//!
//! Honest scope: this provides the *additive* homomorphism needed for
//! encrypted aggregation under a single trusted key. It is NOT a fully
//! homomorphic scheme (no multiplication) and not multi-key. Full FHE remains a
//! research-horizon item; this is the working additive primitive that supports
//! it incrementally.

const std = @import("std");

/// A 61-bit Mersenne-ish prime chosen so additions of many ciphertexts fit in
/// u64 intermediates after each mod; the field is GF(p).
pub const P: u64 = (1 << 61) - 1;

pub const PublicError = error{
    OutOfMemory,
    PlaintextTooLarge,
    EmptyCiphertext,
};

fn prf(key: u64, nonce: u64) u64 {
    var buf: [16]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], key, .little);
    std.mem.writeInt(u64, buf[8..16], nonce, .little);
    const h = std.hash.Wyhash.hash(0xA11CE, &buf);
    return h % P;
}

fn addMod(a: u64, b: u64) u64 {
    // a, b < P < 2^61, so a + b < 2^62 — no overflow before the reduction.
    return (a + b) % P;
}

fn subMod(a: u64, b: u64) u64 {
    return (a + (P - (b % P))) % P;
}

pub const Cipher = struct {
    value: u64,
    nonces: std.ArrayListUnmanaged(u64) = .empty,

    pub fn deinit(self: *Cipher, allocator: std.mem.Allocator) void {
        self.nonces.deinit(allocator);
    }

    pub fn clone(self: Cipher, allocator: std.mem.Allocator) !Cipher {
        var n: std.ArrayListUnmanaged(u64) = .empty;
        try n.appendSlice(allocator, self.nonces.items);
        return .{ .value = self.value, .nonces = n };
    }
};

pub const Key = struct {
    secret: u64,

    pub fn init(secret: u64) Key {
        return .{ .secret = secret };
    }

    /// Encrypt a field element `m` (0 <= m < P) under nonce `nonce`.
    pub fn encrypt(self: Key, allocator: std.mem.Allocator, m: u64, nonce: u64) !Cipher {
        if (m >= P) return error.PlaintextTooLarge;
        var nonces: std.ArrayListUnmanaged(u64) = .empty;
        errdefer nonces.deinit(allocator);
        try nonces.append(allocator, nonce);
        return .{ .value = addMod(m, prf(self.secret, nonce)), .nonces = nonces };
    }

    pub fn decrypt(self: Key, c: Cipher) !u64 {
        if (c.nonces.items.len == 0) return error.EmptyCiphertext;
        var mask: u64 = 0;
        for (c.nonces.items) |n| mask = addMod(mask, prf(self.secret, n));
        return subMod(c.value, mask);
    }
};

/// Homomorphic addition: Dec(add(Enc(a), Enc(b))) == (a + b) mod P, without
/// either operand being decrypted. Allocates a fresh ciphertext.
pub fn add(allocator: std.mem.Allocator, x: Cipher, y: Cipher) !Cipher {
    var nonces: std.ArrayListUnmanaged(u64) = .empty;
    errdefer nonces.deinit(allocator);
    try nonces.appendSlice(allocator, x.nonces.items);
    try nonces.appendSlice(allocator, y.nonces.items);
    return .{ .value = addMod(x.value, y.value), .nonces = nonces };
}

test "he: encrypt/decrypt round-trips" {
    const allocator = std.testing.allocator;
    const key = Key.init(0xDEADBEEF);
    var c = try key.encrypt(allocator, 42, 1);
    defer c.deinit(allocator);
    try std.testing.expectEqual(@as(u64, 42), try key.decrypt(c));
}

test "he: ciphertext sum decrypts to plaintext sum (homomorphic addition)" {
    const allocator = std.testing.allocator;
    const key = Key.init(0x1234_5678);

    var ca = try key.encrypt(allocator, 1000, 11);
    defer ca.deinit(allocator);
    var cb = try key.encrypt(allocator, 337, 22);
    defer cb.deinit(allocator);

    var cs = try add(allocator, ca, cb);
    defer cs.deinit(allocator);

    // The summed ciphertext was never decrypted to its operands.
    try std.testing.expectEqual(@as(u64, 1337), try key.decrypt(cs));
}

test "he: aggregates many encrypted values" {
    const allocator = std.testing.allocator;
    const key = Key.init(99);

    var acc = try key.encrypt(allocator, 0, 0);
    defer acc.deinit(allocator);
    var expected: u64 = 0;
    var i: u64 = 1;
    while (i <= 50) : (i += 1) {
        var ci = try key.encrypt(allocator, i, 1000 + i);
        defer ci.deinit(allocator);
        var next = try add(allocator, acc, ci);
        acc.deinit(allocator);
        acc = next;
        _ = &next;
        expected += i;
    }
    try std.testing.expectEqual(expected, try key.decrypt(acc));
}

test "he: a wrong key does not recover the plaintext" {
    const allocator = std.testing.allocator;
    const key = Key.init(0xAAAA);
    var c = try key.encrypt(allocator, 7, 5);
    defer c.deinit(allocator);
    const wrong = Key.init(0xBBBB);
    try std.testing.expect((try wrong.decrypt(c)) != 7);
}

test "he: rejects out-of-field plaintext" {
    const allocator = std.testing.allocator;
    const key = Key.init(1);
    try std.testing.expectError(error.PlaintextTooLarge, key.encrypt(allocator, P, 1));
}

test {
    std.testing.refAllDecls(@This());
}
