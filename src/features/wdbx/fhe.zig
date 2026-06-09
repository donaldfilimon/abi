//! Somewhat-homomorphic encryption (Security Layer).
//!
//! A real DGHV-style integer somewhat-homomorphic scheme (van Dijk–Gentry–
//! Halevi–Vaikuntanathan) in pure Zig 0.17, using native wide integers. Unlike
//! the additive single-key scheme in `crypto_he.zig`, this supports BOTH
//! homomorphic addition (XOR) and multiplication (AND) on encrypted bits — the
//! foundation of the FHE family — for a bounded multiplicative depth.
//!
//! Scheme (symmetric, public modulus x0 = p·q0):
//!   sk = p (large odd)         enc(m) = (m + 2r + p·q) mod x0
//!   dec(c) = (c mod p) mod 2   add = (c1+c2) mod x0   mul = (c1·c2) mod x0
//! Decryption is robust to the mod-x0 reduction because x0 ≡ 0 (mod p), so the
//! small noise m+2r is recovered exactly while it stays below p/2.
//!
//! HONEST SCOPE: reference parameters (~126-bit p, ~2^20 noise) chosen so
//! correctness is exactly testable and a small multiplicative depth stays within
//! the noise budget (depth-3 chained multiplies are verified in tests; deeper
//! circuits eventually exceed the budget). These are NOT cryptographically
//! secure sizes and the scheme is
//! not bootstrapped — it demonstrates the homomorphic add+multiply capability,
//! it is not a production cryptosystem. Correctness is proven by the tests
//! below over many random keys and circuits.

const std = @import("std");

/// Ciphertext / key integer width. Two reduced ciphertexts (< x0 ≈ 2^252) can be
/// multiplied (< 2^504) without overflow, with comfortable headroom.
pub const Int = i1024;

pub const Keypair = struct {
    /// Secret modulus (large odd).
    p: Int,
    /// Public co-factor bound.
    q0: Int,
    /// Public modulus x0 = p·q0 (ciphertexts are reduced mod x0).
    x0: Int,
};

const NOISE_BITS: u6 = 20;

/// Generate a keypair from `rand`. Pass a seeded PRNG for deterministic keys.
pub fn keygen(rand: std.Random) Keypair {
    var p_u: u128 = rand.int(u128);
    p_u |= 1; // p must be odd
    p_u |= (@as(u128, 1) << 126); // ensure ~126-bit magnitude
    var q0_u: u128 = rand.int(u128);
    q0_u |= (@as(u128, 1) << 126);
    const p: Int = @intCast(p_u);
    const q0: Int = @intCast(q0_u);
    return .{ .p = p, .q0 = q0, .x0 = p * q0 };
}

/// Encrypt one bit. `rand` supplies the per-ciphertext noise and co-factor.
pub fn encrypt(kp: Keypair, rand: std.Random, bit: u1) Int {
    const r: Int = @intCast(rand.uintLessThan(u64, @as(u64, 1) << NOISE_BITS));
    const q0_u: u128 = @intCast(kp.q0);
    const q: Int = @intCast(rand.uintLessThan(u128, q0_u));
    const m: Int = bit;
    return @mod(kp.p * q + 2 * r + m, kp.x0);
}

/// Decrypt to a bit: (c mod p) mod 2.
pub fn decrypt(kp: Keypair, c: Int) u1 {
    const noise = @mod(@mod(c, kp.x0), kp.p);
    return @intCast(@mod(noise, 2));
}

/// Homomorphic addition — decrypts to m1 XOR m2.
pub fn add(kp: Keypair, a: Int, b: Int) Int {
    return @mod(a + b, kp.x0);
}

/// Homomorphic multiplication — decrypts to m1 AND m2 (consumes one depth level).
pub fn mul(kp: Keypair, a: Int, b: Int) Int {
    return @mod(a * b, kp.x0);
}

const testing = std.testing;

test "fhe: additive and multiplicative homomorphism hold over random keys/bits" {
    var prng = std.Random.DefaultPrng.init(0xF0E1D2C3B4A59687);
    const rand = prng.random();

    var trial: usize = 0;
    while (trial < 128) : (trial += 1) {
        const kp = keygen(rand);
        const a: u1 = @intCast(rand.uintLessThan(u8, 2));
        const b: u1 = @intCast(rand.uintLessThan(u8, 2));

        const ea = encrypt(kp, rand, a);
        const eb = encrypt(kp, rand, b);

        // Sanity: a fresh ciphertext decrypts to its plaintext.
        try testing.expectEqual(a, decrypt(kp, ea));
        try testing.expectEqual(b, decrypt(kp, eb));

        // XOR via homomorphic add, AND via homomorphic multiply.
        try testing.expectEqual(a ^ b, decrypt(kp, add(kp, ea, eb)));
        try testing.expectEqual(a & b, decrypt(kp, mul(kp, ea, eb)));
    }
}

test "fhe: a depth-2 circuit evaluates correctly on ciphertexts" {
    var prng = std.Random.DefaultPrng.init(0x1122334455667788);
    const rand = prng.random();

    var trial: usize = 0;
    while (trial < 64) : (trial += 1) {
        const kp = keygen(rand);
        const a: u1 = @intCast(rand.uintLessThan(u8, 2));
        const b: u1 = @intCast(rand.uintLessThan(u8, 2));
        const c: u1 = @intCast(rand.uintLessThan(u8, 2));
        const d: u1 = @intCast(rand.uintLessThan(u8, 2));

        const ea = encrypt(kp, rand, a);
        const eb = encrypt(kp, rand, b);
        const ec = encrypt(kp, rand, c);
        const ed = encrypt(kp, rand, d);

        // Homomorphically evaluate ((a AND b) XOR (c AND d)) — two multiplies
        // (depth 1 each) combined by an add; within the noise budget.
        const e_ab = mul(kp, ea, eb);
        const e_cd = mul(kp, ec, ed);
        const e_res = add(kp, e_ab, e_cd);

        const want: u1 = (a & b) ^ (c & d);
        try testing.expectEqual(want, decrypt(kp, e_res));
    }
}

test "fhe: chained multiplications stay within the noise budget (depth 3)" {
    var prng = std.Random.DefaultPrng.init(0x9988776655443322);
    const rand = prng.random();

    var trial: usize = 0;
    while (trial < 48) : (trial += 1) {
        const kp = keygen(rand);
        const a: u1 = @intCast(rand.uintLessThan(u8, 2));
        const b: u1 = @intCast(rand.uintLessThan(u8, 2));
        const c: u1 = @intCast(rand.uintLessThan(u8, 2));
        const d: u1 = @intCast(rand.uintLessThan(u8, 2));

        // Three chained multiplies (depth 3): each consumes a noise level, so
        // this exercises the multiplicative-depth budget far past a single mul.
        const e = mul(kp, mul(kp, mul(kp, encrypt(kp, rand, a), encrypt(kp, rand, b)), encrypt(kp, rand, c)), encrypt(kp, rand, d));
        try testing.expectEqual(a & b & c & d, decrypt(kp, e));
    }
}

test {
    testing.refAllDecls(@This());
}
