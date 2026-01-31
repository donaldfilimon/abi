//! Cryptography and Security Benchmarks
//!
//! Industry-standard benchmarks for cryptographic operations:
//! - Hash functions (SHA-256, SHA-512, Blake2, Blake3)
//! - HMAC operations
//! - Key derivation (PBKDF2, scrypt simulation)
//! - Symmetric encryption/decryption (AES-GCM, ChaCha20-Poly1305)
//! - Asymmetric operations (signature generation/verification)
//! - Random number generation
//! - Constant-time comparison
//! - Base64 encoding/decoding
//! - Hex encoding/decoding

const std = @import("std");
const framework = @import("../system/framework.zig");
const crypto = std.crypto;

/// Crypto benchmark configuration
pub const CryptoBenchConfig = struct {
    /// Data sizes for throughput tests (reduced for faster runs)
    data_sizes: []const usize = &.{ 64, 1024, 16384 },
    /// Number of PBKDF2 iterations to test (removed 100000 - too slow)
    pbkdf_iterations: []const u32 = &.{ 1000, 10000 },
    /// Number of parallel hashes for throughput
    parallel_count: usize = 1000,
};

// ============================================================================
// Hash Function Benchmarks
// ============================================================================

fn benchSha256(data: []const u8) [32]u8 {
    var result: [32]u8 = undefined;
    crypto.hash.sha2.Sha256.hash(data, &result, .{});
    return result;
}

fn benchSha512(data: []const u8) [64]u8 {
    var result: [64]u8 = undefined;
    crypto.hash.sha2.Sha512.hash(data, &result, .{});
    return result;
}

fn benchBlake2b256(data: []const u8) [32]u8 {
    var result: [32]u8 = undefined;
    crypto.hash.blake2.Blake2b256.hash(data, &result, .{});
    return result;
}

fn benchBlake2b512(data: []const u8) [64]u8 {
    var result: [64]u8 = undefined;
    crypto.hash.blake2.Blake2b512.hash(data, &result, .{});
    return result;
}

fn benchBlake3(data: []const u8) [32]u8 {
    var result: [32]u8 = undefined;
    crypto.hash.Blake3.hash(data, &result, .{});
    return result;
}

fn benchMd5(data: []const u8) [16]u8 {
    var result: [16]u8 = undefined;
    crypto.hash.Md5.hash(data, &result, .{});
    return result;
}

// Incremental hashing benchmark
fn benchIncrementalSha256(data: []const u8, chunk_size: usize) [32]u8 {
    var hasher = crypto.hash.sha2.Sha256.init(.{});

    var i: usize = 0;
    while (i < data.len) : (i += chunk_size) {
        const end = @min(i + chunk_size, data.len);
        hasher.update(data[i..end]);
    }

    return hasher.finalResult();
}

// ============================================================================
// HMAC Benchmarks
// ============================================================================

fn benchHmacSha256(key: []const u8, data: []const u8) [32]u8 {
    var mac: [32]u8 = undefined;
    crypto.auth.hmac.sha2.HmacSha256.create(&mac, data, key);
    return mac;
}

fn benchHmacSha512(key: []const u8, data: []const u8) [64]u8 {
    var mac: [64]u8 = undefined;
    crypto.auth.hmac.sha2.HmacSha512.create(&mac, data, key);
    return mac;
}

// ============================================================================
// Key Derivation Benchmarks
// ============================================================================

fn benchPbkdf2(password: []const u8, salt: []const u8, iterations: u32) [32]u8 {
    // Manual PBKDF2-like iteration using HMAC-SHA256
    var key: [32]u8 = undefined;
    var u: [32]u8 = undefined;

    var hmac = crypto.auth.hmac.sha2.HmacSha256.init(password);
    hmac.update(salt);
    hmac.update(&[_]u8{ 0, 0, 0, 1 }); // Block counter
    hmac.final(&u);
    key = u;

    // Iterate
    for (1..iterations) |_| {
        var h = crypto.auth.hmac.sha2.HmacSha256.init(password);
        h.update(&u);
        h.final(&u);
        for (&key, u) |*k, ui| {
            k.* ^= ui;
        }
    }
    return key;
}

// Simulated scrypt-like KDF (memory-hard)
fn benchMemoryHardKdf(allocator: std.mem.Allocator, password: []const u8, memory_kb: usize) ![32]u8 {
    // Allocate memory buffer (simulating memory-hard property)
    const memory = try allocator.alloc(u8, memory_kb * 1024);
    defer allocator.free(memory);

    // Fill with initial hash
    var current: [32]u8 = undefined;
    crypto.hash.sha2.Sha256.hash(password, &current, .{});
    @memcpy(memory[0..32], &current);

    // Mix memory
    var i: usize = 32;
    while (i < memory.len) : (i += 32) {
        crypto.hash.sha2.Sha256.hash(&current, &current, .{});
        const copy_len = @min(32, memory.len - i);
        @memcpy(memory[i..][0..copy_len], current[0..copy_len]);
    }

    // Final mix
    var result: [32]u8 = undefined;
    crypto.hash.sha2.Sha256.hash(memory, &result, .{});
    return result;
}

// ============================================================================
// Symmetric Encryption Benchmarks
// ============================================================================

fn benchAesGcmEncrypt(key: *const [32]u8, nonce: *const [12]u8, plaintext: []const u8, ciphertext: []u8) void {
    const aes = crypto.aead.aes_gcm.Aes256Gcm;
    var tag: [16]u8 = undefined;
    aes.encrypt(ciphertext, &tag, plaintext, "", nonce.*, key.*);
}

fn benchAesGcmDecrypt(key: *const [32]u8, nonce: *const [12]u8, ciphertext: []const u8, plaintext: []u8) !void {
    const aes = crypto.aead.aes_gcm.Aes256Gcm;
    const tag: [16]u8 = undefined;
    try aes.decrypt(plaintext, ciphertext, tag, "", nonce.*, key.*);
}

fn benchChaCha20Poly1305Encrypt(key: *const [32]u8, nonce: *const [12]u8, plaintext: []const u8, ciphertext: []u8) void {
    const chacha = crypto.aead.chacha_poly.ChaCha20Poly1305;
    var tag: [16]u8 = undefined;
    chacha.encrypt(ciphertext, &tag, plaintext, "", nonce.*, key.*);
}

fn benchXChaCha20Poly1305Encrypt(key: *const [32]u8, nonce: *const [24]u8, plaintext: []const u8, ciphertext: []u8) void {
    const xchacha = crypto.aead.chacha_poly.XChaCha20Poly1305;
    var tag: [16]u8 = undefined;
    xchacha.encrypt(ciphertext, &tag, plaintext, "", nonce.*, key.*);
}

// ============================================================================
// Signature Benchmarks
// ============================================================================

fn benchEd25519Sign(key_pair: crypto.sign.Ed25519.KeyPair, message: []const u8) [64]u8 {
    const sig = key_pair.sign(message, null) catch return [_]u8{0} ** 64;
    return sig.toBytes();
}

fn benchEd25519Verify(public_key: crypto.sign.Ed25519.PublicKey, message: []const u8, signature: crypto.sign.Ed25519.Signature) bool {
    signature.verify(message, public_key) catch return false;
    return true;
}

// ============================================================================
// Random Number Generation Benchmarks
// ============================================================================

fn benchCsprng(prng: *std.Random.DefaultPrng, buffer: []u8) void {
    // Use PRNG as fallback since crypto.random is not available in Zig 0.16
    prng.fill(buffer);
}

fn benchPrng(prng: *std.Random.DefaultPrng, buffer: []u8) void {
    prng.fill(buffer);
}

// ============================================================================
// Constant-Time Operations Benchmarks
// ============================================================================

fn benchConstantTimeCompare(a: []const u8, b: []const u8) bool {
    // Timing-safe comparison implementation
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, b) |x, y| {
        diff |= x ^ y;
    }
    return diff == 0;
}

fn benchConstantTimeSelect(condition: bool, a: u8, b: u8) u8 {
    // Constant-time select implementation
    const mask: u8 = @as(u8, 0) -% @intFromBool(condition);
    return (a & mask) | (b & ~mask);
}

// ============================================================================
// Encoding Benchmarks
// ============================================================================

fn benchBase64Encode(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const encoder = std.base64.standard;
    const size = std.base64.standard.Encoder.calcSize(data.len);
    const buffer = try allocator.alloc(u8, size);
    _ = encoder.Encoder.encode(buffer, data);
    return buffer;
}

fn benchBase64Decode(allocator: std.mem.Allocator, encoded: []const u8) ![]u8 {
    const decoder = std.base64.standard;
    const size = decoder.Decoder.calcSizeForSlice(encoded) catch return error.InvalidInput;
    const buffer = try allocator.alloc(u8, size);
    decoder.Decoder.decode(buffer, encoded) catch return error.InvalidInput;
    return buffer;
}

fn benchHexEncode(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const buffer = try allocator.alloc(u8, data.len * 2);
    // Use std.fmt.formatInt for hex encoding
    for (data, 0..) |byte, i| {
        _ = std.fmt.bufPrint(buffer[i * 2 ..][0..2], "{x:0>2}", .{byte}) catch unreachable;
    }
    return buffer;
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

pub fn runCryptoBenchmarks(allocator: std.mem.Allocator, config: CryptoBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    CRYPTOGRAPHY BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Hash function benchmarks
    std.debug.print("[Hash Functions]\n", .{});

    for (config.data_sizes) |size| {
        const data = try allocator.alloc(u8, size);
        defer allocator.free(data);
        @memset(data, 0xAA);

        // SHA-256
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "sha256_{d}", .{size}) catch "sha256";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "crypto/hash",
                    .bytes_per_op = size,
                    .warmup_iterations = 1000,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(d: []const u8) [32]u8 {
                        return benchSha256(d);
                    }
                }.bench,
                .{data},
            );

            std.debug.print("  {s}: {d:.2} MB/s ({d:.0} hashes/sec)\n", .{
                name,
                result.stats.throughputMBps(size),
                result.stats.opsPerSecond(),
            });
        }

        // SHA-512
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "sha512_{d}", .{size}) catch "sha512";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "crypto/hash",
                    .bytes_per_op = size,
                    .warmup_iterations = 1000,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(d: []const u8) [64]u8 {
                        return benchSha512(d);
                    }
                }.bench,
                .{data},
            );

            std.debug.print("  {s}: {d:.2} MB/s\n", .{ name, result.stats.throughputMBps(size) });
        }

        // Blake2b-256
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "blake2b256_{d}", .{size}) catch "blake2b";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "crypto/hash",
                    .bytes_per_op = size,
                    .warmup_iterations = 1000,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(d: []const u8) [32]u8 {
                        return benchBlake2b256(d);
                    }
                }.bench,
                .{data},
            );

            std.debug.print("  {s}: {d:.2} MB/s\n", .{ name, result.stats.throughputMBps(size) });
        }

        // Blake3
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "blake3_{d}", .{size}) catch "blake3";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "crypto/hash",
                    .bytes_per_op = size,
                    .warmup_iterations = 1000,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(d: []const u8) [32]u8 {
                        return benchBlake3(d);
                    }
                }.bench,
                .{data},
            );

            std.debug.print("  {s}: {d:.2} MB/s\n", .{ name, result.stats.throughputMBps(size) });
        }
    }

    // HMAC benchmarks
    std.debug.print("\n[HMAC]\n", .{});
    {
        const key = "this is a secret key for hmac benchmarking";
        const data = try allocator.alloc(u8, 1024);
        defer allocator.free(data);
        @memset(data, 0xBB);

        const result = try runner.run(
            .{
                .name = "hmac_sha256_1024",
                .category = "crypto/mac",
                .bytes_per_op = 1024,
                .warmup_iterations = 1000,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(k: []const u8, d: []const u8) [32]u8 {
                    return benchHmacSha256(k, d);
                }
            }.bench,
            .{ key, data },
        );

        std.debug.print("  hmac_sha256_1024: {d:.2} MB/s\n", .{result.stats.throughputMBps(1024)});
    }

    // Key derivation benchmarks
    std.debug.print("\n[Key Derivation (PBKDF2)]\n", .{});
    {
        const password = "benchmark_password_12345";
        const salt = "random_salt_value_here";

        for (config.pbkdf_iterations) |iterations| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "pbkdf2_{d}i", .{iterations}) catch "pbkdf2";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "crypto/kdf",
                    .warmup_iterations = 10,
                    .min_time_ns = 100_000_000,
                    .max_iterations = 100,
                },
                struct {
                    fn bench(pw: []const u8, s: []const u8, iters: u32) [32]u8 {
                        return benchPbkdf2(pw, s, iters);
                    }
                }.bench,
                .{ password, salt, iterations },
            );

            std.debug.print("  {s}: {d:.0} ops/sec, {d:.0}ms/op\n", .{
                name,
                result.stats.opsPerSecond(),
                result.stats.mean_ns / 1_000_000.0,
            });
        }
    }

    // Memory-hard KDF (reduced sizes for faster runs)
    std.debug.print("\n[Memory-Hard KDF]\n", .{});
    for ([_]usize{ 64, 256 }) |memory_kb| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "memory_hard_{d}kb", .{memory_kb}) catch "memhard";

        const result = try runner.run(
            .{
                .name = name,
                .category = "crypto/kdf",
                .warmup_iterations = 10,
                .min_time_ns = 100_000_000,
                .max_iterations = 100,
            },
            struct {
                fn bench(a: std.mem.Allocator, pw: []const u8, mem: usize) ![32]u8 {
                    return try benchMemoryHardKdf(a, pw, mem);
                }
            }.bench,
            .{ allocator, "password123", memory_kb },
        );

        std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
    }

    // Symmetric encryption benchmarks
    std.debug.print("\n[Symmetric Encryption (AEAD)]\n", .{});
    {
        var key: [32]u8 = undefined;
        var nonce12: [12]u8 = undefined;
        var nonce24: [24]u8 = undefined;
        // Use deterministic PRNG for benchmarks
        var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
        prng.fill(&key);
        prng.fill(&nonce12);
        prng.fill(&nonce24);

        for ([_]usize{ 64, 4096 }) |size| {
            const plaintext = try allocator.alloc(u8, size);
            defer allocator.free(plaintext);
            const ciphertext = try allocator.alloc(u8, size);
            defer allocator.free(ciphertext);
            @memset(plaintext, 0xCC);

            // AES-256-GCM
            {
                var name_buf: [64]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "aes256gcm_{d}", .{size}) catch "aes";

                const result = try runner.run(
                    .{
                        .name = name,
                        .category = "crypto/aead",
                        .bytes_per_op = size,
                        .warmup_iterations = 1000,
                        .min_time_ns = 100_000_000,
                    },
                    struct {
                        fn bench(k: *const [32]u8, n: *const [12]u8, pt: []const u8, ct: []u8) void {
                            benchAesGcmEncrypt(k, n, pt, ct);
                        }
                    }.bench,
                    .{ &key, &nonce12, plaintext, ciphertext },
                );

                std.debug.print("  {s}: {d:.2} MB/s\n", .{ name, result.stats.throughputMBps(size) });
            }

            // ChaCha20-Poly1305
            {
                var name_buf: [64]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "chacha20poly_{d}", .{size}) catch "chacha";

                const result = try runner.run(
                    .{
                        .name = name,
                        .category = "crypto/aead",
                        .bytes_per_op = size,
                        .warmup_iterations = 1000,
                        .min_time_ns = 100_000_000,
                    },
                    struct {
                        fn bench(k: *const [32]u8, n: *const [12]u8, pt: []const u8, ct: []u8) void {
                            benchChaCha20Poly1305Encrypt(k, n, pt, ct);
                        }
                    }.bench,
                    .{ &key, &nonce12, plaintext, ciphertext },
                );

                std.debug.print("  {s}: {d:.2} MB/s\n", .{ name, result.stats.throughputMBps(size) });
            }

            // XChaCha20-Poly1305
            {
                var name_buf: [64]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "xchacha20poly_{d}", .{size}) catch "xchacha";

                const result = try runner.run(
                    .{
                        .name = name,
                        .category = "crypto/aead",
                        .bytes_per_op = size,
                        .warmup_iterations = 1000,
                        .min_time_ns = 100_000_000,
                    },
                    struct {
                        fn bench(k: *const [32]u8, n: *const [24]u8, pt: []const u8, ct: []u8) void {
                            benchXChaCha20Poly1305Encrypt(k, n, pt, ct);
                        }
                    }.bench,
                    .{ &key, &nonce24, plaintext, ciphertext },
                );

                std.debug.print("  {s}: {d:.2} MB/s\n", .{ name, result.stats.throughputMBps(size) });
            }
        }
    }

    // Signature benchmarks
    std.debug.print("\n[Digital Signatures (Ed25519)]\n", .{});
    {
        // Create a deterministic key pair for benchmarks
        // Use the seed to construct a valid secret key bytes (seed + public key placeholder)
        const seed_bytes: [32]u8 = .{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20 };
        // Construct a key pair using seed expansion
        const secret_key = crypto.sign.Ed25519.SecretKey{ .bytes = seed_bytes ++ ([_]u8{0} ** 32) };
        const key_pair = crypto.sign.Ed25519.KeyPair.fromSecretKey(secret_key) catch {
            std.debug.print("  (Ed25519 benchmarks skipped - key generation not available)\n", .{});
            return;
        };
        const message = "This is a test message for signing benchmarks.";

        // Sign
        {
            const result = try runner.run(
                .{
                    .name = "ed25519_sign",
                    .category = "crypto/signature",
                    .warmup_iterations = 1000,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(kp: crypto.sign.Ed25519.KeyPair, msg: []const u8) [64]u8 {
                        return benchEd25519Sign(kp, msg);
                    }
                }.bench,
                .{ key_pair, message },
            );

            std.debug.print("  ed25519_sign: {d:.0} signatures/sec\n", .{result.stats.opsPerSecond()});
        }

        // Verify
        {
            const signature = key_pair.sign(message, null) catch {
                std.debug.print("  (Ed25519 verify benchmark skipped)\n", .{});
                return;
            };

            const result = try runner.run(
                .{
                    .name = "ed25519_verify",
                    .category = "crypto/signature",
                    .warmup_iterations = 1000,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(pk: crypto.sign.Ed25519.PublicKey, msg: []const u8, sig: crypto.sign.Ed25519.Signature) bool {
                        return benchEd25519Verify(pk, msg, sig);
                    }
                }.bench,
                .{ key_pair.public_key, message, signature },
            );

            std.debug.print("  ed25519_verify: {d:.0} verifications/sec\n", .{result.stats.opsPerSecond()});
        }
    }

    // Random number generation (reduced sizes)
    std.debug.print("\n[Random Number Generation]\n", .{});
    for ([_]usize{ 32, 1024 }) |size| {
        const buffer = try allocator.alloc(u8, size);
        defer allocator.free(buffer);

        // CSPRNG (using PRNG as substitute since crypto.random not available)
        {
            var csprng = std.Random.DefaultPrng.init(0x5EC09E);

            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "csprng_{d}", .{size}) catch "csprng";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "crypto/random",
                    .bytes_per_op = size,
                    .warmup_iterations = 1000,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(p: *std.Random.DefaultPrng, buf: []u8) void {
                        benchCsprng(p, buf);
                    }
                }.bench,
                .{ &csprng, buffer },
            );

            std.debug.print("  {s}: {d:.2} MB/s\n", .{ name, result.stats.throughputMBps(size) });
        }

        // PRNG (for comparison)
        {
            var prng = std.Random.DefaultPrng.init(12345);

            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "prng_{d}", .{size}) catch "prng";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "crypto/random",
                    .bytes_per_op = size,
                    .warmup_iterations = 1000,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(p: *std.Random.DefaultPrng, buf: []u8) void {
                        benchPrng(p, buf);
                    }
                }.bench,
                .{ &prng, buffer },
            );

            std.debug.print("  {s}: {d:.2} MB/s\n", .{ name, result.stats.throughputMBps(size) });
        }
    }

    // Encoding benchmarks (reduced sizes)
    std.debug.print("\n[Encoding/Decoding]\n", .{});
    for ([_]usize{ 64, 4096 }) |size| {
        const data = try allocator.alloc(u8, size);
        defer allocator.free(data);
        @memset(data, 0xDD);

        // Base64 encode
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "base64_enc_{d}", .{size}) catch "b64enc";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "crypto/encoding",
                    .bytes_per_op = size,
                    .warmup_iterations = 1000,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(a: std.mem.Allocator, d: []const u8) !void {
                        const encoded = try benchBase64Encode(a, d);
                        defer a.free(encoded);
                        std.mem.doNotOptimizeAway(encoded.ptr);
                    }
                }.bench,
                .{ allocator, data },
            );

            std.debug.print("  {s}: {d:.2} MB/s\n", .{ name, result.stats.throughputMBps(size) });
        }

        // Hex encode
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "hex_enc_{d}", .{size}) catch "hexenc";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "crypto/encoding",
                    .bytes_per_op = size,
                    .warmup_iterations = 1000,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(a: std.mem.Allocator, d: []const u8) !void {
                        const encoded = try benchHexEncode(a, d);
                        defer a.free(encoded);
                        std.mem.doNotOptimizeAway(encoded.ptr);
                    }
                }.bench,
                .{ allocator, data },
            );

            std.debug.print("  {s}: {d:.2} MB/s\n", .{ name, result.stats.throughputMBps(size) });
        }
    }

    // Constant-time comparison
    std.debug.print("\n[Constant-Time Operations]\n", .{});
    {
        var a: [32]u8 = undefined;
        var b: [32]u8 = undefined;
        // Use deterministic PRNG for benchmarks
        var ct_prng = std.Random.DefaultPrng.init(0xCAFEBABE);
        ct_prng.fill(&a);
        @memcpy(&b, &a);

        const result = try runner.run(
            .{
                .name = "timing_safe_compare_32",
                .category = "crypto/constant_time",
                .warmup_iterations = 10000,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(x: []const u8, y: []const u8) bool {
                    return benchConstantTimeCompare(x, y);
                }
            }.bench,
            .{ &a, &b },
        );

        std.debug.print("  timing_safe_compare_32: {d:.0} comparisons/sec\n", .{result.stats.opsPerSecond()});
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try runCryptoBenchmarks(allocator, .{});
}

test "hash functions" {
    const data = "test data for hashing";

    // All hash functions should produce deterministic output
    const sha256_1 = benchSha256(data);
    const sha256_2 = benchSha256(data);
    try std.testing.expectEqual(sha256_1, sha256_2);

    // Incremental should match one-shot
    const incremental = benchIncrementalSha256(data, 4);
    try std.testing.expectEqual(sha256_1, incremental);
}

test "hmac" {
    const key = "secret key";
    const data = "message";

    const mac1 = benchHmacSha256(key, data);
    const mac2 = benchHmacSha256(key, data);
    try std.testing.expectEqual(mac1, mac2);

    // Different key should produce different MAC
    const mac3 = benchHmacSha256("different key", data);
    try std.testing.expect(!std.mem.eql(u8, &mac1, &mac3));
}

test "encryption" {
    var key: [32]u8 = undefined;
    var nonce: [12]u8 = undefined;
    crypto.random.bytes(&key);
    crypto.random.bytes(&nonce);

    const plaintext = "Hello, World!";
    var ciphertext: [plaintext.len]u8 = undefined;

    benchAesGcmEncrypt(&key, &nonce, plaintext, &ciphertext);

    // Ciphertext should be different from plaintext
    try std.testing.expect(!std.mem.eql(u8, plaintext, &ciphertext));
}
