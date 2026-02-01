//! Secure password hashing module with modern algorithms.
//!
//! This module provides:
//! - Argon2id (recommended, memory-hard)
//! - PBKDF2-SHA256 (fallback, widely compatible)
//! - scrypt (alternative memory-hard)
//! - Password strength validation
//! - Secure comparison
//! - Hash format detection and migration

const std = @import("std");
const crypto = std.crypto;

/// Password hashing algorithm selection
pub const Algorithm = enum {
    /// Argon2id - recommended for most use cases (memory-hard, GPU-resistant)
    argon2id,
    /// PBKDF2-SHA256 - widely compatible fallback
    pbkdf2_sha256,
    /// PBKDF2-SHA512 - stronger PBKDF2 variant
    pbkdf2_sha512,
    /// scrypt - alternative memory-hard function
    scrypt,
    /// Blake3-based KDF - fast but less tested
    blake3_kdf,

    pub fn toString(self: Algorithm) []const u8 {
        return switch (self) {
            .argon2id => "argon2id",
            .pbkdf2_sha256 => "pbkdf2-sha256",
            .pbkdf2_sha512 => "pbkdf2-sha512",
            .scrypt => "scrypt",
            .blake3_kdf => "blake3-kdf",
        };
    }
};

/// Argon2 parameters for different security levels
pub const Argon2Params = struct {
    /// Time cost (iterations)
    time_cost: u32 = 3,
    /// Memory cost in KiB
    memory_cost: u32 = 65536, // 64 MiB
    /// Parallelism (lanes)
    parallelism: u8 = 4,
    /// Output hash length
    hash_length: usize = 32,
    /// Salt length
    salt_length: usize = 16,

    /// Preset for interactive logins (faster, less secure)
    pub const interactive: Argon2Params = .{
        .time_cost = 2,
        .memory_cost = 19456, // ~19 MiB
        .parallelism = 1,
    };

    /// Preset for moderate security (balanced)
    pub const moderate: Argon2Params = .{
        .time_cost = 3,
        .memory_cost = 65536, // 64 MiB
        .parallelism = 4,
    };

    /// Preset for sensitive operations (slower, more secure)
    pub const sensitive: Argon2Params = .{
        .time_cost = 4,
        .memory_cost = 131072, // 128 MiB
        .parallelism = 4,
    };
};

/// PBKDF2 parameters
pub const Pbkdf2Params = struct {
    /// Number of iterations (higher = more secure but slower)
    iterations: u32 = 600_000, // OWASP 2023 recommendation
    /// Output hash length
    hash_length: usize = 32,
    /// Salt length
    salt_length: usize = 16,

    /// Preset for interactive use
    pub const interactive: Pbkdf2Params = .{
        .iterations = 310_000,
    };

    /// Preset for sensitive data
    pub const sensitive: Pbkdf2Params = .{
        .iterations = 1_000_000,
    };
};

/// scrypt parameters
pub const ScryptParams = struct {
    /// CPU/memory cost parameter (N)
    log_n: u6 = 15, // 2^15 = 32768
    /// Block size (r)
    r: u30 = 8,
    /// Parallelization (p)
    p: u30 = 1,
    /// Output hash length
    hash_length: usize = 32,
    /// Salt length
    salt_length: usize = 16,

    /// Preset for interactive use
    pub const interactive: ScryptParams = .{
        .log_n = 14,
        .r = 8,
        .p = 1,
    };

    /// Preset for sensitive data
    pub const sensitive: ScryptParams = .{
        .log_n = 20,
        .r = 8,
        .p = 1,
    };
};

/// Password hasher configuration
pub const PasswordConfig = struct {
    /// Algorithm to use for new hashes
    algorithm: Algorithm = .argon2id,
    /// Argon2 parameters
    argon2: Argon2Params = .{},
    /// PBKDF2 parameters
    pbkdf2: Pbkdf2Params = .{},
    /// scrypt parameters
    scrypt: ScryptParams = .{},
    /// Minimum password length
    min_password_length: usize = 8,
    /// Maximum password length (to prevent DoS)
    max_password_length: usize = 128,
    /// Require password strength check
    require_strength_check: bool = true,
    /// Minimum required strength level
    min_strength: PasswordStrength = .fair,
    /// Auto-migrate old hash formats
    auto_migrate: bool = true,
};

/// Password strength levels
pub const PasswordStrength = enum(u8) {
    very_weak = 0,
    weak = 1,
    fair = 2,
    strong = 3,
    very_strong = 4,

    pub fn toString(self: PasswordStrength) []const u8 {
        return switch (self) {
            .very_weak => "Very Weak",
            .weak => "Weak",
            .fair => "Fair",
            .strong => "Strong",
            .very_strong => "Very Strong",
        };
    }

    pub fn meetsMinimum(self: PasswordStrength, minimum: PasswordStrength) bool {
        return @intFromEnum(self) >= @intFromEnum(minimum);
    }
};

/// Password strength analysis result
pub const StrengthAnalysis = struct {
    strength: PasswordStrength,
    score: u32,
    feedback: []const []const u8,
    has_lowercase: bool,
    has_uppercase: bool,
    has_digits: bool,
    has_special: bool,
    has_common_pattern: bool,
    estimated_crack_time: []const u8,
};

/// Hashed password with metadata
pub const HashedPassword = struct {
    /// The algorithm used
    algorithm: Algorithm,
    /// The derived hash
    hash: []const u8,
    /// The salt used
    salt: []const u8,
    /// Algorithm parameters encoded
    params: []const u8,
    /// Full encoded string (PHC format)
    encoded: []const u8,

    pub fn deinit(self: *HashedPassword, allocator: std.mem.Allocator) void {
        // Securely wipe sensitive data
        crypto.secureZero(u8, @constCast(self.hash));
        crypto.secureZero(u8, @constCast(self.salt));
        allocator.free(self.hash);
        allocator.free(self.salt);
        allocator.free(self.params);
        allocator.free(self.encoded);
    }
};

/// Password hasher
pub const PasswordHasher = struct {
    allocator: std.mem.Allocator,
    config: PasswordConfig,

    pub fn init(allocator: std.mem.Allocator, config: PasswordConfig) PasswordHasher {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Hash a password with the configured algorithm
    pub fn hash(self: *PasswordHasher, password: []const u8) !HashedPassword {
        // Validate password length
        if (password.len < self.config.min_password_length) {
            return error.PasswordTooShort;
        }
        if (password.len > self.config.max_password_length) {
            return error.PasswordTooLong;
        }

        // Check password strength if required
        if (self.config.require_strength_check) {
            const analysis = analyzeStrength(password);
            if (!analysis.strength.meetsMinimum(self.config.min_strength)) {
                return error.PasswordTooWeak;
            }
        }

        return switch (self.config.algorithm) {
            .argon2id => self.hashArgon2(password),
            .pbkdf2_sha256 => self.hashPbkdf2Sha256(password),
            .pbkdf2_sha512 => self.hashPbkdf2Sha512(password),
            .scrypt => self.hashScrypt(password),
            .blake3_kdf => self.hashBlake3(password),
        };
    }

    /// Verify a password against a hash
    pub fn verify(self: *PasswordHasher, password: []const u8, encoded: []const u8) !bool {
        // Detect algorithm from encoded format
        const algorithm = detectAlgorithm(encoded) orelse return error.InvalidHashFormat;

        return switch (algorithm) {
            .argon2id => self.verifyArgon2(password, encoded),
            .pbkdf2_sha256 => self.verifyPbkdf2(password, encoded, .sha256),
            .pbkdf2_sha512 => self.verifyPbkdf2(password, encoded, .sha512),
            .scrypt => self.verifyScrypt(password, encoded),
            .blake3_kdf => self.verifyBlake3(password, encoded),
        };
    }

    /// Check if a hash needs to be upgraded to current settings
    pub fn needsRehash(self: *PasswordHasher, encoded: []const u8) bool {
        const algorithm = detectAlgorithm(encoded) orelse return true;

        // Different algorithm = needs rehash
        if (algorithm != self.config.algorithm) return true;

        // Check if parameters are weaker than current config
        return switch (algorithm) {
            .argon2id => self.argon2NeedsRehash(encoded),
            .pbkdf2_sha256, .pbkdf2_sha512 => self.pbkdf2NeedsRehash(encoded),
            .scrypt => self.scryptNeedsRehash(encoded),
            .blake3_kdf => false, // Blake3 doesn't have adjustable work factor
        };
    }

    // Algorithm-specific implementations

    fn hashArgon2(self: *PasswordHasher, password: []const u8) !HashedPassword {
        const params = self.config.argon2;

        // Generate salt
        var salt: [32]u8 = undefined;
        const salt_slice = salt[0..params.salt_length];
        crypto.random.bytes(salt_slice);

        // Derive hash using Argon2id
        var derived: [64]u8 = undefined;
        const hash_slice = derived[0..params.hash_length];

        crypto.pwhash.argon2.kdf(
            hash_slice,
            password,
            salt_slice,
            .{
                .t = params.time_cost,
                .m = params.memory_cost,
                .p = params.parallelism,
            },
            .argon2id,
        ) catch return error.HashingFailed;

        // Encode to PHC format: $argon2id$v=19$m=65536,t=3,p=4$<salt>$<hash>
        const b64_salt = try self.base64Encode(salt_slice);
        defer self.allocator.free(b64_salt);
        const b64_hash = try self.base64Encode(hash_slice);
        defer self.allocator.free(b64_hash);

        const encoded = try std.fmt.allocPrint(
            self.allocator,
            "$argon2id$v=19$m={d},t={d},p={d}${s}${s}",
            .{
                params.memory_cost,
                params.time_cost,
                params.parallelism,
                b64_salt,
                b64_hash,
            },
        );

        return .{
            .algorithm = .argon2id,
            .hash = try self.allocator.dupe(u8, hash_slice),
            .salt = try self.allocator.dupe(u8, salt_slice),
            .params = try std.fmt.allocPrint(self.allocator, "m={d},t={d},p={d}", .{
                params.memory_cost,
                params.time_cost,
                params.parallelism,
            }),
            .encoded = encoded,
        };
    }

    fn hashPbkdf2Sha256(self: *PasswordHasher, password: []const u8) !HashedPassword {
        const params = self.config.pbkdf2;

        // Generate salt
        var salt: [32]u8 = undefined;
        const salt_slice = salt[0..params.salt_length];
        crypto.random.bytes(salt_slice);

        // Derive hash
        var derived: [64]u8 = undefined;
        const hash_slice = derived[0..params.hash_length];

        crypto.pwhash.pbkdf2(hash_slice, password, salt_slice, params.iterations, .sha256);

        // Encode to format: $pbkdf2-sha256$i=600000$<salt>$<hash>
        const b64_salt = try self.base64Encode(salt_slice);
        defer self.allocator.free(b64_salt);
        const b64_hash = try self.base64Encode(hash_slice);
        defer self.allocator.free(b64_hash);

        const encoded = try std.fmt.allocPrint(
            self.allocator,
            "$pbkdf2-sha256$i={d}${s}${s}",
            .{ params.iterations, b64_salt, b64_hash },
        );

        return .{
            .algorithm = .pbkdf2_sha256,
            .hash = try self.allocator.dupe(u8, hash_slice),
            .salt = try self.allocator.dupe(u8, salt_slice),
            .params = try std.fmt.allocPrint(self.allocator, "i={d}", .{params.iterations}),
            .encoded = encoded,
        };
    }

    fn hashPbkdf2Sha512(self: *PasswordHasher, password: []const u8) !HashedPassword {
        const params = self.config.pbkdf2;

        var salt: [32]u8 = undefined;
        const salt_slice = salt[0..params.salt_length];
        crypto.random.bytes(salt_slice);

        var derived: [64]u8 = undefined;
        const hash_slice = derived[0..params.hash_length];

        crypto.pwhash.pbkdf2(hash_slice, password, salt_slice, params.iterations, .sha512);

        const b64_salt = try self.base64Encode(salt_slice);
        defer self.allocator.free(b64_salt);
        const b64_hash = try self.base64Encode(hash_slice);
        defer self.allocator.free(b64_hash);

        const encoded = try std.fmt.allocPrint(
            self.allocator,
            "$pbkdf2-sha512$i={d}${s}${s}",
            .{ params.iterations, b64_salt, b64_hash },
        );

        return .{
            .algorithm = .pbkdf2_sha512,
            .hash = try self.allocator.dupe(u8, hash_slice),
            .salt = try self.allocator.dupe(u8, salt_slice),
            .params = try std.fmt.allocPrint(self.allocator, "i={d}", .{params.iterations}),
            .encoded = encoded,
        };
    }

    fn hashScrypt(self: *PasswordHasher, password: []const u8) !HashedPassword {
        const params = self.config.scrypt;

        var salt: [32]u8 = undefined;
        const salt_slice = salt[0..params.salt_length];
        crypto.random.bytes(salt_slice);

        var derived: [64]u8 = undefined;
        const hash_slice = derived[0..params.hash_length];

        crypto.pwhash.scrypt.kdf(
            hash_slice,
            password,
            salt_slice,
            .{ .ln = params.log_n, .r = params.r, .p = params.p },
        ) catch return error.HashingFailed;

        const b64_salt = try self.base64Encode(salt_slice);
        defer self.allocator.free(b64_salt);
        const b64_hash = try self.base64Encode(hash_slice);
        defer self.allocator.free(b64_hash);

        const encoded = try std.fmt.allocPrint(
            self.allocator,
            "$scrypt$ln={d},r={d},p={d}${s}${s}",
            .{ params.log_n, params.r, params.p, b64_salt, b64_hash },
        );

        return .{
            .algorithm = .scrypt,
            .hash = try self.allocator.dupe(u8, hash_slice),
            .salt = try self.allocator.dupe(u8, salt_slice),
            .params = try std.fmt.allocPrint(self.allocator, "ln={d},r={d},p={d}", .{
                params.log_n,
                params.r,
                params.p,
            }),
            .encoded = encoded,
        };
    }

    fn hashBlake3(self: *PasswordHasher, password: []const u8) !HashedPassword {
        var salt: [16]u8 = undefined;
        crypto.random.bytes(&salt);

        // Use Blake3 in keyed mode with salt as context
        var hasher = crypto.hash.Blake3.init(.{});
        hasher.update(&salt);
        hasher.update(password);

        var derived: [32]u8 = undefined;
        hasher.final(&derived);

        const b64_salt = try self.base64Encode(&salt);
        defer self.allocator.free(b64_salt);
        const b64_hash = try self.base64Encode(&derived);
        defer self.allocator.free(b64_hash);

        const encoded = try std.fmt.allocPrint(
            self.allocator,
            "$blake3${s}${s}",
            .{ b64_salt, b64_hash },
        );

        return .{
            .algorithm = .blake3_kdf,
            .hash = try self.allocator.dupe(u8, &derived),
            .salt = try self.allocator.dupe(u8, &salt),
            .params = try self.allocator.dupe(u8, ""),
            .encoded = encoded,
        };
    }

    // Verification methods

    fn verifyArgon2(self: *PasswordHasher, password: []const u8, encoded: []const u8) !bool {
        _ = self;
        // Parse encoded string and verify
        const parsed = parseArgon2Encoded(encoded) orelse return error.InvalidHashFormat;

        var derived: [64]u8 = undefined;
        const hash_slice = derived[0..parsed.hash.len];

        crypto.pwhash.argon2.kdf(
            hash_slice,
            password,
            parsed.salt,
            .{
                .t = parsed.time_cost,
                .m = parsed.memory_cost,
                .p = parsed.parallelism,
            },
            .argon2id,
        ) catch return false;

        return crypto.utils.timingSafeEql(u8, hash_slice, parsed.hash);
    }

    fn verifyPbkdf2(
        self: *PasswordHasher,
        password: []const u8,
        encoded: []const u8,
        comptime variant: enum { sha256, sha512 },
    ) !bool {
        _ = self;
        const parsed = parsePbkdf2Encoded(encoded) orelse return error.InvalidHashFormat;

        var derived: [64]u8 = undefined;
        const hash_slice = derived[0..parsed.hash.len];

        const prf = if (variant == .sha256) .sha256 else .sha512;
        crypto.pwhash.pbkdf2(hash_slice, password, parsed.salt, parsed.iterations, prf);

        return crypto.utils.timingSafeEql(u8, hash_slice, parsed.hash);
    }

    fn verifyScrypt(self: *PasswordHasher, password: []const u8, encoded: []const u8) !bool {
        _ = self;
        const parsed = parseScryptEncoded(encoded) orelse return error.InvalidHashFormat;

        var derived: [64]u8 = undefined;
        const hash_slice = derived[0..parsed.hash.len];

        crypto.pwhash.scrypt.kdf(
            hash_slice,
            password,
            parsed.salt,
            .{ .ln = parsed.log_n, .r = parsed.r, .p = parsed.p },
        ) catch return false;

        return crypto.utils.timingSafeEql(u8, hash_slice, parsed.hash);
    }

    fn verifyBlake3(self: *PasswordHasher, password: []const u8, encoded: []const u8) !bool {
        _ = self;
        const parsed = parseBlake3Encoded(encoded) orelse return error.InvalidHashFormat;

        var hasher = crypto.hash.Blake3.init(.{});
        hasher.update(parsed.salt);
        hasher.update(password);

        var derived: [32]u8 = undefined;
        hasher.final(&derived);

        return crypto.utils.timingSafeEql(u8, &derived, parsed.hash);
    }

    // Rehash checking

    fn argon2NeedsRehash(self: *PasswordHasher, encoded: []const u8) bool {
        const parsed = parseArgon2Encoded(encoded) orelse return true;
        const cfg = self.config.argon2;

        return parsed.memory_cost < cfg.memory_cost or
            parsed.time_cost < cfg.time_cost or
            parsed.parallelism < cfg.parallelism;
    }

    fn pbkdf2NeedsRehash(self: *PasswordHasher, encoded: []const u8) bool {
        const parsed = parsePbkdf2Encoded(encoded) orelse return true;
        return parsed.iterations < self.config.pbkdf2.iterations;
    }

    fn scryptNeedsRehash(self: *PasswordHasher, encoded: []const u8) bool {
        const parsed = parseScryptEncoded(encoded) orelse return true;
        const cfg = self.config.scrypt;

        return parsed.log_n < cfg.log_n or
            parsed.r < cfg.r or
            parsed.p < cfg.p;
    }

    fn base64Encode(self: *PasswordHasher, data: []const u8) ![]const u8 {
        const encoder = std.base64.standard.Encoder;
        const size = encoder.calcSize(data.len);
        const buf = try self.allocator.alloc(u8, size);
        _ = encoder.encode(buf, data);
        return buf;
    }
};

// Parsing helpers

const ParsedArgon2 = struct {
    salt: []const u8,
    hash: []const u8,
    memory_cost: u32,
    time_cost: u32,
    parallelism: u8,
};

fn parseArgon2Encoded(encoded: []const u8) ?ParsedArgon2 {
    // Format: $argon2id$v=19$m=65536,t=3,p=4$<salt>$<hash>
    if (!std.mem.startsWith(u8, encoded, "$argon2id$")) return null;

    var parts = std.mem.splitScalar(u8, encoded, '$');

    // Skip empty first part (before first $)
    _ = parts.next() orelse return null;
    // Skip "argon2id"
    _ = parts.next() orelse return null;
    // Skip version "v=19"
    _ = parts.next() orelse return null;

    // Parse parameters "m=65536,t=3,p=4"
    const params_str = parts.next() orelse return null;
    var memory_cost: u32 = 0;
    var time_cost: u32 = 0;
    var parallelism: u8 = 0;

    var param_parts = std.mem.splitScalar(u8, params_str, ',');
    while (param_parts.next()) |param| {
        if (std.mem.startsWith(u8, param, "m=")) {
            memory_cost = std.fmt.parseInt(u32, param[2..], 10) catch return null;
        } else if (std.mem.startsWith(u8, param, "t=")) {
            time_cost = std.fmt.parseInt(u32, param[2..], 10) catch return null;
        } else if (std.mem.startsWith(u8, param, "p=")) {
            parallelism = std.fmt.parseInt(u8, param[2..], 10) catch return null;
        }
    }

    // Get salt (base64 encoded)
    const salt_b64 = parts.next() orelse return null;
    // Get hash (base64 encoded)
    const hash_b64 = parts.next() orelse return null;

    // Decode base64 - we need static buffers for the parsed result
    const decoder = std.base64.standard.Decoder;
    var salt_buf: [64]u8 = undefined;
    var hash_buf: [64]u8 = undefined;

    const salt_len = decoder.calcSizeForSlice(salt_b64) catch return null;
    const hash_len = decoder.calcSizeForSlice(hash_b64) catch return null;

    if (salt_len > salt_buf.len or hash_len > hash_buf.len) return null;

    decoder.decode(salt_buf[0..salt_len], salt_b64) catch return null;
    decoder.decode(hash_buf[0..hash_len], hash_b64) catch return null;

    return ParsedArgon2{
        .salt = salt_buf[0..salt_len],
        .hash = hash_buf[0..hash_len],
        .memory_cost = memory_cost,
        .time_cost = time_cost,
        .parallelism = parallelism,
    };
}

const ParsedPbkdf2 = struct {
    salt: []const u8,
    hash: []const u8,
    iterations: u32,
};

fn parsePbkdf2Encoded(encoded: []const u8) ?ParsedPbkdf2 {
    // Format: $pbkdf2-sha256$i=600000$<salt>$<hash>
    // or: $pbkdf2-sha512$i=600000$<salt>$<hash>
    if (!std.mem.startsWith(u8, encoded, "$pbkdf2-")) return null;

    var parts = std.mem.splitScalar(u8, encoded, '$');

    // Skip empty first part
    _ = parts.next() orelse return null;
    // Skip algorithm identifier (pbkdf2-sha256 or pbkdf2-sha512)
    _ = parts.next() orelse return null;

    // Parse iterations "i=600000"
    const iter_str = parts.next() orelse return null;
    var iterations: u32 = 0;
    if (std.mem.startsWith(u8, iter_str, "i=")) {
        iterations = std.fmt.parseInt(u32, iter_str[2..], 10) catch return null;
    } else {
        return null;
    }

    // Get salt (base64 encoded)
    const salt_b64 = parts.next() orelse return null;
    // Get hash (base64 encoded)
    const hash_b64 = parts.next() orelse return null;

    // Decode base64
    const decoder = std.base64.standard.Decoder;
    var salt_buf: [64]u8 = undefined;
    var hash_buf: [64]u8 = undefined;

    const salt_len = decoder.calcSizeForSlice(salt_b64) catch return null;
    const hash_len = decoder.calcSizeForSlice(hash_b64) catch return null;

    if (salt_len > salt_buf.len or hash_len > hash_buf.len) return null;

    decoder.decode(salt_buf[0..salt_len], salt_b64) catch return null;
    decoder.decode(hash_buf[0..hash_len], hash_b64) catch return null;

    return ParsedPbkdf2{
        .salt = salt_buf[0..salt_len],
        .hash = hash_buf[0..hash_len],
        .iterations = iterations,
    };
}

const ParsedScrypt = struct {
    salt: []const u8,
    hash: []const u8,
    log_n: u6,
    r: u30,
    p: u30,
};

fn parseScryptEncoded(encoded: []const u8) ?ParsedScrypt {
    // Format: $scrypt$ln=15,r=8,p=1$<salt>$<hash>
    if (!std.mem.startsWith(u8, encoded, "$scrypt$")) return null;

    var parts = std.mem.splitScalar(u8, encoded, '$');

    // Skip empty first part
    _ = parts.next() orelse return null;
    // Skip "scrypt"
    _ = parts.next() orelse return null;

    // Parse parameters "ln=15,r=8,p=1"
    const params_str = parts.next() orelse return null;
    var log_n: u6 = 0;
    var r: u30 = 0;
    var p: u30 = 0;

    var param_parts = std.mem.splitScalar(u8, params_str, ',');
    while (param_parts.next()) |param| {
        if (std.mem.startsWith(u8, param, "ln=")) {
            log_n = std.fmt.parseInt(u6, param[3..], 10) catch return null;
        } else if (std.mem.startsWith(u8, param, "r=")) {
            r = std.fmt.parseInt(u30, param[2..], 10) catch return null;
        } else if (std.mem.startsWith(u8, param, "p=")) {
            p = std.fmt.parseInt(u30, param[2..], 10) catch return null;
        }
    }

    // Get salt (base64 encoded)
    const salt_b64 = parts.next() orelse return null;
    // Get hash (base64 encoded)
    const hash_b64 = parts.next() orelse return null;

    // Decode base64
    const decoder = std.base64.standard.Decoder;
    var salt_buf: [64]u8 = undefined;
    var hash_buf: [64]u8 = undefined;

    const salt_len = decoder.calcSizeForSlice(salt_b64) catch return null;
    const hash_len = decoder.calcSizeForSlice(hash_b64) catch return null;

    if (salt_len > salt_buf.len or hash_len > hash_buf.len) return null;

    decoder.decode(salt_buf[0..salt_len], salt_b64) catch return null;
    decoder.decode(hash_buf[0..hash_len], hash_b64) catch return null;

    return ParsedScrypt{
        .salt = salt_buf[0..salt_len],
        .hash = hash_buf[0..hash_len],
        .log_n = log_n,
        .r = r,
        .p = p,
    };
}

const ParsedBlake3 = struct {
    salt: []const u8,
    hash: []const u8,
};

fn parseBlake3Encoded(encoded: []const u8) ?ParsedBlake3 {
    // Format: $blake3$<salt>$<hash>
    if (!std.mem.startsWith(u8, encoded, "$blake3$")) return null;

    var parts = std.mem.splitScalar(u8, encoded, '$');

    // Skip empty first part
    _ = parts.next() orelse return null;
    // Skip "blake3"
    _ = parts.next() orelse return null;

    // Get salt (base64 encoded)
    const salt_b64 = parts.next() orelse return null;
    // Get hash (base64 encoded)
    const hash_b64 = parts.next() orelse return null;

    // Decode base64
    const decoder = std.base64.standard.Decoder;
    var salt_buf: [32]u8 = undefined;
    var hash_buf: [32]u8 = undefined;

    const salt_len = decoder.calcSizeForSlice(salt_b64) catch return null;
    const hash_len = decoder.calcSizeForSlice(hash_b64) catch return null;

    if (salt_len > salt_buf.len or hash_len > hash_buf.len) return null;

    decoder.decode(salt_buf[0..salt_len], salt_b64) catch return null;
    decoder.decode(hash_buf[0..hash_len], hash_b64) catch return null;

    return ParsedBlake3{
        .salt = salt_buf[0..salt_len],
        .hash = hash_buf[0..hash_len],
    };
}

fn detectAlgorithm(encoded: []const u8) ?Algorithm {
    if (std.mem.startsWith(u8, encoded, "$argon2id$")) return .argon2id;
    if (std.mem.startsWith(u8, encoded, "$pbkdf2-sha256$")) return .pbkdf2_sha256;
    if (std.mem.startsWith(u8, encoded, "$pbkdf2-sha512$")) return .pbkdf2_sha512;
    if (std.mem.startsWith(u8, encoded, "$scrypt$")) return .scrypt;
    if (std.mem.startsWith(u8, encoded, "$blake3$")) return .blake3_kdf;
    return null;
}

const ClassFlags = struct {
    has_lower: bool,
    has_upper: bool,
    has_digit: bool,
    has_special: bool,
};

fn scoreLength(len: usize) u32 {
    var score: u32 = 0;
    if (len >= 8) score += 10;
    if (len >= 12) score += 10;
    if (len >= 16) score += 10;
    if (len >= 20) score += 10;
    return score;
}

fn analyzeClasses(password: []const u8) ClassFlags {
    var flags = ClassFlags{
        .has_lower = false,
        .has_upper = false,
        .has_digit = false,
        .has_special = false,
    };

    for (password) |c| {
        if (c >= 'a' and c <= 'z') flags.has_lower = true;
        if (c >= 'A' and c <= 'Z') flags.has_upper = true;
        if (c >= '0' and c <= '9') flags.has_digit = true;
        if ((c >= '!' and c <= '/') or (c >= ':' and c <= '@') or
            (c >= '[' and c <= '`') or (c >= '{' and c <= '~'))
        {
            flags.has_special = true;
        }
    }

    return flags;
}

fn appendFeedback(feedback: *std.BoundedArray([]const u8, 10), message: []const u8) void {
    feedback.append(message) catch {};
}

fn scoreClasses(feedback: *std.BoundedArray([]const u8, 10), flags: ClassFlags) u32 {
    var score: u32 = 0;

    if (flags.has_lower) score += 10 else appendFeedback(feedback, "Add lowercase letters");
    if (flags.has_upper) score += 10 else appendFeedback(feedback, "Add uppercase letters");
    if (flags.has_digit) score += 10 else appendFeedback(feedback, "Add numbers");
    if (flags.has_special) score += 15 else appendFeedback(feedback, "Add special characters");

    return score;
}

fn applyPenalty(feedback: *std.BoundedArray([]const u8, 10), score: *u32, condition: bool, penalty: u32, message: []const u8) void {
    if (!condition) return;
    score.* -|= penalty;
    appendFeedback(feedback, message);
}

fn strengthFromScore(score: u32) PasswordStrength {
    return if (score >= 60)
        .very_strong
    else if (score >= 45)
        .strong
    else if (score >= 30)
        .fair
    else if (score >= 15)
        .weak
    else
        .very_weak;
}

fn crackTimeFromScore(score: u32) []const u8 {
    return if (score >= 60)
        "centuries"
    else if (score >= 45)
        "years"
    else if (score >= 30)
        "months"
    else if (score >= 15)
        "days"
    else
        "instant";
}

/// Analyze password strength
pub fn analyzeStrength(password: []const u8) StrengthAnalysis {
    var score: u32 = scoreLength(password.len);
    var feedback = std.BoundedArray([]const u8, 10){};

    // Character class analysis
    const class_flags = analyzeClasses(password);
    score += scoreClasses(&feedback, class_flags);

    // Check for common patterns
    const has_common = containsCommonPattern(password);
    applyPenalty(&feedback, &score, has_common, 20, "Avoid common patterns");

    // Sequential characters penalty
    applyPenalty(&feedback, &score, hasSequentialChars(password), 10, "Avoid sequential characters");

    // Repeated characters penalty
    applyPenalty(&feedback, &score, hasRepeatedChars(password), 10, "Avoid repeated characters");

    // Determine strength level
    const strength = strengthFromScore(score);

    // Estimate crack time (simplified)
    const crack_time = crackTimeFromScore(score);

    return .{
        .strength = strength,
        .score = score,
        .feedback = feedback.slice(),
        .has_lowercase = class_flags.has_lower,
        .has_uppercase = class_flags.has_upper,
        .has_digits = class_flags.has_digit,
        .has_special = class_flags.has_special,
        .has_common_pattern = has_common,
        .estimated_crack_time = crack_time,
    };
}

fn containsCommonPattern(password: []const u8) bool {
    const common_patterns = &[_][]const u8{
        "password", "123456", "qwerty",   "abc123",   "letmein",
        "welcome",  "monkey", "dragon",   "master",   "login",
        "admin",    "root",   "pass",     "test",     "guest",
        "hello",    "shadow", "sunshine", "princess", "football",
    };

    var lower_buf: [128]u8 = undefined;
    const lower = std.ascii.lowerString(lower_buf[0..@min(password.len, 128)], password);

    for (common_patterns) |pattern| {
        if (std.mem.indexOf(u8, lower, pattern) != null) {
            return true;
        }
    }

    return false;
}

fn hasSequentialChars(password: []const u8) bool {
    if (password.len < 3) return false;

    for (0..password.len - 2) |i| {
        const a = password[i];
        const b = password[i + 1];
        const c = password[i + 2];

        // Ascending sequence
        if (b == a + 1 and c == b + 1) return true;
        // Descending sequence
        if (b == a -% 1 and c == b -% 1 and a > 0 and b > 0) return true;
    }

    return false;
}

fn hasRepeatedChars(password: []const u8) bool {
    if (password.len < 3) return false;

    for (0..password.len - 2) |i| {
        if (password[i] == password[i + 1] and password[i + 1] == password[i + 2]) {
            return true;
        }
    }

    return false;
}

/// Generate a secure random password
pub fn generatePassword(allocator: std.mem.Allocator, length: usize, options: GenerateOptions) ![]u8 {
    if (length < 8) return error.PasswordTooShort;
    if (length > 128) return error.PasswordTooLong;

    var charset = std.ArrayListUnmanaged(u8).empty;
    defer charset.deinit(allocator);

    if (options.include_lowercase) {
        try charset.appendSlice(allocator, "abcdefghijklmnopqrstuvwxyz");
    }
    if (options.include_uppercase) {
        try charset.appendSlice(allocator, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }
    if (options.include_digits) {
        try charset.appendSlice(allocator, "0123456789");
    }
    if (options.include_special) {
        try charset.appendSlice(allocator, "!@#$%^&*()_+-=[]{}|;:,.<>?");
    }

    if (charset.items.len == 0) {
        return error.InvalidOptions;
    }

    const result = try allocator.alloc(u8, length);
    errdefer allocator.free(result);

    // Fill with random characters
    for (result) |*c| {
        const idx = crypto.random.uintLessThan(usize, charset.items.len);
        c.* = charset.items[idx];
    }

    // Ensure at least one character from each required class
    var pos: usize = 0;
    if (options.include_lowercase and options.require_all_classes) {
        result[pos] = "abcdefghijklmnopqrstuvwxyz"[crypto.random.uintLessThan(usize, 26)];
        pos += 1;
    }
    if (options.include_uppercase and options.require_all_classes and pos < length) {
        result[pos] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[crypto.random.uintLessThan(usize, 26)];
        pos += 1;
    }
    if (options.include_digits and options.require_all_classes and pos < length) {
        result[pos] = "0123456789"[crypto.random.uintLessThan(usize, 10)];
        pos += 1;
    }
    if (options.include_special and options.require_all_classes and pos < length) {
        const special = "!@#$%^&*()_+-=[]{}|;:,.<>?";
        result[pos] = special[crypto.random.uintLessThan(usize, special.len)];
        pos += 1;
    }

    // Shuffle the result
    crypto.random.shuffle(u8, result);

    return result;
}

pub const GenerateOptions = struct {
    include_lowercase: bool = true,
    include_uppercase: bool = true,
    include_digits: bool = true,
    include_special: bool = true,
    require_all_classes: bool = true,
    exclude_ambiguous: bool = false, // Exclude 0, O, l, 1, etc.
};

// Error types
pub const PasswordError = error{
    PasswordTooShort,
    PasswordTooLong,
    PasswordTooWeak,
    InvalidHashFormat,
    HashingFailed,
    VerificationFailed,
    InvalidOptions,
    OutOfMemory,
};

// Tests

test "password strength analysis" {
    // Weak password
    const weak = analyzeStrength("password");
    try std.testing.expect(weak.strength == .very_weak or weak.strength == .weak);
    try std.testing.expect(weak.has_common_pattern);

    // Strong password
    const strong = analyzeStrength("MyStr0ng!P@ssw0rd#2024");
    try std.testing.expect(strong.strength == .strong or strong.strength == .very_strong);
    try std.testing.expect(strong.has_lowercase);
    try std.testing.expect(strong.has_uppercase);
    try std.testing.expect(strong.has_digits);
    try std.testing.expect(strong.has_special);
}

test "password generation" {
    const allocator = std.testing.allocator;

    const password = try generatePassword(allocator, 16, .{});
    defer allocator.free(password);

    try std.testing.expectEqual(@as(usize, 16), password.len);

    // Check for required character classes
    var has_lower = false;
    var has_upper = false;
    var has_digit = false;
    var has_special = false;

    for (password) |c| {
        if (c >= 'a' and c <= 'z') has_lower = true;
        if (c >= 'A' and c <= 'Z') has_upper = true;
        if (c >= '0' and c <= '9') has_digit = true;
        if (!std.ascii.isAlphanumeric(c)) has_special = true;
    }

    try std.testing.expect(has_lower);
    try std.testing.expect(has_upper);
    try std.testing.expect(has_digit);
    try std.testing.expect(has_special);
}

test "sequential chars detection" {
    try std.testing.expect(hasSequentialChars("abc"));
    try std.testing.expect(hasSequentialChars("123"));
    try std.testing.expect(hasSequentialChars("xyz"));
    try std.testing.expect(!hasSequentialChars("aZb"));
    try std.testing.expect(!hasSequentialChars("a1b"));
}

test "repeated chars detection" {
    try std.testing.expect(hasRepeatedChars("aaa"));
    try std.testing.expect(hasRepeatedChars("111"));
    try std.testing.expect(!hasRepeatedChars("aba"));
    try std.testing.expect(!hasRepeatedChars("121"));
}
