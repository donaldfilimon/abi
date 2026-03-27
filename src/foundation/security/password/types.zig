//! Type definitions for the password module.

const std = @import("std");

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
        const crypto = std.crypto;
        crypto.secureZero(u8, @constCast(self.hash));
        crypto.secureZero(u8, @constCast(self.salt));
        allocator.free(self.hash);
        allocator.free(self.salt);
        allocator.free(self.params);
        allocator.free(self.encoded);
    }
};

pub const GenerateOptions = struct {
    include_lowercase: bool = true,
    include_uppercase: bool = true,
    include_digits: bool = true,
    include_special: bool = true,
    require_all_classes: bool = true,
    exclude_ambiguous: bool = false,
};

/// Parsed Argon2 encoded hash
pub const ParsedArgon2 = struct {
    salt: []const u8,
    hash: []const u8,
    memory_cost: u32,
    time_cost: u32,
    parallelism: u8,
};

/// Parsed PBKDF2 encoded hash
pub const ParsedPbkdf2 = struct {
    salt: []const u8,
    hash: []const u8,
    iterations: u32,
};

/// Parsed scrypt encoded hash
pub const ParsedScrypt = struct {
    salt: []const u8,
    hash: []const u8,
    log_n: u6,
    r: u30,
    p: u30,
};

/// Parsed Blake3 encoded hash
pub const ParsedBlake3 = struct {
    salt: []const u8,
    hash: []const u8,
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

test {
    std.testing.refAllDecls(@This());
}
