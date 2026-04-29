//! Password hashing implementation with multiple algorithm support.

const std = @import("std");
const crypto = std.crypto;
const csprng = @import("../csprng.zig");
const types = @import("types.zig");
const parse = @import("parse.zig");
const strength_mod = @import("strength.zig");

/// Password hasher
pub const PasswordHasher = struct {
    allocator: std.mem.Allocator,
    config: types.PasswordConfig,

    pub fn init(allocator: std.mem.Allocator, config: types.PasswordConfig) PasswordHasher {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Hash a password with the configured algorithm
    pub fn hash(self: *PasswordHasher, password: []const u8) !types.HashedPassword {
        if (password.len < self.config.min_password_length) {
            return error.PasswordTooShort;
        }
        if (password.len > self.config.max_password_length) {
            return error.PasswordTooLong;
        }

        if (self.config.require_strength_check) {
            const analysis = strength_mod.analyzeStrength(password);
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
        const algorithm = parse.detectAlgorithm(encoded) orelse return error.InvalidHashFormat;

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
        const algorithm = parse.detectAlgorithm(encoded) orelse return true;

        if (algorithm != self.config.algorithm) return true;

        return switch (algorithm) {
            .argon2id => self.argon2NeedsRehash(encoded),
            .pbkdf2_sha256, .pbkdf2_sha512 => self.pbkdf2NeedsRehash(encoded),
            .scrypt => self.scryptNeedsRehash(encoded),
            .blake3_kdf => false,
        };
    }

    fn hashArgon2(self: *PasswordHasher, password: []const u8) !types.HashedPassword {
        const params = self.config.argon2;

        var salt: [32]u8 = undefined;
        const salt_slice = salt[0..params.salt_length];
        try csprng.fillRandom(salt_slice);

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

    fn hashPbkdf2Sha256(self: *PasswordHasher, password: []const u8) !types.HashedPassword {
        const params = self.config.pbkdf2;

        var salt: [32]u8 = undefined;
        const salt_slice = salt[0..params.salt_length];
        try csprng.fillRandom(salt_slice);

        var derived: [64]u8 = undefined;
        const hash_slice = derived[0..params.hash_length];

        crypto.pwhash.pbkdf2(hash_slice, password, salt_slice, params.iterations, .sha256);

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

    fn hashPbkdf2Sha512(self: *PasswordHasher, password: []const u8) !types.HashedPassword {
        const params = self.config.pbkdf2;

        var salt: [32]u8 = undefined;
        const salt_slice = salt[0..params.salt_length];
        try csprng.fillRandom(salt_slice);

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

    fn hashScrypt(self: *PasswordHasher, password: []const u8) !types.HashedPassword {
        const params = self.config.scrypt;

        var salt: [32]u8 = undefined;
        const salt_slice = salt[0..params.salt_length];
        try csprng.fillRandom(salt_slice);

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

    fn hashBlake3(self: *PasswordHasher, password: []const u8) !types.HashedPassword {
        var salt: [16]u8 = undefined;
        try csprng.fillRandom(&salt);

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

    fn verifyArgon2(self: *PasswordHasher, password: []const u8, encoded: []const u8) !bool {
        _ = self;
        const parsed = parse.parseArgon2Encoded(encoded) orelse return error.InvalidHashFormat;

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
        const parsed = parse.parsePbkdf2Encoded(encoded) orelse return error.InvalidHashFormat;

        var derived: [64]u8 = undefined;
        const hash_slice = derived[0..parsed.hash.len];

        const prf = if (variant == .sha256) .sha256 else .sha512;
        crypto.pwhash.pbkdf2(hash_slice, password, parsed.salt, parsed.iterations, prf);

        return crypto.utils.timingSafeEql(u8, hash_slice, parsed.hash);
    }

    fn verifyScrypt(self: *PasswordHasher, password: []const u8, encoded: []const u8) !bool {
        _ = self;
        const parsed = parse.parseScryptEncoded(encoded) orelse return error.InvalidHashFormat;

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
        const parsed = parse.parseBlake3Encoded(encoded) orelse return error.InvalidHashFormat;

        var hasher = crypto.hash.Blake3.init(.{});
        hasher.update(parsed.salt);
        hasher.update(password);

        var derived: [32]u8 = undefined;
        hasher.final(&derived);

        return crypto.utils.timingSafeEql(u8, &derived, parsed.hash);
    }

    fn argon2NeedsRehash(self: *PasswordHasher, encoded: []const u8) bool {
        const parsed = parse.parseArgon2Encoded(encoded) orelse return true;
        const cfg = self.config.argon2;

        return parsed.memory_cost < cfg.memory_cost or
            parsed.time_cost < cfg.time_cost or
            parsed.parallelism < cfg.parallelism;
    }

    fn pbkdf2NeedsRehash(self: *PasswordHasher, encoded: []const u8) bool {
        const parsed = parse.parsePbkdf2Encoded(encoded) orelse return true;
        return parsed.iterations < self.config.pbkdf2.iterations;
    }

    fn scryptNeedsRehash(self: *PasswordHasher, encoded: []const u8) bool {
        const parsed = parse.parseScryptEncoded(encoded) orelse return true;
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

/// Generate a secure random password
pub fn generatePassword(allocator: std.mem.Allocator, length: usize, options: types.GenerateOptions) ![]u8 {
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

    for (result) |*c| {
        const idx = try csprng.uintLessThan(usize, charset.items.len);
        c.* = charset.items[idx];
    }

    var pos: usize = 0;
    if (options.include_lowercase and options.require_all_classes) {
        result[pos] = "abcdefghijklmnopqrstuvwxyz"[try csprng.uintLessThan(usize, 26)];
        pos += 1;
    }
    if (options.include_uppercase and options.require_all_classes and pos < length) {
        result[pos] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[try csprng.uintLessThan(usize, 26)];
        pos += 1;
    }
    if (options.include_digits and options.require_all_classes and pos < length) {
        result[pos] = "0123456789"[try csprng.uintLessThan(usize, 10)];
        pos += 1;
    }
    if (options.include_special and options.require_all_classes and pos < length) {
        const special = "!@#$%^&*()_+-=[]{}|;:,.<>?";
        result[pos] = special[try csprng.uintLessThan(usize, special.len)];
        pos += 1;
    }

    try csprng.shuffle(u8, result);

    return result;
}

test "password generation" {
    const allocator = std.testing.allocator;

    const pw = try generatePassword(allocator, 16, .{});
    defer allocator.free(pw);

    try std.testing.expectEqual(@as(usize, 16), pw.len);

    var has_lower = false;
    var has_upper = false;
    var has_digit = false;
    var has_special = false;

    for (pw) |c| {
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

test {
    std.testing.refAllDecls(@This());
}
