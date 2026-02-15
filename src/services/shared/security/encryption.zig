//! Encryption at rest utilities.
//!
//! This module provides:
//! - File encryption/decryption
//! - Field-level encryption
//! - Key derivation (HKDF, PBKDF2)
//! - Envelope encryption
//! - Key wrapping
//! - Secure deletion

const std = @import("std");
const crypto = std.crypto;
const csprng = @import("csprng.zig");

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

/// Encryption algorithm
pub const Algorithm = enum {
    /// AES-256-GCM (recommended)
    aes_256_gcm,
    /// ChaCha20-Poly1305 (fast on platforms without AES-NI)
    chacha20_poly1305,
    /// XChaCha20-Poly1305 (extended nonce)
    xchacha20_poly1305,

    pub fn toString(self: Algorithm) []const u8 {
        return switch (self) {
            .aes_256_gcm => "AES-256-GCM",
            .chacha20_poly1305 => "ChaCha20-Poly1305",
            .xchacha20_poly1305 => "XChaCha20-Poly1305",
        };
    }

    pub fn nonceSize(self: Algorithm) usize {
        return switch (self) {
            .aes_256_gcm => 12,
            .chacha20_poly1305 => 12,
            .xchacha20_poly1305 => 24,
        };
    }

    pub fn tagSize(self: Algorithm) usize {
        return switch (self) {
            .aes_256_gcm => 16,
            .chacha20_poly1305 => 16,
            .xchacha20_poly1305 => 16,
        };
    }
};

/// Key derivation function
pub const Kdf = enum {
    /// HKDF with SHA-256 (fast, for key derivation from existing key material)
    hkdf_sha256,
    /// PBKDF2 with SHA-256 (password-based)
    pbkdf2_sha256,
    /// Argon2id (password-based, memory-hard)
    argon2id,
    /// scrypt (password-based, memory-hard)
    scrypt,
};

/// Encryption configuration
pub const EncryptionConfig = struct {
    /// Algorithm to use
    algorithm: Algorithm = .aes_256_gcm,
    /// KDF for password-based encryption
    kdf: Kdf = .argon2id,
    /// KDF iterations (for PBKDF2)
    kdf_iterations: u32 = 600_000,
    /// Argon2 memory cost (KB)
    argon2_memory: u32 = 65536,
    /// Argon2 time cost
    argon2_time: u32 = 3,
    /// Include algorithm in header
    include_header: bool = true,
    /// Enable compression before encryption
    enable_compression: bool = false,
};

/// Encrypted data header
pub const EncryptedHeader = struct {
    /// Magic bytes
    magic: [4]u8 = "ENC\x01".*,
    /// Algorithm used
    algorithm: Algorithm,
    /// Nonce
    nonce: []const u8,
    /// Salt (for password-based encryption)
    salt: ?[16]u8 = null,
    /// Additional authenticated data length
    aad_length: u32 = 0,

    pub fn serialize(self: EncryptedHeader, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(allocator);

        try buffer.appendSlice(allocator, &self.magic);
        try buffer.append(allocator, @intFromEnum(self.algorithm));
        try buffer.append(allocator, @intCast(self.nonce.len));
        try buffer.appendSlice(allocator, self.nonce);

        if (self.salt) |salt| {
            try buffer.append(allocator, 1); // Has salt
            try buffer.appendSlice(allocator, &salt);
        } else {
            try buffer.append(allocator, 0); // No salt
        }

        const aad_bytes = std.mem.asBytes(&self.aad_length);
        try buffer.appendSlice(allocator, aad_bytes);

        return buffer.toOwnedSlice(allocator);
    }

    pub fn deserialize(data: []const u8) !EncryptedHeader {
        if (data.len < 6) return error.InvalidHeader;

        // Check magic
        if (!std.mem.eql(u8, data[0..4], "ENC\x01")) {
            return error.InvalidMagic;
        }

        const alg = try parseAlgorithm(data[4]);
        const nonce_len = data[5];

        if (data.len < 6 + nonce_len + 1) return error.InvalidHeader;

        const nonce = data[6 .. 6 + nonce_len];
        var offset: usize = 6 + nonce_len;

        const has_salt = data[offset] == 1;
        offset += 1;

        var salt: ?[16]u8 = null;
        if (has_salt) {
            if (data.len < offset + 16) return error.InvalidHeader;
            salt = data[offset..][0..16].*;
            offset += 16;
        }

        if (data.len < offset + 4) return error.InvalidHeader;
        const aad_length = std.mem.readInt(u32, data[offset..][0..4], .little);

        return .{
            .algorithm = alg,
            .nonce = nonce,
            .salt = salt,
            .aad_length = aad_length,
        };
    }

    fn parseAlgorithm(raw: u8) !Algorithm {
        return switch (raw) {
            @intFromEnum(Algorithm.aes_256_gcm) => .aes_256_gcm,
            @intFromEnum(Algorithm.chacha20_poly1305) => .chacha20_poly1305,
            @intFromEnum(Algorithm.xchacha20_poly1305) => .xchacha20_poly1305,
            else => error.InvalidAlgorithm,
        };
    }

    pub fn headerSize(self: EncryptedHeader) usize {
        var size: usize = 4 + 1 + 1 + self.nonce.len + 1 + 4;
        if (self.salt != null) size += 16;
        return size;
    }
};

/// Encryptor
pub const Encryptor = struct {
    allocator: std.mem.Allocator,
    config: EncryptionConfig,

    pub fn init(allocator: std.mem.Allocator, config: EncryptionConfig) Encryptor {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Encrypt data with a key
    pub fn encrypt(self: *Encryptor, plaintext: []const u8, key: [32]u8, aad: ?[]const u8) !EncryptedData {
        const nonce_size = self.config.algorithm.nonceSize();
        _ = self.config.algorithm.tagSize(); // validated implicitly by AEAD

        // Generate random nonce
        var nonce: [24]u8 = undefined;
        csprng.fillRandom(nonce[0..nonce_size]);

        // Allocate ciphertext buffer
        const ciphertext = try self.allocator.alloc(u8, plaintext.len);
        errdefer self.allocator.free(ciphertext);

        var tag: [16]u8 = undefined;

        // Encrypt based on algorithm
        switch (self.config.algorithm) {
            .aes_256_gcm => {
                const aes = crypto.aead.aes_gcm.Aes256Gcm;
                aes.encrypt(ciphertext, &tag, plaintext, aad orelse &.{}, nonce[0..12].*, key);
            },
            .chacha20_poly1305 => {
                const chacha = crypto.aead.chacha_poly.ChaCha20Poly1305;
                chacha.encrypt(ciphertext, &tag, plaintext, aad orelse &.{}, nonce[0..12].*, key);
            },
            .xchacha20_poly1305 => {
                const xchacha = crypto.aead.chacha_poly.XChaCha20Poly1305;
                xchacha.encrypt(ciphertext, &tag, plaintext, aad orelse &.{}, nonce[0..24].*, key);
            },
        }

        // Build header
        const header = EncryptedHeader{
            .algorithm = self.config.algorithm,
            .nonce = try self.allocator.dupe(u8, nonce[0..nonce_size]),
            .aad_length = if (aad) |a| @intCast(a.len) else 0,
        };

        return EncryptedData{
            .header = header,
            .ciphertext = ciphertext,
            .tag = tag,
        };
    }

    /// Encrypt with password (derives key internally)
    pub fn encryptWithPassword(self: *Encryptor, plaintext: []const u8, password: []const u8) !EncryptedData {
        // Generate salt
        var salt: [16]u8 = undefined;
        csprng.fillRandom(&salt);

        // Derive key
        const key = try self.deriveKey(password, &salt);

        // Encrypt
        var result = try self.encrypt(plaintext, key, null);
        result.header.salt = salt;

        return result;
    }

    /// Decrypt data with a key
    pub fn decrypt(self: *Encryptor, encrypted: EncryptedData, key: [32]u8, aad: ?[]const u8) ![]u8 {
        const plaintext = try self.allocator.alloc(u8, encrypted.ciphertext.len);
        errdefer self.allocator.free(plaintext);

        var tag_buf: [16]u8 = undefined;
        const tag_size = encrypted.header.algorithm.tagSize();
        @memcpy(tag_buf[0..tag_size], encrypted.tag[0..tag_size]);

        // Decrypt based on algorithm
        switch (encrypted.header.algorithm) {
            .aes_256_gcm => {
                const aes = crypto.aead.aes_gcm.Aes256Gcm;
                aes.decrypt(
                    plaintext,
                    encrypted.ciphertext,
                    tag_buf[0..16].*,
                    aad orelse &.{},
                    encrypted.header.nonce[0..12].*,
                    key,
                ) catch return error.DecryptionFailed;
            },
            .chacha20_poly1305 => {
                const chacha = crypto.aead.chacha_poly.ChaCha20Poly1305;
                chacha.decrypt(
                    plaintext,
                    encrypted.ciphertext,
                    tag_buf[0..16].*,
                    aad orelse &.{},
                    encrypted.header.nonce[0..12].*,
                    key,
                ) catch return error.DecryptionFailed;
            },
            .xchacha20_poly1305 => {
                const xchacha = crypto.aead.chacha_poly.XChaCha20Poly1305;
                xchacha.decrypt(
                    plaintext,
                    encrypted.ciphertext,
                    tag_buf[0..16].*,
                    aad orelse &.{},
                    encrypted.header.nonce[0..24].*,
                    key,
                ) catch return error.DecryptionFailed;
            },
        }

        return plaintext;
    }

    /// Decrypt with password
    pub fn decryptWithPassword(self: *Encryptor, encrypted: EncryptedData, password: []const u8) ![]u8 {
        const salt = encrypted.header.salt orelse return error.MissingSalt;
        const key = try self.deriveKey(password, &salt);
        return self.decrypt(encrypted, key, null);
    }

    /// Derive key from password
    pub fn deriveKey(self: *Encryptor, password: []const u8, salt: []const u8) ![32]u8 {
        var key: [32]u8 = undefined;

        switch (self.config.kdf) {
            .hkdf_sha256 => {
                const Hkdf = crypto.kdf.hkdf.Hkdf(crypto.auth.hmac.sha2.HmacSha256);
                const prk = Hkdf.extract(salt, password);
                Hkdf.expand(&key, "encryption-key", prk);
            },
            .pbkdf2_sha256 => {
                var salt_arr: [16]u8 = undefined;
                const copy_len = @min(salt.len, 16);
                @memcpy(salt_arr[0..copy_len], salt[0..copy_len]);
                if (copy_len < 16) @memset(salt_arr[copy_len..], 0);
                try crypto.pwhash.pbkdf2(&key, password, &salt_arr, self.config.kdf_iterations, crypto.auth.hmac.sha2.HmacSha256);
            },
            .argon2id => {
                var io_backend = initIoBackend(self.allocator);
                defer io_backend.deinit();
                crypto.pwhash.argon2.kdf(
                    self.allocator,
                    &key,
                    password,
                    salt,
                    .{
                        .t = self.config.argon2_time,
                        .m = self.config.argon2_memory,
                        .p = 4,
                    },
                    .argon2id,
                    io_backend.io(),
                ) catch return error.KeyDerivationFailed;
            },
            .scrypt => {
                crypto.pwhash.scrypt.kdf(
                    self.allocator,
                    &key,
                    password,
                    salt,
                    .{ .ln = 15, .r = 8, .p = 1 },
                ) catch return error.KeyDerivationFailed;
            },
        }

        return key;
    }

    /// Serialize encrypted data to bytes
    pub fn serialize(self: *Encryptor, encrypted: EncryptedData) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(self.allocator);

        // Header
        const header_bytes = try encrypted.header.serialize(self.allocator);
        defer self.allocator.free(header_bytes);
        try buffer.appendSlice(self.allocator, header_bytes);

        // Tag
        const tag_size = encrypted.header.algorithm.tagSize();
        try buffer.appendSlice(self.allocator, encrypted.tag[0..tag_size]);

        // Ciphertext
        try buffer.appendSlice(self.allocator, encrypted.ciphertext);

        return buffer.toOwnedSlice(self.allocator);
    }

    /// Deserialize encrypted data from bytes
    pub fn deserialize(self: *Encryptor, data: []const u8) !EncryptedData {
        const header = try EncryptedHeader.deserialize(data);
        const header_size = header.headerSize();
        const tag_size = header.algorithm.tagSize();

        if (data.len < header_size + tag_size) {
            return error.InvalidData;
        }

        var tag: [16]u8 = undefined;
        @memcpy(tag[0..tag_size], data[header_size .. header_size + tag_size]);

        const ciphertext = try self.allocator.dupe(u8, data[header_size + tag_size ..]);

        return EncryptedData{
            .header = .{
                .algorithm = header.algorithm,
                .nonce = try self.allocator.dupe(u8, header.nonce),
                .salt = header.salt,
                .aad_length = header.aad_length,
            },
            .ciphertext = ciphertext,
            .tag = tag,
        };
    }
};

/// Encrypted data container
pub const EncryptedData = struct {
    header: EncryptedHeader,
    ciphertext: []const u8,
    tag: [16]u8,

    pub fn deinit(self: *EncryptedData, allocator: std.mem.Allocator) void {
        allocator.free(self.header.nonce);
        crypto.secureZero(u8, @constCast(self.ciphertext));
        allocator.free(self.ciphertext);
        crypto.secureZero(u8, &self.tag);
    }
};

/// Key wrapper for envelope encryption
pub const KeyWrapper = struct {
    allocator: std.mem.Allocator,
    master_key: [32]u8,

    pub fn init(allocator: std.mem.Allocator, master_key: [32]u8) KeyWrapper {
        return .{
            .allocator = allocator,
            .master_key = master_key,
        };
    }

    pub fn deinit(self: *KeyWrapper) void {
        crypto.secureZero(u8, &self.master_key);
    }

    /// Wrap (encrypt) a data encryption key.
    /// Returns [60]u8: 12-byte nonce + 32-byte ciphertext + 16-byte tag.
    pub fn wrap(self: *KeyWrapper, dek: [32]u8) ![60]u8 {
        var nonce: [12]u8 = undefined;
        csprng.fillRandom(&nonce);

        var ciphertext: [32]u8 = undefined;
        var tag: [16]u8 = undefined;

        const aes = crypto.aead.aes_gcm.Aes256Gcm;
        aes.encrypt(&ciphertext, &tag, &dek, &.{}, nonce, self.master_key);

        var wrapped: [60]u8 = undefined;
        @memcpy(wrapped[0..12], &nonce);
        @memcpy(wrapped[12..44], &ciphertext);
        @memcpy(wrapped[44..60], &tag);

        return wrapped;
    }

    /// Unwrap (decrypt) a data encryption key.
    /// Accepts [60]u8: 12-byte nonce + 32-byte ciphertext + 16-byte tag.
    pub fn unwrap(self: *KeyWrapper, wrapped: [60]u8) ![32]u8 {
        const nonce = wrapped[0..12].*;
        const ciphertext = wrapped[12..44];
        const tag = wrapped[44..60].*;

        var dek: [32]u8 = undefined;

        const aes = crypto.aead.aes_gcm.Aes256Gcm;
        aes.decrypt(
            &dek,
            ciphertext,
            tag,
            &.{},
            nonce,
            self.master_key,
        ) catch return error.UnwrapFailed;

        return dek;
    }
};

/// Securely delete a file by overwriting with random data
/// Note: In Zig 0.16+, this function requires an allocator for I/O backend initialization
pub fn secureDelete(allocator: std.mem.Allocator, path: []const u8, passes: u8) !void {
    // Initialize I/O backend (Zig 0.16)
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
    errdefer file.close(io);

    const stat = try file.stat(io);
    const size = stat.size;

    var rand_buf: [4096]u8 = undefined;
    var write_buf: [4096]u8 = undefined;

    for (0..passes) |_| {
        var writer = file.writer(io, &write_buf);
        try writer.seekTo(0);

        var remaining = size;
        while (remaining > 0) {
            const to_write = @min(rand_buf.len, remaining);
            csprng.fillRandom(rand_buf[0..to_write]);
            try writer.interface.writeAll(rand_buf[0..to_write]);
            remaining -= to_write;
        }
        try writer.flush();
    }

    // Final pass with zeros
    {
        var writer = file.writer(io, &write_buf);
        try writer.seekTo(0);
        @memset(&rand_buf, 0);

        var remaining = size;
        while (remaining > 0) {
            const to_write = @min(rand_buf.len, remaining);
            try writer.interface.writeAll(rand_buf[0..to_write]);
            remaining -= to_write;
        }
        try writer.flush();
    }

    file.close(io);

    // Delete file
    try std.Io.Dir.cwd().deleteFile(io, path);
}

/// Generate a random encryption key
pub fn generateKey() [32]u8 {
    var key: [32]u8 = undefined;
    csprng.fillRandom(&key);
    return key;
}

/// Encryption errors
pub const EncryptionError = error{
    DecryptionFailed,
    KeyDerivationFailed,
    InvalidHeader,
    InvalidMagic,
    InvalidAlgorithm,
    InvalidData,
    MissingSalt,
    UnwrapFailed,
    OutOfMemory,
};

// Tests

test "encryption round trip" {
    const allocator = std.testing.allocator;
    var encryptor = Encryptor.init(allocator, .{});

    const plaintext = "Hello, World! This is a test message.";
    const key = generateKey();

    var encrypted = try encryptor.encrypt(plaintext, key, null);
    defer encrypted.deinit(allocator);

    const decrypted = try encryptor.decrypt(encrypted, key, null);
    defer allocator.free(decrypted);

    try std.testing.expectEqualStrings(plaintext, decrypted);
}

test "password-based encryption" {
    const allocator = std.testing.allocator;
    var encryptor = Encryptor.init(allocator, .{
        .kdf = .pbkdf2_sha256,
        .kdf_iterations = 1000, // Low for testing
    });

    const plaintext = "Secret message";
    const password = "my-password";

    var encrypted = try encryptor.encryptWithPassword(plaintext, password);
    defer encrypted.deinit(allocator);

    const decrypted = try encryptor.decryptWithPassword(encrypted, password);
    defer allocator.free(decrypted);

    try std.testing.expectEqualStrings(plaintext, decrypted);
}

test "different algorithms" {
    const allocator = std.testing.allocator;
    const key = generateKey();
    const plaintext = "Test data for encryption";

    inline for (&[_]Algorithm{ .aes_256_gcm, .chacha20_poly1305, .xchacha20_poly1305 }) |alg| {
        var encryptor = Encryptor.init(allocator, .{ .algorithm = alg });

        var encrypted = try encryptor.encrypt(plaintext, key, null);
        defer encrypted.deinit(allocator);

        const decrypted = try encryptor.decrypt(encrypted, key, null);
        defer allocator.free(decrypted);

        try std.testing.expectEqualStrings(plaintext, decrypted);
    }
}

test "serialization" {
    const allocator = std.testing.allocator;
    var encryptor = Encryptor.init(allocator, .{});

    const plaintext = "Data to serialize";
    const key = generateKey();

    var encrypted = try encryptor.encrypt(plaintext, key, null);
    defer encrypted.deinit(allocator);

    const serialized = try encryptor.serialize(encrypted);
    defer allocator.free(serialized);

    var deserialized = try encryptor.deserialize(serialized);
    defer deserialized.deinit(allocator);

    const decrypted = try encryptor.decrypt(deserialized, key, null);
    defer allocator.free(decrypted);

    try std.testing.expectEqualStrings(plaintext, decrypted);
}

test "deserialize rejects invalid algorithm id" {
    const allocator = std.testing.allocator;
    var encryptor = Encryptor.init(allocator, .{});

    const plaintext = "Data to serialize";
    const key = generateKey();

    var encrypted = try encryptor.encrypt(plaintext, key, null);
    defer encrypted.deinit(allocator);

    const serialized = try encryptor.serialize(encrypted);
    defer allocator.free(serialized);

    var tampered = try allocator.dupe(u8, serialized);
    defer allocator.free(tampered);
    tampered[4] = 0xff;

    try std.testing.expectError(error.InvalidAlgorithm, encryptor.deserialize(tampered));
}

test "key wrapper" {
    const allocator = std.testing.allocator;
    const master_key = generateKey();

    var wrapper = KeyWrapper.init(allocator, master_key);
    defer wrapper.deinit();

    const dek = generateKey();

    const wrapped = try wrapper.wrap(dek);
    const unwrapped = try wrapper.unwrap(wrapped);

    try std.testing.expectEqualSlices(u8, &dek, &unwrapped);
}
