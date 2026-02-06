//! Secure Channel Abstraction
//!
//! Provides encrypted, authenticated communication channels between nodes.
//! Supports multiple encryption backends and protocols.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const shared_utils = @import("../../../services/shared/utils.zig");

/// Encryption type for the channel.
pub const EncryptionType = enum {
    /// No encryption (use only for local/trusted networks).
    none,
    /// TLS 1.2 (legacy compatibility).
    tls_1_2,
    /// TLS 1.3 (recommended).
    tls_1_3,
    /// Noise Protocol Framework (for P2P).
    noise_xx,
    /// WireGuard-style encryption.
    wireguard,
    /// ChaCha20-Poly1305 with custom key exchange.
    chacha20_poly1305,
    /// AES-256-GCM with custom key exchange.
    aes_256_gcm,

    pub fn keySize(self: EncryptionType) usize {
        return switch (self) {
            .none => 0,
            .tls_1_2, .tls_1_3 => 32, // Depends on cipher
            .noise_xx => 32,
            .wireguard => 32,
            .chacha20_poly1305 => 32,
            .aes_256_gcm => 32,
        };
    }

    pub fn nonceSize(self: EncryptionType) usize {
        return switch (self) {
            .none => 0,
            .tls_1_2, .tls_1_3 => 12,
            .noise_xx => 8,
            .wireguard => 8,
            .chacha20_poly1305 => 12,
            .aes_256_gcm => 12,
        };
    }

    pub fn tagSize(self: EncryptionType) usize {
        return switch (self) {
            .none => 0,
            .tls_1_2, .tls_1_3 => 16,
            .noise_xx => 16,
            .wireguard => 16,
            .chacha20_poly1305 => 16,
            .aes_256_gcm => 16,
        };
    }
};

/// Channel configuration.
pub const ChannelConfig = struct {
    /// Encryption type.
    encryption: EncryptionType = .tls_1_3,

    /// Pre-shared key (optional, for additional security).
    psk: ?[32]u8 = null,

    /// Local certificate (PEM format).
    local_cert: ?[]const u8 = null,

    /// Local private key (PEM format).
    local_key: ?[]const u8 = null,

    /// CA certificate for verification.
    ca_cert: ?[]const u8 = null,

    /// Verify peer certificate.
    verify_peer: bool = true,

    /// Expected peer hostname (for verification).
    peer_hostname: ?[]const u8 = null,

    /// Maximum message size.
    max_message_size: usize = 16 * 1024 * 1024, // 16 MB

    /// Enable message authentication.
    authenticate_messages: bool = true,

    /// Enable replay protection.
    replay_protection: bool = true,

    /// Handshake timeout (milliseconds).
    handshake_timeout_ms: u64 = 10000,

    /// Session resumption enabled.
    session_resumption: bool = true,

    /// Key rotation interval (seconds, 0 = no rotation).
    key_rotation_interval_s: u64 = 3600, // 1 hour
};

/// Channel state.
pub const ChannelState = enum {
    /// Not initialized.
    uninitialized,
    /// Performing handshake.
    handshaking,
    /// Handshake complete, channel ready.
    established,
    /// Channel is being rekeyed.
    rekeying,
    /// Channel has error.
    error_state,
    /// Channel closed.
    closed,

    pub fn isReady(self: ChannelState) bool {
        return self == .established or self == .rekeying;
    }
};

/// Channel statistics.
pub const ChannelStats = struct {
    /// Bytes encrypted.
    bytes_encrypted: u64 = 0,
    /// Bytes decrypted.
    bytes_decrypted: u64 = 0,
    /// Encryption operations.
    encrypt_ops: u64 = 0,
    /// Decryption operations.
    decrypt_ops: u64 = 0,
    /// Authentication failures.
    auth_failures: u64 = 0,
    /// Replay attacks detected.
    replay_attacks: u64 = 0,
    /// Key rotations performed.
    key_rotations: u64 = 0,
    /// Handshakes completed.
    handshakes: u64 = 0,
    /// Session resumptions.
    resumptions: u64 = 0,
    /// Channel established timestamp.
    established_at_ms: i64 = 0,
    /// Last activity timestamp.
    last_activity_ms: i64 = 0,
};

/// Secure channel for encrypted communication.
pub const SecureChannel = struct {
    allocator: std.mem.Allocator,

    /// Configuration.
    config: ChannelConfig,

    /// Current state.
    state: ChannelState,

    /// Statistics.
    stats: ChannelStats,

    /// Encryption key (sending).
    send_key: [32]u8,

    /// Encryption key (receiving).
    recv_key: [32]u8,

    /// Send nonce counter.
    send_nonce: u64,

    /// Receive nonce counter (for replay protection).
    recv_nonce: u64,

    /// Nonce bitmap for replay protection.
    nonce_bitmap: NonceBitmap,

    /// Session ID (for resumption).
    session_id: [32]u8,

    /// Peer's public key.
    peer_public_key: ?[32]u8,

    /// Our key pair.
    local_keypair: ?KeyPair,

    /// Remote address.
    remote_address: ?[]const u8,

    /// Underlying transport stream.
    stream: ?*anyopaque,

    /// Lock for thread safety.
    mutex: sync.Mutex,

    pub const KeyPair = struct {
        public_key: [32]u8,
        secret_key: [32]u8,
    };

    pub const NonceBitmap = struct {
        base: u64 = 0,
        bitmap: u128 = 0,

        pub fn check(self: *NonceBitmap, nonce: u64) bool {
            if (nonce < self.base) return false; // Too old
            if (nonce >= self.base + 128) {
                // Advance window
                const advance = nonce - self.base - 127;
                self.base += advance;
                self.bitmap >>= @intCast(advance);
            }

            const bit_index: u7 = @intCast(nonce - self.base);
            const mask = @as(u128, 1) << bit_index;
            if (self.bitmap & mask != 0) return false; // Replay
            self.bitmap |= mask;
            return true;
        }
    };

    /// Initialize a secure channel.
    pub fn init(allocator: std.mem.Allocator, config: ChannelConfig) !*SecureChannel {
        const channel = try allocator.create(SecureChannel);
        errdefer allocator.destroy(channel);

        channel.* = .{
            .allocator = allocator,
            .config = config,
            .state = .uninitialized,
            .stats = .{},
            .send_key = std.mem.zeroes([32]u8),
            .recv_key = std.mem.zeroes([32]u8),
            .send_nonce = 0,
            .recv_nonce = 0,
            .nonce_bitmap = .{},
            .session_id = std.mem.zeroes([32]u8),
            .peer_public_key = null,
            .local_keypair = null,
            .remote_address = null,
            .stream = null,
            .mutex = .{},
        };

        // Generate ephemeral keypair if needed
        if (config.encryption != .none) {
            channel.local_keypair = try generateKeyPair();
        }

        return channel;
    }

    /// Deinitialize channel.
    pub fn deinit(self: *SecureChannel) void {
        self.close();

        // Securely wipe keys
        std.crypto.secureZero(u8, &self.send_key);
        std.crypto.secureZero(u8, &self.recv_key);

        if (self.local_keypair) |*kp| {
            std.crypto.secureZero(u8, &kp.secret_key);
        }

        if (self.remote_address) |addr| {
            self.allocator.free(addr);
        }

        self.allocator.destroy(self);
    }

    /// Connect to a remote address.
    pub fn connect(self: *SecureChannel, address: []const u8) ChannelError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state != .uninitialized and self.state != .closed) {
            return error.InvalidState;
        }

        self.remote_address = self.allocator.dupe(u8, address) catch return error.OutOfMemory;
        self.state = .handshaking;

        // Perform handshake based on encryption type
        if (self.config.encryption != .none) {
            self.performHandshake() catch |err| {
                self.state = .error_state;
                return err;
            };
        }

        self.state = .established;
        self.stats.established_at_ms = shared_utils.unixMs();
        self.stats.handshakes += 1;
    }

    /// Accept an incoming connection.
    pub fn accept(self: *SecureChannel, stream: *anyopaque) ChannelError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state != .uninitialized) {
            return error.InvalidState;
        }

        self.stream = stream;
        self.state = .handshaking;

        // Perform server-side handshake
        if (self.config.encryption != .none) {
            self.performServerHandshake() catch |err| {
                self.state = .error_state;
                return err;
            };
        }

        self.state = .established;
        self.stats.established_at_ms = shared_utils.unixMs();
        self.stats.handshakes += 1;
    }

    /// Send data through the channel.
    pub fn send(self: *SecureChannel, plaintext: []const u8) ChannelError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.state.isReady()) {
            return error.NotEstablished;
        }

        if (plaintext.len > self.config.max_message_size) {
            return error.MessageTooLarge;
        }

        // Encrypt if needed
        if (self.config.encryption != .none) {
            _ = try self.encrypt(plaintext);
        }

        self.send_nonce += 1;
        self.stats.encrypt_ops += 1;
        self.stats.bytes_encrypted += plaintext.len;
        self.stats.last_activity_ms = shared_utils.unixMs();
    }

    /// Receive data from the channel.
    pub fn receive(self: *SecureChannel, buffer: []u8) ChannelError!usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.state.isReady()) {
            return error.NotEstablished;
        }

        // For now, return 0 (would read from underlying transport)
        _ = buffer;

        self.stats.decrypt_ops += 1;
        self.stats.last_activity_ms = shared_utils.unixMs();

        return 0;
    }

    /// Close the channel.
    pub fn close(self: *SecureChannel) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state == .closed) return;

        // Send close notification if connected
        if (self.state == .established) {
            // Would send close message
        }

        self.state = .closed;
    }

    /// Rotate session keys.
    pub fn rotateKeys(self: *SecureChannel) ChannelError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state != .established) {
            return error.NotEstablished;
        }

        self.state = .rekeying;

        // Derive new keys from existing ones
        try self.deriveNewKeys();

        self.state = .established;
        self.stats.key_rotations += 1;
    }

    /// Get channel statistics.
    pub fn getStats(self: *SecureChannel) ChannelStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Check if channel needs rekeying.
    pub fn needsRekey(self: *SecureChannel) bool {
        if (self.config.key_rotation_interval_s == 0) return false;
        const now = shared_utils.unixMs();
        const elapsed = @as(u64, @intCast(now - self.stats.established_at_ms)) / 1000;
        return elapsed >= self.config.key_rotation_interval_s;
    }

    // Private methods

    fn performHandshake(self: *SecureChannel) ChannelError!void {
        switch (self.config.encryption) {
            .noise_xx => try self.noiseXXHandshake(),
            .wireguard => try self.wireguardHandshake(),
            .tls_1_2, .tls_1_3 => try self.tlsHandshake(),
            .chacha20_poly1305, .aes_256_gcm => try self.customHandshake(),
            .none => {},
        }
    }

    fn performServerHandshake(self: *SecureChannel) ChannelError!void {
        // Similar to client but as responder
        try self.performHandshake();
    }

    /// Noise XX pattern handshake.
    ///
    /// **WARNING: PLACEHOLDER IMPLEMENTATION - NOT FOR PRODUCTION USE**
    ///
    /// This is a simplified placeholder that derives keys from local public key only.
    /// A real Noise XX implementation requires:
    /// 1. Actual key exchange with the remote peer
    /// 2. Ephemeral key generation and exchange
    /// 3. Proper Noise pattern state machine
    ///
    /// Do NOT use this for securing sensitive communications until a proper
    /// Noise Protocol implementation is added.
    fn noiseXXHandshake(self: *SecureChannel) ChannelError!void {
        const kp = self.local_keypair orelse return error.NoKeyPair;

        // PLACEHOLDER: Derives keys from local public key only - NO ACTUAL KEY EXCHANGE
        var hash = std.crypto.hash.Blake2b256.init(.{});
        hash.update(&kp.public_key);
        hash.update("noise-xx-handshake");
        hash.final(&self.send_key);

        @memcpy(&self.recv_key, &self.send_key);
    }

    /// WireGuard-style handshake.
    ///
    /// **WARNING: PLACEHOLDER IMPLEMENTATION - NOT FOR PRODUCTION USE**
    ///
    /// This is a simplified placeholder that derives keys locally without peer exchange.
    /// A real WireGuard implementation requires:
    /// 1. Noise_IKpsk2 handshake pattern
    /// 2. Peer public key exchange
    /// 3. Cookie/MAC computation for DoS protection
    ///
    /// Do NOT use this for securing sensitive communications until a proper
    /// WireGuard implementation is added.
    fn wireguardHandshake(self: *SecureChannel) ChannelError!void {
        const kp = self.local_keypair orelse return error.NoKeyPair;

        // PLACEHOLDER: Derives keys locally - NO ACTUAL KEY EXCHANGE
        var hash = std.crypto.hash.Blake2b256.init(.{});
        hash.update(&kp.public_key);
        hash.update("wireguard-handshake");

        if (self.config.psk) |psk| {
            hash.update(&psk);
        }

        hash.final(&self.send_key);
        @memcpy(&self.recv_key, &self.send_key);
    }

    /// TLS handshake.
    ///
    /// **WARNING: PLACEHOLDER IMPLEMENTATION - NOT FOR PRODUCTION USE**
    ///
    /// This is a simplified placeholder that does NOT perform actual TLS negotiation.
    /// A real TLS implementation requires:
    /// 1. Full TLS handshake state machine
    /// 2. Certificate validation
    /// 3. Cipher suite negotiation
    /// 4. Use of a proper TLS library (e.g., OpenSSL, BoringSSL, s2n)
    ///
    /// Do NOT use this for securing sensitive communications. Use a real TLS
    /// library for production deployments.
    fn tlsHandshake(self: *SecureChannel) ChannelError!void {
        const kp = self.local_keypair orelse return error.NoKeyPair;

        // PLACEHOLDER: Derives keys locally - NO ACTUAL TLS NEGOTIATION
        var hash = std.crypto.hash.Blake2b256.init(.{});
        hash.update(&kp.public_key);
        hash.update("tls-handshake");
        hash.final(&self.send_key);

        @memcpy(&self.recv_key, &self.send_key);
    }

    /// Custom X25519-based handshake.
    ///
    /// **WARNING: PLACEHOLDER IMPLEMENTATION - NOT FOR PRODUCTION USE**
    ///
    /// This is a simplified placeholder that derives keys from local public key only.
    /// A real implementation requires:
    /// 1. X25519 Diffie-Hellman key exchange with peer
    /// 2. Nonce/session ID exchange
    /// 3. Proper key derivation function (HKDF)
    ///
    /// Do NOT use this for securing sensitive communications until proper
    /// X25519 key exchange is implemented.
    fn customHandshake(self: *SecureChannel) ChannelError!void {
        const kp = self.local_keypair orelse return error.NoKeyPair;

        // PLACEHOLDER: Derives keys from local public key only - NO PEER EXCHANGE
        var hash = std.crypto.hash.Blake2b256.init(.{});
        hash.update(&kp.public_key);
        hash.update("custom-handshake");
        hash.final(&self.send_key);

        @memcpy(&self.recv_key, &self.send_key);
    }

    fn deriveNewKeys(self: *SecureChannel) ChannelError!void {
        var hash = std.crypto.hash.Blake2b256.init(.{});
        hash.update(&self.send_key);
        hash.update(&self.recv_key);
        hash.update("key-rotation");

        var new_key: [32]u8 = undefined;
        hash.final(&new_key);

        // Securely update keys
        std.crypto.secureZero(u8, &self.send_key);
        std.crypto.secureZero(u8, &self.recv_key);
        @memcpy(&self.send_key, &new_key);
        @memcpy(&self.recv_key, &new_key);
        std.crypto.secureZero(u8, &new_key);
    }

    fn encrypt(self: *SecureChannel, plaintext: []const u8) ChannelError![]u8 {
        _ = self;
        _ = plaintext;
        // Would perform actual encryption
        // For now, return empty (placeholder)
        return &[_]u8{};
    }

    fn decrypt(self: *SecureChannel, ciphertext: []const u8) ChannelError![]u8 {
        _ = self;
        _ = ciphertext;
        // Would perform actual decryption
        return &[_]u8{};
    }
};

fn generateKeyPair() !SecureChannel.KeyPair {
    var kp: SecureChannel.KeyPair = undefined;

    // Generate random secret key
    std.crypto.random.bytes(&kp.secret_key);

    // Derive public key (using Blake2b as placeholder for X25519)
    var hash = std.crypto.hash.Blake2b256.init(.{});
    hash.update(&kp.secret_key);
    hash.final(&kp.public_key);

    return kp;
}

/// Channel error types.
pub const ChannelError = error{
    InvalidState,
    NotEstablished,
    HandshakeFailed,
    AuthenticationFailed,
    DecryptionFailed,
    MessageTooLarge,
    ReplayDetected,
    NoKeyPair,
    PeerVerificationFailed,
    OutOfMemory,
    Timeout,
    Closed,
};

/// Encrypted message format.
pub const EncryptedMessage = struct {
    /// Nonce (unique per message).
    nonce: [12]u8,
    /// Authentication tag.
    tag: [16]u8,
    /// Encrypted payload.
    ciphertext: []const u8,

    /// Total overhead (nonce + tag).
    pub const OVERHEAD = 12 + 16;

    /// Serialize to bytes.
    pub fn toBytes(self: EncryptedMessage, allocator: std.mem.Allocator) ![]u8 {
        const total_len = OVERHEAD + self.ciphertext.len;
        const buffer = try allocator.alloc(u8, total_len);
        @memcpy(buffer[0..12], &self.nonce);
        @memcpy(buffer[12..28], &self.tag);
        @memcpy(buffer[28..], self.ciphertext);
        return buffer;
    }

    /// Deserialize from bytes.
    pub fn fromBytes(data: []const u8) ?EncryptedMessage {
        if (data.len < OVERHEAD) return null;
        return .{
            .nonce = data[0..12].*,
            .tag = data[12..28].*,
            .ciphertext = data[28..],
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "EncryptionType sizes" {
    try std.testing.expectEqual(@as(usize, 32), EncryptionType.tls_1_3.keySize());
    try std.testing.expectEqual(@as(usize, 12), EncryptionType.chacha20_poly1305.nonceSize());
    try std.testing.expectEqual(@as(usize, 16), EncryptionType.aes_256_gcm.tagSize());
    try std.testing.expectEqual(@as(usize, 0), EncryptionType.none.keySize());
}

test "ChannelState checks" {
    try std.testing.expect(ChannelState.established.isReady());
    try std.testing.expect(ChannelState.rekeying.isReady());
    try std.testing.expect(!ChannelState.handshaking.isReady());
    try std.testing.expect(!ChannelState.closed.isReady());
}

test "NonceBitmap replay protection" {
    var bitmap = SecureChannel.NonceBitmap{};

    // First use should succeed
    try std.testing.expect(bitmap.check(0));
    // Replay should fail
    try std.testing.expect(!bitmap.check(0));

    // New nonces should succeed
    try std.testing.expect(bitmap.check(1));
    try std.testing.expect(bitmap.check(5));
    try std.testing.expect(bitmap.check(100));

    // Replays should fail
    try std.testing.expect(!bitmap.check(1));
    try std.testing.expect(!bitmap.check(5));
}

test "SecureChannel initialization" {
    const allocator = std.testing.allocator;

    const channel = try SecureChannel.init(allocator, .{});
    defer channel.deinit();

    try std.testing.expectEqual(ChannelState.uninitialized, channel.state);
    try std.testing.expect(channel.local_keypair != null);
}

test "EncryptedMessage serialization" {
    const allocator = std.testing.allocator;

    const msg = EncryptedMessage{
        .nonce = [_]u8{1} ** 12,
        .tag = [_]u8{2} ** 16,
        .ciphertext = "hello",
    };

    const bytes = try msg.toBytes(allocator);
    defer allocator.free(bytes);

    const restored = EncryptedMessage.fromBytes(bytes) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualSlices(u8, &msg.nonce, &restored.nonce);
    try std.testing.expectEqualSlices(u8, &msg.tag, &restored.tag);
    try std.testing.expectEqualStrings("hello", restored.ciphertext);
}

test "SecureChannel state transition: uninitialized to handshaking to established" {
    const allocator = std.testing.allocator;

    const channel = try SecureChannel.init(allocator, .{ .encryption = .noise_xx });
    defer channel.deinit();

    // Initial state should be uninitialized
    try std.testing.expectEqual(ChannelState.uninitialized, channel.state);

    // Connect should transition through handshaking to established
    try channel.connect("test-address");

    try std.testing.expectEqual(ChannelState.established, channel.state);
    try std.testing.expect(channel.stats.handshakes == 1);
    try std.testing.expect(channel.stats.established_at_ms > 0);
}

test "SecureChannel state transition: established to rekeying to established" {
    const allocator = std.testing.allocator;

    const channel = try SecureChannel.init(allocator, .{ .encryption = .chacha20_poly1305 });
    defer channel.deinit();

    // Connect first
    try channel.connect("test-address");
    try std.testing.expectEqual(ChannelState.established, channel.state);

    // Store original keys for comparison
    var original_send_key: [32]u8 = undefined;
    @memcpy(&original_send_key, &channel.send_key);

    // Rotate keys should transition through rekeying and back to established
    try channel.rotateKeys();

    try std.testing.expectEqual(ChannelState.established, channel.state);
    try std.testing.expect(channel.stats.key_rotations == 1);

    // Keys should have changed after rotation
    try std.testing.expect(!std.mem.eql(u8, &original_send_key, &channel.send_key));
}

test "SecureChannel state transition: established to closed" {
    const allocator = std.testing.allocator;

    const channel = try SecureChannel.init(allocator, .{ .encryption = .tls_1_3 });
    defer channel.deinit();

    // Connect first
    try channel.connect("test-address");
    try std.testing.expectEqual(ChannelState.established, channel.state);

    // Close should transition to closed
    channel.close();
    try std.testing.expectEqual(ChannelState.closed, channel.state);

    // Cannot send after close
    const result = channel.send("test message");
    try std.testing.expectError(error.NotEstablished, result);

    // Cannot rotate keys after close
    const rekey_result = channel.rotateKeys();
    try std.testing.expectError(error.NotEstablished, rekey_result);
}

test "SecureChannel connect from invalid state returns error" {
    const allocator = std.testing.allocator;

    const channel = try SecureChannel.init(allocator, .{ .encryption = .wireguard });
    defer channel.deinit();

    // First connect should succeed
    try channel.connect("first-address");
    try std.testing.expectEqual(ChannelState.established, channel.state);

    // Second connect from established state should fail
    const result = channel.connect("second-address");
    try std.testing.expectError(error.InvalidState, result);
}
