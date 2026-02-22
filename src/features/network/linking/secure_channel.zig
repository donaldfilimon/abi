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

    /// Noise XX pattern handshake (simplified).
    ///
    /// Implements a simplified Noise XX-like pattern:
    ///   -> e          (initiator sends ephemeral public key)
    ///   <- e, ee, s   (responder ephemeral + DH + static)
    ///   -> s, se      (initiator static + DH)
    ///
    /// Uses X25519 for Diffie-Hellman and HKDF-SHA256 for key derivation.
    /// Without a real peer, both sides are simulated locally.
    fn noiseXXHandshake(self: *SecureChannel) ChannelError!void {
        const X25519 = std.crypto.dh.X25519;
        const Hkdf = std.crypto.hkdf.HkdfSha256;

        // --- Message 1: -> e (initiator sends ephemeral key) ---
        const initiator_ephemeral = X25519.KeyPair.generate(null) catch
            return error.HandshakeFailed;

        // --- Message 2: <- e, ee (responder ephemeral + DH) ---
        const responder_ephemeral = X25519.KeyPair.generate(null) catch
            return error.HandshakeFailed;

        // ee: DH(initiator_ephemeral, responder_ephemeral)
        const shared_ee = X25519.scalarmult(
            initiator_ephemeral.secret_key,
            responder_ephemeral.public_key,
        ) catch return error.HandshakeFailed;

        // --- Message 3: -> s, se (initiator static + DH) ---
        const local_kp = self.local_keypair orelse return error.NoKeyPair;

        // se: DH(local_static, responder_ephemeral)
        const shared_se = X25519.scalarmult(
            local_kp.secret_key,
            responder_ephemeral.public_key,
        ) catch return error.HandshakeFailed;

        // Derive transport keys by mixing both DH results.
        // chaining_key = HKDF-Extract(ee_secret, se_secret)
        const prk = Hkdf.extract(&shared_ee, &shared_se);
        Hkdf.expand(&self.send_key, "noise-xx-send", prk);
        Hkdf.expand(&self.recv_key, "noise-xx-recv", prk);

        // Generate session ID from handshake transcript
        var session_hash = std.crypto.hash.Blake2b256.init(.{});
        session_hash.update(&initiator_ephemeral.public_key);
        session_hash.update(&responder_ephemeral.public_key);
        session_hash.update(&local_kp.public_key);
        session_hash.final(&self.session_id);

        // Store peer public key (responder ephemeral acts as peer identity)
        self.peer_public_key = responder_ephemeral.public_key;

        // Wipe intermediate secrets
        var ee_copy = shared_ee;
        var se_copy = shared_se;
        std.crypto.secureZero(u8, &ee_copy);
        std.crypto.secureZero(u8, &se_copy);
    }

    /// WireGuard-style handshake (simplified).
    ///
    /// Implements a simplified version of WireGuard's Noise_IKpsk2 pattern:
    ///   1. Generate initiator ephemeral keypair
    ///   2. Compute DH(ephemeral, responder_static) for initial key
    ///   3. Mix in optional pre-shared key (PSK) for post-quantum resistance
    ///   4. Derive transport keys via HKDF
    ///
    /// Without a real peer, the responder side is simulated locally.
    fn wireguardHandshake(self: *SecureChannel) ChannelError!void {
        const X25519 = std.crypto.dh.X25519;
        const Hkdf = std.crypto.hkdf.HkdfSha256;

        // Initiator ephemeral keypair
        const initiator_ephemeral = X25519.KeyPair.generate(null) catch
            return error.HandshakeFailed;

        // Simulate responder static keypair
        const responder_static = X25519.KeyPair.generate(null) catch
            return error.HandshakeFailed;

        // --- Handshake Initiation ---
        // DH(initiator_ephemeral, responder_static)
        const shared_ei = X25519.scalarmult(
            initiator_ephemeral.secret_key,
            responder_static.public_key,
        ) catch return error.HandshakeFailed;

        // Responder ephemeral for forward secrecy
        const responder_ephemeral = X25519.KeyPair.generate(null) catch
            return error.HandshakeFailed;

        // --- Handshake Response ---
        // DH(initiator_ephemeral, responder_ephemeral)
        const shared_ee = X25519.scalarmult(
            initiator_ephemeral.secret_key,
            responder_ephemeral.public_key,
        ) catch return error.HandshakeFailed;

        // Construct chaining key: mix both DH secrets
        var chaining_key: [32]u8 = undefined;
        var ck_hash = std.crypto.hash.Blake2b256.init(.{});
        ck_hash.update(&shared_ei);
        ck_hash.update(&shared_ee);
        ck_hash.final(&chaining_key);

        // Mix in pre-shared key if configured (PSK mode)
        if (self.config.psk) |psk| {
            var psk_hash = std.crypto.hash.Blake2b256.init(.{});
            psk_hash.update(&chaining_key);
            psk_hash.update(&psk);
            psk_hash.final(&chaining_key);
        }

        // Derive transport keys via HKDF
        const prk = Hkdf.extract(&chaining_key, "wireguard-transport");
        Hkdf.expand(&self.send_key, "wg-send", prk);
        Hkdf.expand(&self.recv_key, "wg-recv", prk);

        // Generate session ID (sender index in WireGuard terms)
        var nonce_buf: [32]u8 = undefined;
        std.c.arc4random_buf(&nonce_buf, nonce_buf.len);
        @memcpy(&self.session_id, &nonce_buf);

        // Store peer identity
        self.peer_public_key = responder_static.public_key;
        self.local_keypair = .{
            .public_key = initiator_ephemeral.public_key,
            .secret_key = initiator_ephemeral.secret_key,
        };

        // Wipe intermediate secrets
        var ei_copy = shared_ei;
        var ee_copy = shared_ee;
        std.crypto.secureZero(u8, &ei_copy);
        std.crypto.secureZero(u8, &ee_copy);
        std.crypto.secureZero(u8, &chaining_key);
    }

    /// TLS handshake (simplified).
    ///
    /// Implements a simplified TLS 1.3-like handshake:
    ///   ClientHello: ephemeral X25519 key share + random nonce
    ///   ServerHello: ephemeral X25519 key share + random nonce
    ///   Key derivation via HKDF-SHA256 (Early -> Handshake -> Application)
    ///
    /// Without a real TLS library, certificate validation is skipped.
    /// The key exchange and key schedule follow the TLS 1.3 structure.
    fn tlsHandshake(self: *SecureChannel) ChannelError!void {
        const X25519 = std.crypto.dh.X25519;
        const Hkdf = std.crypto.hkdf.HkdfSha256;

        // --- ClientHello ---
        var client_random: [32]u8 = undefined;
        std.c.arc4random_buf(&client_random, client_random.len);

        const client_keypair = X25519.KeyPair.generate(null) catch
            return error.HandshakeFailed;

        // --- ServerHello (simulated) ---
        var server_random: [32]u8 = undefined;
        std.c.arc4random_buf(&server_random, server_random.len);

        const server_keypair = X25519.KeyPair.generate(null) catch
            return error.HandshakeFailed;

        // --- Key Exchange ---
        // Shared secret = DH(client_ephemeral, server_ephemeral)
        const shared_secret = X25519.scalarmult(
            client_keypair.secret_key,
            server_keypair.public_key,
        ) catch return error.HandshakeFailed;

        // --- TLS 1.3 Key Schedule (simplified) ---
        // Early Secret = HKDF-Extract(0, 0)  (no PSK)
        var zero_salt: [32]u8 = std.mem.zeroes([32]u8);
        const early_secret = Hkdf.extract(&zero_salt, &zero_salt);

        // Handshake Secret = HKDF-Extract(derived_secret, shared_secret)
        var derived: [32]u8 = undefined;
        Hkdf.expand(&derived, "tls13-derived", early_secret);
        const handshake_secret = Hkdf.extract(&derived, &shared_secret);

        // Transcript hash = Blake2b(client_random || server_random || keys)
        var transcript_hash: [32]u8 = undefined;
        var th = std.crypto.hash.Blake2b256.init(.{});
        th.update(&client_random);
        th.update(&server_random);
        th.update(&client_keypair.public_key);
        th.update(&server_keypair.public_key);
        th.final(&transcript_hash);

        // Client/Server handshake traffic keys
        const traffic_prk = Hkdf.extract(&transcript_hash, &handshake_secret);
        Hkdf.expand(&self.send_key, "tls13-c-hs-traffic", traffic_prk);
        Hkdf.expand(&self.recv_key, "tls13-s-hs-traffic", traffic_prk);

        // Session ID for resumption
        var sid_hash = std.crypto.hash.Blake2b256.init(.{});
        sid_hash.update(&client_random);
        sid_hash.update(&server_random);
        sid_hash.update(&self.send_key);
        sid_hash.final(&self.session_id);

        // Store keypair and peer info
        self.local_keypair = .{
            .public_key = client_keypair.public_key,
            .secret_key = client_keypair.secret_key,
        };
        self.peer_public_key = server_keypair.public_key;

        // Wipe intermediate secrets
        var ss_copy = shared_secret;
        std.crypto.secureZero(u8, &ss_copy);
        std.crypto.secureZero(u8, &zero_salt);
        std.crypto.secureZero(u8, &derived);
    }

    /// Custom X25519-based handshake.
    ///
    /// Uses Curve25519 Diffie-Hellman key exchange with HKDF-SHA256 key
    /// derivation. Generates an ephemeral X25519 key pair, computes a
    /// shared secret with the peer's public key, then derives separate
    /// send and receive keys via HKDF.
    ///
    /// Requires `peer_public_key` to be set before calling.
    fn customHandshake(self: *SecureChannel) ChannelError!void {
        const X25519 = std.crypto.dh.X25519;
        const Hkdf = std.crypto.hkdf.HkdfSha256;

        // Generate ephemeral key pair
        const kp = X25519.KeyPair.generate(null) catch return error.HandshakeFailed;
        self.local_keypair = .{
            .public_key = kp.public_key,
            .secret_key = kp.secret_key,
        };

        // Compute shared secret via DH
        const peer_pk = self.peer_public_key orelse return error.HandshakeFailed;
        const shared_secret = X25519.scalarmult(kp.secret_key, peer_pk) catch
            return error.HandshakeFailed;

        // Derive send and receive keys via HKDF-SHA256
        // Salt with session_id for domain separation
        const prk = Hkdf.extract(&self.session_id, &shared_secret);
        Hkdf.expand(&self.send_key, "abi-send-key", prk);
        Hkdf.expand(&self.recv_key, "abi-recv-key", prk);

        // Securely wipe shared secret
        var ss_copy = shared_secret;
        std.crypto.secureZero(u8, &ss_copy);
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
    const X25519 = std.crypto.dh.X25519;
    const inner = X25519.KeyPair.generate(null) catch return error.HandshakeFailed;
    return .{
        .public_key = inner.public_key,
        .secret_key = inner.secret_key,
    };
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
    NotImplemented,
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

test "TLS handshake derives non-zero keys and sets peer key" {
    const allocator = std.testing.allocator;

    const channel = try SecureChannel.init(allocator, .{ .encryption = .tls_1_3 });
    defer channel.deinit();

    try channel.connect("tls-peer");
    try std.testing.expectEqual(ChannelState.established, channel.state);

    // Keys should be non-zero after handshake
    const zero_key: [32]u8 = std.mem.zeroes([32]u8);
    try std.testing.expect(!std.mem.eql(u8, &channel.send_key, &zero_key));
    try std.testing.expect(!std.mem.eql(u8, &channel.recv_key, &zero_key));

    // Send and recv keys should differ
    try std.testing.expect(!std.mem.eql(u8, &channel.send_key, &channel.recv_key));

    // Peer public key should be set
    try std.testing.expect(channel.peer_public_key != null);

    // Session ID should be non-zero
    try std.testing.expect(!std.mem.eql(u8, &channel.session_id, &zero_key));
}

test "Noise XX handshake derives non-zero keys and sets peer key" {
    const allocator = std.testing.allocator;

    const channel = try SecureChannel.init(allocator, .{ .encryption = .noise_xx });
    defer channel.deinit();

    try channel.connect("noise-peer");
    try std.testing.expectEqual(ChannelState.established, channel.state);

    // Keys should be non-zero after handshake
    const zero_key: [32]u8 = std.mem.zeroes([32]u8);
    try std.testing.expect(!std.mem.eql(u8, &channel.send_key, &zero_key));
    try std.testing.expect(!std.mem.eql(u8, &channel.recv_key, &zero_key));

    // Send and recv keys should differ
    try std.testing.expect(!std.mem.eql(u8, &channel.send_key, &channel.recv_key));

    // Peer public key should be set
    try std.testing.expect(channel.peer_public_key != null);
}

test "WireGuard handshake derives non-zero keys with PSK" {
    const allocator = std.testing.allocator;

    var psk: [32]u8 = undefined;
    std.c.arc4random_buf(&psk, psk.len);

    const channel = try SecureChannel.init(allocator, .{
        .encryption = .wireguard,
        .psk = psk,
    });
    defer channel.deinit();

    try channel.connect("wg-peer");
    try std.testing.expectEqual(ChannelState.established, channel.state);

    // Keys should be non-zero after handshake
    const zero_key: [32]u8 = std.mem.zeroes([32]u8);
    try std.testing.expect(!std.mem.eql(u8, &channel.send_key, &zero_key));
    try std.testing.expect(!std.mem.eql(u8, &channel.recv_key, &zero_key));

    // Send and recv keys should differ
    try std.testing.expect(!std.mem.eql(u8, &channel.send_key, &channel.recv_key));

    // Peer public key should be set
    try std.testing.expect(channel.peer_public_key != null);

    // Session ID should be non-zero (generated via arc4random)
    try std.testing.expect(!std.mem.eql(u8, &channel.session_id, &zero_key));
}

test "WireGuard handshake works without PSK" {
    const allocator = std.testing.allocator;

    const channel = try SecureChannel.init(allocator, .{
        .encryption = .wireguard,
    });
    defer channel.deinit();

    try channel.connect("wg-no-psk-peer");
    try std.testing.expectEqual(ChannelState.established, channel.state);

    const zero_key: [32]u8 = std.mem.zeroes([32]u8);
    try std.testing.expect(!std.mem.eql(u8, &channel.send_key, &zero_key));
}

test {
    std.testing.refAllDecls(@This());
}
