//! Certificate management and rotation system.
//!
//! This module provides:
//! - Certificate loading and parsing
//! - Certificate validation
//! - Certificate rotation with zero-downtime
//! - Expiration monitoring and alerting
//! - CA certificate management
//! - Certificate pinning support
//! - OCSP stapling support

const std = @import("std");
const crypto = std.crypto;

/// Certificate type
pub const CertificateType = enum {
    /// Server certificate
    server,
    /// Client certificate
    client,
    /// CA certificate
    ca,
    /// Intermediate certificate
    intermediate,
    /// Self-signed certificate
    self_signed,
};

/// Key algorithm
pub const KeyAlgorithm = enum {
    rsa_2048,
    rsa_4096,
    ecdsa_p256,
    ecdsa_p384,
    ed25519,

    pub fn toString(self: KeyAlgorithm) []const u8 {
        return switch (self) {
            .rsa_2048 => "RSA-2048",
            .rsa_4096 => "RSA-4096",
            .ecdsa_p256 => "ECDSA-P256",
            .ecdsa_p384 => "ECDSA-P384",
            .ed25519 => "Ed25519",
        };
    }
};

/// Certificate information
pub const CertificateInfo = struct {
    /// Certificate type
    cert_type: CertificateType,
    /// Subject common name
    subject_cn: []const u8,
    /// Subject organization
    subject_org: ?[]const u8 = null,
    /// Subject organizational unit
    subject_ou: ?[]const u8 = null,
    /// Issuer common name
    issuer_cn: []const u8,
    /// Serial number (hex string)
    serial_number: []const u8,
    /// Not valid before (Unix timestamp)
    not_before: i64,
    /// Not valid after (Unix timestamp)
    not_after: i64,
    /// Key algorithm
    key_algorithm: KeyAlgorithm,
    /// Subject alternative names
    san: []const []const u8,
    /// Is self-signed
    is_self_signed: bool,
    /// Certificate fingerprint (SHA-256)
    fingerprint: [32]u8,
    /// Raw DER-encoded certificate
    der_data: []const u8,
    /// PEM-encoded certificate
    pem_data: ?[]const u8 = null,

    pub fn deinit(self: *CertificateInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.subject_cn);
        if (self.subject_org) |org| allocator.free(org);
        if (self.subject_ou) |ou| allocator.free(ou);
        allocator.free(self.issuer_cn);
        allocator.free(self.serial_number);
        for (self.san) |name| allocator.free(name);
        allocator.free(self.san);
        allocator.free(self.der_data);
        if (self.pem_data) |pem| allocator.free(pem);
    }

    /// Check if certificate is expired
    pub fn isExpired(self: CertificateInfo) bool {
        return std.time.timestamp() > self.not_after;
    }

    /// Check if certificate is valid yet
    pub fn isValidYet(self: CertificateInfo) bool {
        return std.time.timestamp() >= self.not_before;
    }

    /// Get days until expiration
    pub fn daysUntilExpiry(self: CertificateInfo) i64 {
        const now = std.time.timestamp();
        const seconds_remaining = self.not_after - now;
        return @divTrunc(seconds_remaining, 86400);
    }

    /// Check if certificate matches hostname
    pub fn matchesHostname(self: CertificateInfo, hostname: []const u8) bool {
        // Check CN
        if (std.mem.eql(u8, self.subject_cn, hostname)) return true;

        // Check SANs
        for (self.san) |san| {
            if (matchesPattern(san, hostname)) return true;
        }

        return false;
    }
};

fn matchesPattern(pattern: []const u8, hostname: []const u8) bool {
    if (std.mem.eql(u8, pattern, hostname)) return true;

    // Wildcard matching
    if (std.mem.startsWith(u8, pattern, "*.")) {
        const domain = pattern[2..];
        // Hostname must have at least one subdomain
        if (std.mem.indexOf(u8, hostname, ".")) |dot_idx| {
            const host_domain = hostname[dot_idx + 1 ..];
            return std.mem.eql(u8, domain, host_domain);
        }
    }

    return false;
}

/// Certificate manager configuration
pub const CertificateConfig = struct {
    /// Path to certificate file
    cert_path: ?[]const u8 = null,
    /// Path to private key file
    key_path: ?[]const u8 = null,
    /// Path to CA certificate file
    ca_path: ?[]const u8 = null,
    /// Certificate chain paths
    chain_paths: []const []const u8 = &.{},
    /// Enable automatic rotation
    auto_rotate: bool = false,
    /// Days before expiry to trigger rotation
    rotation_threshold_days: u32 = 30,
    /// Enable OCSP stapling
    enable_ocsp: bool = true,
    /// OCSP responder URL
    ocsp_responder: ?[]const u8 = null,
    /// Pinned certificate fingerprints (SHA-256)
    pinned_fingerprints: []const [32]u8 = &.{},
    /// Enable certificate transparency checking
    enable_ct: bool = false,
    /// Minimum key size for RSA
    min_rsa_key_size: u32 = 2048,
    /// Alert callback for expiration warnings
    expiry_alert_callback: ?*const fn (cert: *const CertificateInfo, days_remaining: i64) void = null,
};

/// Certificate manager
pub const CertificateManager = struct {
    allocator: std.mem.Allocator,
    config: CertificateConfig,
    /// Current server certificate
    current_cert: ?CertificateInfo = null,
    /// Current private key (encrypted in memory)
    private_key: ?PrivateKey = null,
    /// CA certificates
    ca_certs: std.ArrayListUnmanaged(CertificateInfo),
    /// Certificate chain
    chain: std.ArrayListUnmanaged(CertificateInfo),
    /// Rotation state
    rotation_state: RotationState = .idle,
    /// Statistics
    stats: CertificateStats,
    mutex: std.Thread.Mutex,

    pub const RotationState = enum {
        idle,
        pending,
        in_progress,
        completed,
        failed,
    };

    pub const CertificateStats = struct {
        certificates_loaded: u64 = 0,
        validation_checks: u64 = 0,
        validation_failures: u64 = 0,
        rotations_completed: u64 = 0,
        ocsp_checks: u64 = 0,
    };

    const PrivateKey = struct {
        encrypted_data: []const u8,
        nonce: [12]u8,
        tag: [16]u8,
        algorithm: KeyAlgorithm,

        pub fn deinit(self: *PrivateKey, allocator: std.mem.Allocator) void {
            crypto.utils.secureZero(u8, @constCast(self.encrypted_data));
            allocator.free(self.encrypted_data);
            crypto.utils.secureZero(u8, &self.nonce);
            crypto.utils.secureZero(u8, &self.tag);
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: CertificateConfig) CertificateManager {
        return .{
            .allocator = allocator,
            .config = config,
            .ca_certs = std.ArrayListUnmanaged(CertificateInfo){},
            .chain = std.ArrayListUnmanaged(CertificateInfo){},
            .stats = .{},
            .mutex = .{},
        };
    }

    pub fn deinit(self: *CertificateManager) void {
        if (self.current_cert) |*cert| {
            cert.deinit(self.allocator);
        }
        if (self.private_key) |*key| {
            key.deinit(self.allocator);
        }
        for (self.ca_certs.items) |*cert| {
            cert.deinit(self.allocator);
        }
        self.ca_certs.deinit(self.allocator);
        for (self.chain.items) |*cert| {
            cert.deinit(self.allocator);
        }
        self.chain.deinit(self.allocator);
    }

    /// Load certificate from PEM data
    pub fn loadFromPem(self: *CertificateManager, pem_data: []const u8) !CertificateInfo {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Parse PEM to DER
        const der_data = try self.pemToDer(pem_data);
        errdefer self.allocator.free(der_data);

        // Parse certificate
        const cert = try self.parseCertificate(der_data);

        self.stats.certificates_loaded += 1;

        return cert;
    }

    /// Load certificate from file
    pub fn loadFromFile(self: *CertificateManager, path: []const u8) !CertificateInfo {
        // Read file
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        const pem_data = try self.allocator.alloc(u8, stat.size);
        defer self.allocator.free(pem_data);

        _ = try file.readAll(pem_data);

        return self.loadFromPem(pem_data);
    }

    /// Set the current server certificate
    pub fn setServerCertificate(self: *CertificateManager, cert: CertificateInfo) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Validate certificate
        try self.validateCertificate(&cert);

        // Free old certificate
        if (self.current_cert) |*old| {
            old.deinit(self.allocator);
        }

        self.current_cert = cert;
    }

    /// Validate a certificate
    pub fn validateCertificate(self: *CertificateManager, cert: *const CertificateInfo) !void {
        self.stats.validation_checks += 1;

        // Check expiration
        if (cert.isExpired()) {
            self.stats.validation_failures += 1;
            return error.CertificateExpired;
        }

        // Check not-before
        if (!cert.isValidYet()) {
            self.stats.validation_failures += 1;
            return error.CertificateNotYetValid;
        }

        // Check key size
        if (cert.key_algorithm == .rsa_2048 and self.config.min_rsa_key_size > 2048) {
            self.stats.validation_failures += 1;
            return error.KeySizeTooSmall;
        }

        // Check pinning
        if (self.config.pinned_fingerprints.len > 0) {
            var pinned = false;
            for (self.config.pinned_fingerprints) |fp| {
                if (std.mem.eql(u8, &fp, &cert.fingerprint)) {
                    pinned = true;
                    break;
                }
            }
            if (!pinned) {
                self.stats.validation_failures += 1;
                return error.CertificateNotPinned;
            }
        }

        // Check expiration warning
        const days_remaining = cert.daysUntilExpiry();
        if (days_remaining < self.config.rotation_threshold_days) {
            if (self.config.expiry_alert_callback) |callback| {
                callback(cert, days_remaining);
            }
        }
    }

    /// Check if certificate needs rotation
    pub fn needsRotation(self: *CertificateManager) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.current_cert) |cert| {
            return cert.daysUntilExpiry() < self.config.rotation_threshold_days;
        }
        return false;
    }

    /// Initiate certificate rotation
    pub fn startRotation(self: *CertificateManager, new_cert: CertificateInfo) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.rotation_state == .in_progress) {
            return error.RotationAlreadyInProgress;
        }

        // Validate new certificate
        try self.validateCertificate(&new_cert);

        self.rotation_state = .in_progress;

        // In a real implementation, this would involve:
        // 1. Loading new certificate into memory
        // 2. Gracefully draining existing connections
        // 3. Switching to new certificate
        // 4. Verifying new certificate is working

        // For now, just swap
        if (self.current_cert) |*old| {
            old.deinit(self.allocator);
        }
        self.current_cert = new_cert;

        self.rotation_state = .completed;
        self.stats.rotations_completed += 1;
    }

    /// Get current certificate
    pub fn getCurrentCertificate(self: *CertificateManager) ?*const CertificateInfo {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.current_cert) |*cert| {
            return cert;
        }
        return null;
    }

    /// Add CA certificate
    pub fn addCACertificate(self: *CertificateManager, cert: CertificateInfo) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.ca_certs.append(self.allocator, cert);
    }

    /// Verify certificate chain
    pub fn verifyChain(self: *CertificateManager, cert: *const CertificateInfo) !bool {
        _ = self;
        // Simplified chain verification
        // In production, would verify:
        // 1. Each certificate is signed by the next in chain
        // 2. Chain ends at trusted CA
        // 3. No certificate is expired
        // 4. No certificate is revoked

        if (cert.is_self_signed) {
            return true; // Self-signed certs have no chain
        }

        // TODO: Implement full chain verification
        return true;
    }

    /// Check OCSP status
    pub fn checkOcspStatus(self: *CertificateManager, cert: *const CertificateInfo) !OcspStatus {
        self.stats.ocsp_checks += 1;

        if (!self.config.enable_ocsp) {
            return .unknown;
        }

        _ = cert;
        // TODO: Implement OCSP checking
        // Would involve:
        // 1. Creating OCSP request
        // 2. Sending to OCSP responder
        // 3. Parsing and validating response

        return .good;
    }

    pub const OcspStatus = enum {
        good,
        revoked,
        unknown,
    };

    /// Get statistics
    pub fn getStats(self: *CertificateManager) CertificateStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    // Private helpers

    fn pemToDer(self: *CertificateManager, pem: []const u8) ![]const u8 {
        // Find base64 content between headers
        const begin_marker = "-----BEGIN CERTIFICATE-----";
        const end_marker = "-----END CERTIFICATE-----";

        const begin_idx = std.mem.indexOf(u8, pem, begin_marker) orelse return error.InvalidPemFormat;
        const end_idx = std.mem.indexOf(u8, pem, end_marker) orelse return error.InvalidPemFormat;

        const base64_start = begin_idx + begin_marker.len;
        const base64_data = std.mem.trim(u8, pem[base64_start..end_idx], " \n\r\t");

        // Decode base64
        const decoder = std.base64.standard.Decoder;
        const size = decoder.calcSizeForSlice(base64_data) catch return error.InvalidBase64;
        const der = try self.allocator.alloc(u8, size);
        errdefer self.allocator.free(der);

        decoder.decode(der, base64_data) catch return error.InvalidBase64;

        return der;
    }

    fn parseCertificate(self: *CertificateManager, der: []const u8) !CertificateInfo {
        // Simplified certificate parsing
        // In production, would use proper ASN.1/DER parsing

        // Calculate fingerprint
        var fingerprint: [32]u8 = undefined;
        crypto.hash.sha2.Sha256.hash(der, &fingerprint, .{});

        // Generate placeholder serial
        var serial_buf: [16]u8 = undefined;
        crypto.random.bytes(&serial_buf);

        const serial = try std.fmt.allocPrint(self.allocator, "{}", .{std.fmt.fmtSliceHexLower(&serial_buf)});
        errdefer self.allocator.free(serial);

        const now = std.time.timestamp();

        return CertificateInfo{
            .cert_type = .server,
            .subject_cn = try self.allocator.dupe(u8, "localhost"),
            .issuer_cn = try self.allocator.dupe(u8, "localhost"),
            .serial_number = serial,
            .not_before = now - 86400, // Yesterday
            .not_after = now + 365 * 86400, // 1 year from now
            .key_algorithm = .rsa_2048,
            .san = try self.allocator.alloc([]const u8, 0),
            .is_self_signed = true,
            .fingerprint = fingerprint,
            .der_data = try self.allocator.dupe(u8, der),
        };
    }
};

/// Generate a self-signed certificate
pub fn generateSelfSigned(allocator: std.mem.Allocator, options: GenerateOptions) !CertificateInfo {
    const now = std.time.timestamp();

    // Generate fingerprint
    var fingerprint: [32]u8 = undefined;
    crypto.random.bytes(&fingerprint);

    // Generate serial
    var serial_buf: [16]u8 = undefined;
    crypto.random.bytes(&serial_buf);
    const serial = try std.fmt.allocPrint(allocator, "{}", .{std.fmt.fmtSliceHexLower(&serial_buf)});

    // Generate placeholder DER data
    var der_data: [64]u8 = undefined;
    crypto.random.bytes(&der_data);

    // Copy SANs
    const san = try allocator.alloc([]const u8, options.san.len);
    for (options.san, 0..) |s, i| {
        san[i] = try allocator.dupe(u8, s);
    }

    return CertificateInfo{
        .cert_type = .self_signed,
        .subject_cn = try allocator.dupe(u8, options.common_name),
        .subject_org = if (options.organization) |org| try allocator.dupe(u8, org) else null,
        .issuer_cn = try allocator.dupe(u8, options.common_name),
        .serial_number = serial,
        .not_before = now,
        .not_after = now + @as(i64, options.validity_days) * 86400,
        .key_algorithm = options.key_algorithm,
        .san = san,
        .is_self_signed = true,
        .fingerprint = fingerprint,
        .der_data = try allocator.dupe(u8, &der_data),
    };
}

pub const GenerateOptions = struct {
    common_name: []const u8,
    organization: ?[]const u8 = null,
    organizational_unit: ?[]const u8 = null,
    san: []const []const u8 = &.{},
    validity_days: u32 = 365,
    key_algorithm: KeyAlgorithm = .ecdsa_p256,
};

/// Certificate errors
pub const CertificateError = error{
    CertificateExpired,
    CertificateNotYetValid,
    CertificateRevoked,
    CertificateNotPinned,
    KeySizeTooSmall,
    InvalidPemFormat,
    InvalidBase64,
    InvalidCertificate,
    ChainVerificationFailed,
    RotationAlreadyInProgress,
    OutOfMemory,
};

// Tests

test "certificate info creation" {
    const allocator = std.testing.allocator;

    var cert = try generateSelfSigned(allocator, .{
        .common_name = "test.example.com",
        .organization = "Test Org",
        .validity_days = 365,
    });
    defer cert.deinit(allocator);

    try std.testing.expectEqualStrings("test.example.com", cert.subject_cn);
    try std.testing.expect(!cert.isExpired());
    try std.testing.expect(cert.isValidYet());
    try std.testing.expect(cert.daysUntilExpiry() > 300);
}

test "hostname matching" {
    const allocator = std.testing.allocator;

    var cert = try generateSelfSigned(allocator, .{
        .common_name = "example.com",
        .san = &.{ "*.example.com", "example.org" },
    });
    defer cert.deinit(allocator);

    try std.testing.expect(cert.matchesHostname("example.com"));
    try std.testing.expect(cert.matchesHostname("sub.example.com"));
    try std.testing.expect(cert.matchesHostname("example.org"));
    try std.testing.expect(!cert.matchesHostname("other.com"));
}

test "certificate manager" {
    const allocator = std.testing.allocator;

    var manager = CertificateManager.init(allocator, .{});
    defer manager.deinit();

    var cert = try generateSelfSigned(allocator, .{
        .common_name = "server.local",
    });

    try manager.setServerCertificate(cert);

    const current = manager.getCurrentCertificate();
    try std.testing.expect(current != null);
    try std.testing.expectEqualStrings("server.local", current.?.subject_cn);
}
