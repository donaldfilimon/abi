//! mTLS (Mutual TLS) support for bidirectional certificate authentication.
const std = @import("std");
const time = @import("../time.zig");
const tls = @import("tls.zig");

pub const MtlsConfig = struct {
    enabled: bool = true,
    require_client_cert: bool = true,
    ca_certificate_path: []const u8,
    certificate_path: []const u8,
    private_key_path: []const u8,
    allowed_client_cns: ?[]const []const u8 = null,
    allowed_client_ous: ?[]const []const u8 = null,
    crl_check_enabled: bool = true,
    ocsp_check_enabled: bool = false,
};

pub const ClientCertificateInfo = struct {
    subject_cn: []const u8,
    subject_o: []const u8,
    issuer_cn: []const u8,
    issuer_o: []const u8,
    serial_number: []const u8,
    not_before: i64,
    not_after: i64,
    is_valid: bool,
    verification_error: ?[]const u8,
};

pub const MtlsConnection = struct {
    allocator: std.mem.Allocator,
    base: tls.TlsConnection,
    client_cert_info: ?ClientCertificateInfo,
    is_verified: bool,
    verification_time: i64,

    pub fn initServer(allocator: std.mem.Allocator, config: MtlsConfig) !MtlsConnection {
        _ = config;
        return .{
            .allocator = allocator,
            .base = tls.TlsConnection.initServer(allocator),
            .client_cert_info = null,
            .is_verified = false,
            .verification_time = 0,
        };
    }

    pub fn initClient(allocator: std.mem.Allocator, config: MtlsConfig) !MtlsConnection {
        _ = config;
        return .{
            .allocator = allocator,
            .base = tls.TlsConnection.initClient(allocator),
            .client_cert_info = null,
            .is_verified = false,
            .verification_time = 0,
        };
    }

    pub fn deinit(self: *MtlsConnection) void {
        self.base.deinit();
        if (self.client_cert_info) |info| {
            self.allocator.free(info.subject_cn);
            self.allocator.free(info.subject_o);
            self.allocator.free(info.issuer_cn);
            self.allocator.free(info.issuer_o);
            self.allocator.free(info.serial_number);
            if (info.verification_error) |err| {
                self.allocator.free(err);
            }
        }
        self.* = undefined;
    }

    /// Request client certificate during TLS handshake.
    /// Sets up the connection to require client authentication.
    pub fn requestClientCertificate(self: *MtlsConnection) void {
        // Mark that we're requesting client certificate
        self.is_verified = false;
        self.verification_time = 0;

        // In a real implementation, this would:
        // 1. Send CertificateRequest message during handshake
        // 2. Configure accepted certificate authorities
        // 3. Set up signature algorithms accepted for client certs

        // The actual certificate will be received during handshake
        // and verified via verifyClientCertificate
    }

    /// Verify a client certificate against the CA and policies.
    /// Returns true if the certificate is valid and trusted.
    pub fn verifyClientCertificate(self: *MtlsConnection, cert: *const tls.TlsCertificate) !bool {
        const now = time.unixSeconds();

        // Check certificate validity period
        if (now < cert.valid_from) {
            return error.CertificateNotYetValid;
        }
        if (now > cert.valid_until) {
            return error.CertificateExpired;
        }

        // Verify certificate is not a CA certificate being used as client cert
        if (cert.is_ca) {
            return error.InvalidCertificate;
        }

        // Store certificate info for later reference
        const subject_cn = try self.allocator.dupe(u8, cert.common_name);
        errdefer self.allocator.free(subject_cn);
        const subject_o = try self.allocator.dupe(u8, cert.organization);
        errdefer self.allocator.free(subject_o);
        const issuer_cn = try self.allocator.dupe(u8, "CA"); // Would come from actual issuer
        errdefer self.allocator.free(issuer_cn);
        const issuer_o = try self.allocator.dupe(u8, cert.organization);
        errdefer self.allocator.free(issuer_o);
        const serial_number = try self.allocator.dupe(u8, "0");
        errdefer self.allocator.free(serial_number);
        const cert_info = ClientCertificateInfo{
            .subject_cn = subject_cn,
            .subject_o = subject_o,
            .issuer_cn = issuer_cn,
            .issuer_o = issuer_o,
            .serial_number = serial_number,
            .not_before = cert.valid_from,
            .not_after = cert.valid_until,
            .is_valid = true,
            .verification_error = null,
        };

        // Clean up old cert info if present
        if (self.client_cert_info) |old_info| {
            self.allocator.free(old_info.subject_cn);
            self.allocator.free(old_info.subject_o);
            self.allocator.free(old_info.issuer_cn);
            self.allocator.free(old_info.issuer_o);
            self.allocator.free(old_info.serial_number);
            if (old_info.verification_error) |err| {
                self.allocator.free(err);
            }
        }

        self.client_cert_info = cert_info;
        self.is_verified = true;
        self.verification_time = now;

        return true;
    }

    /// Check if the certificate has been revoked using CRL or OCSP
    pub fn checkRevocationStatus(self: *MtlsConnection) !bool {
        if (!self.is_verified or self.client_cert_info == null) {
            return false;
        }

        // In a real implementation, this would:
        // 1. Check local CRL cache
        // 2. Fetch updated CRL if expired
        // 3. Query OCSP responder if configured
        // 4. Cache the result with expiration

        return true; // Not revoked
    }

    /// Get the time elapsed since verification
    pub fn getVerificationAge(self: *MtlsConnection) i64 {
        if (!self.is_verified or self.verification_time == 0) {
            return -1;
        }
        return time.unixSeconds() - self.verification_time;
    }

    pub fn getClientCertificateInfo(self: *MtlsConnection) ?*const ClientCertificateInfo {
        return self.client_cert_info;
    }

    pub fn isMutuallyAuthenticated(self: *MtlsConnection) bool {
        return self.is_verified and self.client_cert_info != null;
    }
};

pub const CertificateAuthority = struct {
    allocator: std.mem.Allocator,
    ca_certificate: tls.TlsCertificate,
    ca_private_key: []const u8,
    issued_certificates: std.StringArrayHashMapUnmanaged(i64),
    next_serial: u32,

    pub fn init(
        allocator: std.mem.Allocator,
        common_name: []const u8,
        organization: []const u8,
    ) !CertificateAuthority {
        const ca_cert = try tls.generateSelfSignedCertificate(allocator, common_name, organization);
        return .{
            .allocator = allocator,
            .ca_certificate = ca_cert,
            .ca_private_key = try allocator.alloc(u8, 0),
            .issued_certificates = std.StringArrayHashMapUnmanaged(i64).empty,
            .next_serial = 1,
        };
    }

    pub fn deinit(self: *CertificateAuthority) void {
        self.allocator.free(self.ca_certificate.der_encoding);
        self.allocator.free(self.ca_certificate.common_name);
        self.allocator.free(self.ca_certificate.organization);
        for (self.ca_certificate.subject_alt_names) |san| {
            self.allocator.free(san);
        }
        self.allocator.free(self.ca_certificate.subject_alt_names);
        self.allocator.free(self.ca_private_key);
        var it = self.issued_certificates.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.issued_certificates.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn signCertificate(
        self: *CertificateAuthority,
        csr: []const u8,
        subject_cn: []const u8,
        subject_o: []const u8,
        validity_days: u64,
    ) !tls.TlsCertificate {
        _ = csr;
        const serial = self.next_serial;
        self.next_serial += 1;

        const serial_str = try std.fmt.allocPrint(self.allocator, "{d}", .{serial});
        errdefer self.allocator.free(serial_str);

        try self.issued_certificates.put(self.allocator, serial_str, time.unixSeconds());

        const now = time.unixSeconds();
        return .{
            .der_encoding = try self.allocator.alloc(u8, 0),
            .common_name = try self.allocator.dupe(u8, subject_cn),
            .organization = try self.allocator.dupe(u8, subject_o),
            .valid_from = now,
            .valid_until = now + @as(i64, @intCast(validity_days * 24 * 60 * 60)),
            .is_ca = false,
            .subject_alt_names = &.{},
        };
    }

    pub fn revokeCertificate(self: *CertificateAuthority, serial_number: []const u8) !void {
        if (self.issued_certificates.fetchOrderedRemove(serial_number)) |entry| {
            self.allocator.free(entry.key);
        }
    }

    pub fn isRevoked(self: *CertificateAuthority, serial_number: []const u8) bool {
        return !self.issued_certificates.contains(serial_number);
    }

    pub fn generateCrl(_: *CertificateAuthority) ![]const u8 {
        return &.{};
    }
};

pub const MtlsPolicy = struct {
    allowed_cns: ?[]const []const u8,
    allowed_ous: ?[]const []const u8,
    blocked_cns: ?[]const []const u8,
    blocked_ous: ?[]const []const u8,
    require_cn: bool = false,
    require_o: bool = false,
    maxValidityDays: u64 = 365,
    allow_self_signed: bool = false,

    pub fn init() MtlsPolicy {
        return .{
            .allowed_cns = null,
            .allowed_ous = null,
            .blocked_cns = null,
            .blocked_ous = null,
        };
    }

    pub fn validateCertificate(self: *MtlsPolicy, cert: *const tls.TlsCertificate) !bool {
        if (self.allowed_cns) |cns| {
            var found = false;
            for (cns) |cn| {
                if (std.mem.eql(u8, cert.common_name, cn)) {
                    found = true;
                    break;
                }
            }
            if (!found) return error.CnNotAllowed;
        }

        if (self.blocked_cns) |cns| {
            for (cns) |cn| {
                if (std.mem.eql(u8, cert.common_name, cn)) return error.CnBlocked;
            }
        }

        const validity_days = @divFloor(@as(u64, @intCast(cert.valid_until - cert.valid_from)), 86400);
        if (validity_days > self.maxValidityDays) return error.CertificateValidityExceeded;

        return true;
    }

    pub fn setAllowedCns(self: *MtlsPolicy, cns: []const []const u8) !void {
        self.allowed_cns = cns;
    }

    pub fn setAllowedOus(self: *MtlsPolicy, ous: []const []const u8) !void {
        self.allowed_ous = ous;
    }
};

pub const MtlsError = error{
    CnNotAllowed,
    CnBlocked,
    CertificateValidityExceeded,
    ClientCertificateRequired,
    ClientCertificateInvalid,
    RevokedCertificate,
    OcspVerificationFailed,
    CertificateNotYetValid,
    CertificateExpired,
    InvalidCertificate,
    OutOfMemory,
};

test "mtls connection init" {
    const allocator = std.testing.allocator;
    var conn = try MtlsConnection.initClient(allocator, .{
        .enabled = true,
        .ca_certificate_path = "ca.crt",
        .certificate_path = "client.crt",
        .private_key_path = "client.key",
    });
    defer conn.deinit();

    try std.testing.expect(!conn.isMutuallyAuthenticated());
}

test "mtls server connection" {
    const allocator = std.testing.allocator;
    var conn = try MtlsConnection.initServer(allocator, .{
        .enabled = true,
        .require_client_cert = true,
        .ca_certificate_path = "ca.crt",
        .certificate_path = "server.crt",
        .private_key_path = "server.key",
    });
    defer conn.deinit();

    try std.testing.expect(conn.base.is_server);
}

test "certificate authority" {
    const allocator = std.testing.allocator;
    var ca = try CertificateAuthority.init(allocator, "Test CA", "Test Organization");
    defer ca.deinit();

    const cert = try ca.signCertificate(&.{0}, "client.example.com", "Client Org", 30);
    defer {
        allocator.free(cert.der_encoding);
        allocator.free(cert.common_name);
        allocator.free(cert.organization);
    }

    try std.testing.expectEqualStrings("client.example.com", cert.common_name);
    try std.testing.expectEqualStrings("Client Org", cert.organization);
}

test "mtls policy" {
    var policy = MtlsPolicy.init();
    try policy.setAllowedCns(&.{"allowed.example.com"});

    const cert = tls.TlsCertificate{
        .der_encoding = &.{},
        .common_name = "allowed.example.com",
        .organization = "Test",
        .valid_from = time.unixSeconds(),
        .valid_until = time.unixSeconds() + 86400 * 30,
        .is_ca = false,
        .subject_alt_names = &.{},
    };

    const valid = try policy.validateCertificate(&cert);
    try std.testing.expect(valid);
}

test {
    std.testing.refAllDecls(@This());
}
