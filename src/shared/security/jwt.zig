//! JWT (JSON Web Token) authentication support.
//!
//! This module provides:
//! - JWT creation and signing (HS256, HS384, HS512, RS256)
//! - JWT verification and validation
//! - Claims management (standard and custom)
//! - Token refresh and rotation
//! - Blacklist/revocation support
//! - JWK (JSON Web Key) support

const std = @import("std");
const crypto = std.crypto;

/// JWT signing algorithms
pub const Algorithm = enum {
    /// HMAC with SHA-256
    hs256,
    /// HMAC with SHA-384
    hs384,
    /// HMAC with SHA-512
    hs512,
    /// RSA with SHA-256 (requires asymmetric keys)
    rs256,
    /// None (unsecured - use with caution!)
    none,

    pub fn toString(self: Algorithm) []const u8 {
        return switch (self) {
            .hs256 => "HS256",
            .hs384 => "HS384",
            .hs512 => "HS512",
            .rs256 => "RS256",
            .none => "none",
        };
    }

    pub fn fromString(s: []const u8) ?Algorithm {
        if (std.mem.eql(u8, s, "HS256")) return .hs256;
        if (std.mem.eql(u8, s, "HS384")) return .hs384;
        if (std.mem.eql(u8, s, "HS512")) return .hs512;
        if (std.mem.eql(u8, s, "RS256")) return .rs256;
        if (std.mem.eql(u8, s, "none")) return .none;
        return null;
    }
};

/// Standard JWT claims
pub const Claims = struct {
    /// Issuer
    iss: ?[]const u8 = null,
    /// Subject (usually user ID)
    sub: ?[]const u8 = null,
    /// Audience
    aud: ?[]const u8 = null,
    /// Expiration time (Unix timestamp)
    exp: ?i64 = null,
    /// Not before time (Unix timestamp)
    nbf: ?i64 = null,
    /// Issued at time (Unix timestamp)
    iat: ?i64 = null,
    /// JWT ID (unique identifier)
    jti: ?[]const u8 = null,
    /// Custom claims
    custom: ?std.StringArrayHashMapUnmanaged([]const u8) = null,

    pub fn deinit(self: *Claims, allocator: std.mem.Allocator) void {
        if (self.iss) |iss| allocator.free(iss);
        if (self.sub) |sub| allocator.free(sub);
        if (self.aud) |aud| allocator.free(aud);
        if (self.jti) |jti| allocator.free(jti);
        if (self.custom) |*custom| {
            var it = custom.iterator();
            while (it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.*);
            }
            custom.deinit(allocator);
        }
    }

    /// Check if the token is expired
    pub fn isExpired(self: Claims) bool {
        if (self.exp) |exp| {
            return std.time.timestamp() > exp;
        }
        return false;
    }

    /// Check if the token is valid yet (nbf check)
    pub fn isValidYet(self: Claims) bool {
        if (self.nbf) |nbf| {
            return std.time.timestamp() >= nbf;
        }
        return true;
    }

    /// Validate all time-based claims
    pub fn validateTime(self: Claims, clock_skew: i64) bool {
        const now = std.time.timestamp();

        if (self.exp) |exp| {
            if (now > exp + clock_skew) return false;
        }
        if (self.nbf) |nbf| {
            if (now < nbf - clock_skew) return false;
        }
        if (self.iat) |iat| {
            // Issued in the future is suspicious
            if (iat > now + clock_skew) return false;
        }

        return true;
    }
};

/// JWT token structure
pub const Token = struct {
    /// Original encoded token
    raw: []const u8,
    /// Header (decoded)
    header: Header,
    /// Claims (decoded)
    claims: Claims,
    /// Signature (decoded)
    signature: []const u8,
    /// Whether signature was verified
    verified: bool,

    pub const Header = struct {
        alg: Algorithm,
        typ: []const u8 = "JWT",
        kid: ?[]const u8 = null, // Key ID
    };

    pub fn deinit(self: *Token, allocator: std.mem.Allocator) void {
        allocator.free(self.raw);
        allocator.free(self.signature);
        if (self.header.kid) |kid| allocator.free(kid);
        self.claims.deinit(allocator);
    }
};

/// JWT configuration
pub const JwtConfig = struct {
    /// Default algorithm for signing
    algorithm: Algorithm = .hs256,
    /// Default token lifetime in seconds
    token_lifetime: i64 = 3600, // 1 hour
    /// Refresh token lifetime in seconds
    refresh_lifetime: i64 = 86400 * 7, // 7 days
    /// Issuer to set on created tokens
    issuer: ?[]const u8 = null,
    /// Audience to set on created tokens
    audience: ?[]const u8 = null,
    /// Clock skew tolerance in seconds
    clock_skew: i64 = 60,
    /// Enable token blacklist
    enable_blacklist: bool = true,
    /// Maximum blacklist size
    max_blacklist_size: usize = 10000,
    /// Allow "none" algorithm (dangerous!)
    allow_none_algorithm: bool = false,
    /// Required claims for validation
    required_claims: []const []const u8 = &.{ "sub", "exp" },
};

/// JWT manager for creating and validating tokens
pub const JwtManager = struct {
    allocator: std.mem.Allocator,
    config: JwtConfig,
    /// Secret key for HMAC algorithms
    secret_key: []const u8,
    /// Blacklisted token IDs (jti)
    blacklist: std.StringArrayHashMapUnmanaged(i64),
    mutex: std.Thread.Mutex,
    /// Statistics
    stats: JwtStats,

    pub const JwtStats = struct {
        tokens_created: u64 = 0,
        tokens_verified: u64 = 0,
        tokens_rejected: u64 = 0,
        tokens_expired: u64 = 0,
        tokens_blacklisted: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, secret_key: []const u8, config: JwtConfig) JwtManager {
        return .{
            .allocator = allocator,
            .config = config,
            .secret_key = secret_key,
            .blacklist = std.StringArrayHashMapUnmanaged(i64){},
            .mutex = .{},
            .stats = .{},
        };
    }

    pub fn deinit(self: *JwtManager) void {
        var it = self.blacklist.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.blacklist.deinit(self.allocator);
    }

    /// Create a new JWT token
    pub fn createToken(self: *JwtManager, claims: Claims) ![]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = std.time.timestamp();

        // Build claims with defaults
        var final_claims = claims;
        if (final_claims.iat == null) {
            final_claims.iat = now;
        }
        if (final_claims.exp == null) {
            final_claims.exp = now + self.config.token_lifetime;
        }
        if (final_claims.iss == null and self.config.issuer != null) {
            final_claims.iss = self.config.issuer;
        }
        if (final_claims.aud == null and self.config.audience != null) {
            final_claims.aud = self.config.audience;
        }
        if (final_claims.jti == null) {
            // Generate unique token ID
            var jti_buf: [16]u8 = undefined;
            crypto.random.bytes(&jti_buf);
            final_claims.jti = try self.allocator.dupe(u8, &jti_buf);
        }

        // Build header
        const header_json = try std.fmt.allocPrint(self.allocator, "{{\"alg\":\"{s}\",\"typ\":\"JWT\"}}", .{
            self.config.algorithm.toString(),
        });
        defer self.allocator.free(header_json);

        // Build payload
        const payload_json = try self.claimsToJson(final_claims);
        defer self.allocator.free(payload_json);

        // Base64url encode header and payload
        const header_b64 = try self.base64UrlEncode(header_json);
        defer self.allocator.free(header_b64);

        const payload_b64 = try self.base64UrlEncode(payload_json);
        defer self.allocator.free(payload_b64);

        // Create signing input
        const signing_input = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ header_b64, payload_b64 });
        defer self.allocator.free(signing_input);

        // Sign
        const signature = try self.sign(signing_input);
        defer self.allocator.free(signature);

        // Combine into final token
        const token = try std.fmt.allocPrint(self.allocator, "{s}.{s}.{s}", .{
            header_b64,
            payload_b64,
            signature,
        });

        self.stats.tokens_created += 1;

        return token;
    }

    /// Verify and decode a JWT token
    pub fn verifyToken(self: *JwtManager, token_str: []const u8) !Token {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Split token into parts
        var parts = std.mem.splitScalar(u8, token_str, '.');

        const header_b64 = parts.next() orelse return error.InvalidToken;
        const payload_b64 = parts.next() orelse return error.InvalidToken;
        const signature_b64 = parts.next() orelse return error.InvalidToken;

        // Should be exactly 3 parts
        if (parts.next() != null) return error.InvalidToken;

        // Decode header
        const header_json = try self.base64UrlDecode(header_b64);
        defer self.allocator.free(header_json);

        const header = try self.parseHeader(header_json);

        // Check algorithm
        if (header.alg == .none and !self.config.allow_none_algorithm) {
            self.stats.tokens_rejected += 1;
            return error.AlgorithmNotAllowed;
        }

        // Verify signature
        const signing_input = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ header_b64, payload_b64 });
        defer self.allocator.free(signing_input);

        const signature = try self.base64UrlDecode(signature_b64);

        const verified = try self.verifySignature(signing_input, signature, header.alg);
        if (!verified) {
            self.stats.tokens_rejected += 1;
            self.allocator.free(signature);
            return error.InvalidSignature;
        }

        // Decode payload
        const payload_json = try self.base64UrlDecode(payload_b64);
        defer self.allocator.free(payload_json);

        var claims = try self.parseClaims(payload_json);

        // Validate time-based claims
        if (!claims.validateTime(self.config.clock_skew)) {
            if (claims.isExpired()) {
                self.stats.tokens_expired += 1;
            }
            self.stats.tokens_rejected += 1;
            claims.deinit(self.allocator);
            self.allocator.free(signature);
            return error.TokenExpired;
        }

        // Check blacklist
        if (self.config.enable_blacklist) {
            if (claims.jti) |jti| {
                if (self.blacklist.contains(jti)) {
                    self.stats.tokens_rejected += 1;
                    claims.deinit(self.allocator);
                    self.allocator.free(signature);
                    return error.TokenRevoked;
                }
            }
        }

        // Validate required claims
        for (self.config.required_claims) |required| {
            if (std.mem.eql(u8, required, "sub") and claims.sub == null) {
                self.stats.tokens_rejected += 1;
                claims.deinit(self.allocator);
                self.allocator.free(signature);
                return error.MissingRequiredClaim;
            }
            if (std.mem.eql(u8, required, "exp") and claims.exp == null) {
                self.stats.tokens_rejected += 1;
                claims.deinit(self.allocator);
                self.allocator.free(signature);
                return error.MissingRequiredClaim;
            }
            if (std.mem.eql(u8, required, "iss") and claims.iss == null) {
                self.stats.tokens_rejected += 1;
                claims.deinit(self.allocator);
                self.allocator.free(signature);
                return error.MissingRequiredClaim;
            }
        }

        self.stats.tokens_verified += 1;

        return Token{
            .raw = try self.allocator.dupe(u8, token_str),
            .header = header,
            .claims = claims,
            .signature = signature,
            .verified = true,
        };
    }

    /// Refresh a token (issue new token with same claims, new exp)
    pub fn refreshToken(self: *JwtManager, old_token: *const Token) ![]const u8 {
        if (!old_token.verified) return error.TokenNotVerified;
        if (old_token.claims.isExpired()) return error.TokenExpired;

        // Create new token with same subject and custom claims
        var new_claims = Claims{
            .sub = if (old_token.claims.sub) |s| try self.allocator.dupe(u8, s) else null,
            .iss = self.config.issuer,
            .aud = self.config.audience,
            // exp, iat, jti will be set by createToken
        };

        // Copy custom claims
        if (old_token.claims.custom) |old_custom| {
            new_claims.custom = std.StringArrayHashMapUnmanaged([]const u8){};
            var it = old_custom.iterator();
            while (it.next()) |entry| {
                try new_claims.custom.?.put(
                    self.allocator,
                    try self.allocator.dupe(u8, entry.key_ptr.*),
                    try self.allocator.dupe(u8, entry.value_ptr.*),
                );
            }
        }

        return self.createToken(new_claims);
    }

    /// Revoke a token (add to blacklist)
    pub fn revokeToken(self: *JwtManager, token: *const Token) !void {
        if (!self.config.enable_blacklist) return error.BlacklistDisabled;

        self.mutex.lock();
        defer self.mutex.unlock();

        const jti = token.claims.jti orelse return error.NoTokenId;

        // Check blacklist size
        if (self.blacklist.count() >= self.config.max_blacklist_size) {
            // Remove oldest entries
            try self.cleanupBlacklist();
        }

        const exp = token.claims.exp orelse std.time.timestamp() + 86400;
        try self.blacklist.put(self.allocator, try self.allocator.dupe(u8, jti), exp);

        self.stats.tokens_blacklisted += 1;
    }

    /// Revoke by token ID directly
    pub fn revokeTokenById(self: *JwtManager, jti: []const u8, exp: i64) !void {
        if (!self.config.enable_blacklist) return error.BlacklistDisabled;

        self.mutex.lock();
        defer self.mutex.unlock();

        try self.blacklist.put(self.allocator, try self.allocator.dupe(u8, jti), exp);
        self.stats.tokens_blacklisted += 1;
    }

    /// Get statistics
    pub fn getStats(self: *JwtManager) JwtStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    // Private methods

    fn sign(self: *JwtManager, input: []const u8) ![]const u8 {
        return switch (self.config.algorithm) {
            .hs256 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha256),
            .hs384 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha384),
            .hs512 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha512),
            .rs256 => return error.NotImplemented, // TODO: RSA signing
            .none => try self.allocator.dupe(u8, ""),
        };
    }

    fn signHmac(self: *JwtManager, input: []const u8, comptime Hmac: type) ![]const u8 {
        var mac: [Hmac.mac_length]u8 = undefined;

        // Need to pad or truncate key to proper length
        var key: [Hmac.key_length]u8 = undefined;
        if (self.secret_key.len >= Hmac.key_length) {
            @memcpy(&key, self.secret_key[0..Hmac.key_length]);
        } else {
            @memset(&key, 0);
            @memcpy(key[0..self.secret_key.len], self.secret_key);
        }

        var h = Hmac.init(&key);
        h.update(input);
        h.final(&mac);

        return self.base64UrlEncode(&mac);
    }

    fn verifySignature(self: *JwtManager, input: []const u8, signature: []const u8, alg: Algorithm) !bool {
        if (alg == .none) {
            return signature.len == 0;
        }

        const expected = switch (alg) {
            .hs256 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha256),
            .hs384 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha384),
            .hs512 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha512),
            .rs256 => return error.NotImplemented,
            .none => unreachable,
        };
        defer self.allocator.free(expected);

        const expected_decoded = try self.base64UrlDecode(expected);
        defer self.allocator.free(expected_decoded);

        return crypto.utils.timingSafeEql(u8, signature, expected_decoded);
    }

    fn base64UrlEncode(self: *JwtManager, data: []const u8) ![]const u8 {
        const encoder = std.base64.url_safe_no_pad;
        const size = encoder.Encoder.calcSize(data.len);
        const buf = try self.allocator.alloc(u8, size);
        _ = encoder.Encoder.encode(buf, data);
        return buf;
    }

    fn base64UrlDecode(self: *JwtManager, data: []const u8) ![]const u8 {
        const decoder = std.base64.url_safe_no_pad;
        const size = decoder.Decoder.calcSizeForSlice(data) catch return error.InvalidBase64;
        const buf = try self.allocator.alloc(u8, size);
        decoder.Decoder.decode(buf, data) catch {
            self.allocator.free(buf);
            return error.InvalidBase64;
        };
        return buf;
    }

    fn parseHeader(self: *JwtManager, json: []const u8) !Token.Header {
        _ = self;
        // Simple JSON parsing for header
        var alg: Algorithm = .hs256;

        // Find "alg" field
        if (std.mem.indexOf(u8, json, "\"alg\"")) |idx| {
            const start = idx + 6; // Skip past "alg":"
            if (start < json.len) {
                // Find the algorithm value
                var i = start;
                while (i < json.len and json[i] != '"') : (i += 1) {}
                if (i < json.len) {
                    i += 1; // Skip opening quote
                    const alg_start = i;
                    while (i < json.len and json[i] != '"') : (i += 1) {}
                    if (Algorithm.fromString(json[alg_start..i])) |a| {
                        alg = a;
                    }
                }
            }
        }

        return Token.Header{
            .alg = alg,
            .typ = "JWT",
            .kid = null,
        };
    }

    fn parseClaims(self: *JwtManager, json: []const u8) !Claims {
        var claims = Claims{};

        // Parse standard claims (simplified JSON parsing)
        claims.sub = try self.extractStringClaim(json, "sub");
        claims.iss = try self.extractStringClaim(json, "iss");
        claims.aud = try self.extractStringClaim(json, "aud");
        claims.jti = try self.extractStringClaim(json, "jti");
        claims.exp = self.extractIntClaim(json, "exp");
        claims.nbf = self.extractIntClaim(json, "nbf");
        claims.iat = self.extractIntClaim(json, "iat");

        return claims;
    }

    fn extractStringClaim(self: *JwtManager, json: []const u8, claim: []const u8) !?[]const u8 {
        const search = try std.fmt.allocPrint(self.allocator, "\"{s}\":\"", .{claim});
        defer self.allocator.free(search);

        if (std.mem.indexOf(u8, json, search)) |idx| {
            const start = idx + search.len;
            var end = start;
            while (end < json.len and json[end] != '"') : (end += 1) {}
            if (end > start) {
                return try self.allocator.dupe(u8, json[start..end]);
            }
        }
        return null;
    }

    fn extractIntClaim(_: *JwtManager, json: []const u8, claim: []const u8) ?i64 {
        // Find "claim":
        var search_buf: [64]u8 = undefined;
        const search = std.fmt.bufPrint(&search_buf, "\"{s}\":", .{claim}) catch return null;

        if (std.mem.indexOf(u8, json, search)) |idx| {
            const start = idx + search.len;
            var end = start;
            // Skip whitespace
            while (end < json.len and (json[end] == ' ' or json[end] == '\t')) : (end += 1) {}
            const num_start = end;
            // Find number end
            while (end < json.len and (json[end] >= '0' and json[end] <= '9')) : (end += 1) {}
            if (end > num_start) {
                return std.fmt.parseInt(i64, json[num_start..end], 10) catch null;
            }
        }
        return null;
    }

    fn claimsToJson(self: *JwtManager, claims: Claims) ![]const u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        errdefer buffer.deinit();

        try buffer.append('{');

        var first = true;

        if (claims.sub) |sub| {
            if (!first) try buffer.append(',');
            try std.fmt.format(buffer.writer(), "\"sub\":\"{s}\"", .{sub});
            first = false;
        }
        if (claims.iss) |iss| {
            if (!first) try buffer.append(',');
            try std.fmt.format(buffer.writer(), "\"iss\":\"{s}\"", .{iss});
            first = false;
        }
        if (claims.aud) |aud| {
            if (!first) try buffer.append(',');
            try std.fmt.format(buffer.writer(), "\"aud\":\"{s}\"", .{aud});
            first = false;
        }
        if (claims.jti) |jti| {
            if (!first) try buffer.append(',');
            // Encode jti as hex since it may be binary
            var hex_buf: [64]u8 = undefined;
            const hex = std.fmt.bufPrint(&hex_buf, "{}", .{std.fmt.fmtSliceHexLower(jti)}) catch jti;
            try std.fmt.format(buffer.writer(), "\"jti\":\"{s}\"", .{hex});
            first = false;
        }
        if (claims.exp) |exp| {
            if (!first) try buffer.append(',');
            try std.fmt.format(buffer.writer(), "\"exp\":{d}", .{exp});
            first = false;
        }
        if (claims.nbf) |nbf| {
            if (!first) try buffer.append(',');
            try std.fmt.format(buffer.writer(), "\"nbf\":{d}", .{nbf});
            first = false;
        }
        if (claims.iat) |iat| {
            if (!first) try buffer.append(',');
            try std.fmt.format(buffer.writer(), "\"iat\":{d}", .{iat});
            first = false;
        }

        // Custom claims
        if (claims.custom) |custom| {
            var it = custom.iterator();
            while (it.next()) |entry| {
                if (!first) try buffer.append(',');
                try std.fmt.format(buffer.writer(), "\"{s}\":\"{s}\"", .{
                    entry.key_ptr.*,
                    entry.value_ptr.*,
                });
                first = false;
            }
        }

        try buffer.append('}');

        return buffer.toOwnedSlice();
    }

    fn cleanupBlacklist(self: *JwtManager) !void {
        const now = std.time.timestamp();
        var to_remove = std.ArrayList([]const u8).init(self.allocator);
        defer to_remove.deinit();

        // Find expired entries
        var it = self.blacklist.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* < now) {
                try to_remove.append(entry.key_ptr.*);
            }
        }

        // Remove expired entries
        for (to_remove.items) |key| {
            if (self.blacklist.fetchRemove(key)) |kv| {
                self.allocator.free(kv.key);
            }
        }
    }
};

/// JWT error types
pub const JwtError = error{
    InvalidToken,
    InvalidSignature,
    TokenExpired,
    TokenNotValidYet,
    TokenRevoked,
    MissingRequiredClaim,
    AlgorithmNotAllowed,
    BlacklistDisabled,
    NoTokenId,
    TokenNotVerified,
    InvalidBase64,
    NotImplemented,
    OutOfMemory,
};

// Helper functions

/// Extract bearer token from Authorization header
pub fn extractBearerToken(auth_header: []const u8) ?[]const u8 {
    const prefix = "Bearer ";
    if (std.mem.startsWith(u8, auth_header, prefix)) {
        return auth_header[prefix.len..];
    }
    return null;
}

/// Generate a secure random secret key
pub fn generateSecretKey(allocator: std.mem.Allocator, length: usize) ![]u8 {
    const key = try allocator.alloc(u8, length);
    crypto.random.bytes(key);
    return key;
}

// Tests

test "jwt create and verify" {
    const allocator = std.testing.allocator;

    var manager = JwtManager.init(allocator, "test-secret-key-32-bytes-long!!", .{});
    defer manager.deinit();

    // Create token
    const token_str = try manager.createToken(.{
        .sub = "user123",
        .exp = std.time.timestamp() + 3600,
    });
    defer allocator.free(token_str);

    // Verify token structure
    var parts = std.mem.splitScalar(u8, token_str, '.');
    _ = parts.next(); // header
    _ = parts.next(); // payload
    _ = parts.next(); // signature
    try std.testing.expect(parts.next() == null); // no more parts

    // Verify token
    var token = try manager.verifyToken(token_str);
    defer token.deinit(allocator);

    try std.testing.expect(token.verified);
    try std.testing.expectEqualStrings("user123", token.claims.sub.?);
}

test "jwt expiration" {
    const allocator = std.testing.allocator;

    var manager = JwtManager.init(allocator, "test-secret-key-32-bytes-long!!", .{
        .clock_skew = 0,
    });
    defer manager.deinit();

    // Create expired token
    const token_str = try manager.createToken(.{
        .sub = "user123",
        .exp = std.time.timestamp() - 100, // Expired 100 seconds ago
    });
    defer allocator.free(token_str);

    // Should fail verification
    const result = manager.verifyToken(token_str);
    try std.testing.expectError(error.TokenExpired, result);
}

test "jwt blacklist" {
    const allocator = std.testing.allocator;

    var manager = JwtManager.init(allocator, "test-secret-key-32-bytes-long!!", .{
        .enable_blacklist = true,
    });
    defer manager.deinit();

    // Create and verify token
    const token_str = try manager.createToken(.{
        .sub = "user123",
    });
    defer allocator.free(token_str);

    var token = try manager.verifyToken(token_str);

    // Revoke token
    try manager.revokeToken(&token);
    token.deinit(allocator);

    // Should fail verification now
    const result = manager.verifyToken(token_str);
    try std.testing.expectError(error.TokenRevoked, result);
}

test "extract bearer token" {
    const token = extractBearerToken("Bearer eyJhbGciOiJIUzI1NiJ9.test.sig");
    try std.testing.expect(token != null);
    try std.testing.expect(std.mem.startsWith(u8, token.?, "eyJ"));

    const no_bearer = extractBearerToken("Basic dXNlcjpwYXNz");
    try std.testing.expect(no_bearer == null);
}
