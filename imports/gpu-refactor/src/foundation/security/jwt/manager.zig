//! JWT manager for creating and validating tokens.

const std = @import("std");
const sync = @import("../../sync.zig");
const csprng = @import("../csprng.zig");
const types = @import("types.zig");
const crypto = std.crypto;

pub const Algorithm = types.Algorithm;
pub const Claims = types.Claims;
pub const Token = types.Token;
pub const JwtConfig = types.JwtConfig;
pub const wallClockSeconds = types.wallClockSeconds;
pub const constantTimeEqlSlice = types.constantTimeEqlSlice;

pub const JwtManager = struct {
    allocator: std.mem.Allocator,
    config: JwtConfig,
    secret_key: []const u8,
    blacklist: std.StringArrayHashMapUnmanaged(i64),
    mutex: sync.Mutex,
    stats: JwtStats,

    pub const JwtStats = struct {
        tokens_created: u64 = 0,
        tokens_verified: u64 = 0,
        tokens_rejected: u64 = 0,
        tokens_expired: u64 = 0,
        tokens_blacklisted: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, secret_key: []const u8, config: JwtConfig) JwtManager {
        if (config.allow_none_algorithm) {
            std.log.warn("JWT 'none' algorithm enabled - tokens can be forged without signatures! This is a security risk.", .{});
        }

        return .{
            .allocator = allocator,
            .config = config,
            .secret_key = secret_key,
            .blacklist = std.StringArrayHashMapUnmanaged(i64).empty,
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

    pub fn createToken(self: *JwtManager, claims: Claims) ![]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = wallClockSeconds();

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
            var jti_buf: [16]u8 = undefined;
            try csprng.fillRandom(&jti_buf);
            final_claims.jti = try self.allocator.dupe(u8, &jti_buf);
        }
        defer if (claims.jti == null) {
            if (final_claims.jti) |jti| self.allocator.free(jti);
        };

        const header_json = try std.fmt.allocPrint(self.allocator, "{{\"alg\":\"{s}\",\"typ\":\"JWT\"}}", .{
            self.config.algorithm.toString(),
        });
        defer self.allocator.free(header_json);

        const payload_json = try self.claimsToJson(final_claims);
        defer self.allocator.free(payload_json);

        const header_b64 = try self.base64UrlEncode(header_json);
        defer self.allocator.free(header_b64);

        const payload_b64 = try self.base64UrlEncode(payload_json);
        defer self.allocator.free(payload_b64);

        const signing_input = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ header_b64, payload_b64 });
        defer self.allocator.free(signing_input);

        const signature = try self.sign(signing_input);
        defer self.allocator.free(signature);

        const token = try std.fmt.allocPrint(self.allocator, "{s}.{s}.{s}", .{
            header_b64,
            payload_b64,
            signature,
        });

        self.stats.tokens_created += 1;

        return token;
    }

    pub fn verifyToken(self: *JwtManager, token_str: []const u8) !Token {
        self.mutex.lock();
        defer self.mutex.unlock();

        var parts = std.mem.splitScalar(u8, token_str, '.');

        const header_b64 = parts.next() orelse return error.InvalidToken;
        const payload_b64 = parts.next() orelse return error.InvalidToken;
        const signature_b64 = parts.next() orelse return error.InvalidToken;

        if (parts.next() != null) return error.InvalidToken;

        const header_json = try self.base64UrlDecode(header_b64);
        defer self.allocator.free(header_json);

        const header = try self.parseHeader(header_json);

        if (header.alg == .none and !self.config.allow_none_algorithm) {
            self.stats.tokens_rejected += 1;
            return error.AlgorithmNotAllowed;
        }

        const signing_input = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ header_b64, payload_b64 });
        defer self.allocator.free(signing_input);

        const verified = try self.verifySignature(signing_input, signature_b64, header.alg);
        if (!verified) {
            self.stats.tokens_rejected += 1;
            return error.InvalidSignature;
        }

        const payload_json = try self.base64UrlDecode(payload_b64);
        defer self.allocator.free(payload_json);

        var claims = try self.parseClaims(payload_json);

        if (!claims.validateTime(self.config.clock_skew)) {
            if (claims.isExpired()) {
                self.stats.tokens_expired += 1;
            }
            self.stats.tokens_rejected += 1;
            claims.deinit(self.allocator);
            return error.TokenExpired;
        }

        if (self.config.enable_blacklist) {
            if (claims.jti) |jti| {
                if (self.blacklist.contains(jti)) {
                    self.stats.tokens_rejected += 1;
                    claims.deinit(self.allocator);
                    return error.TokenRevoked;
                }
            }
        }

        for (self.config.required_claims) |required| {
            if (std.mem.eql(u8, required, "sub") and claims.sub == null) {
                self.stats.tokens_rejected += 1;
                claims.deinit(self.allocator);
                return error.MissingRequiredClaim;
            }
            if (std.mem.eql(u8, required, "exp") and claims.exp == null) {
                self.stats.tokens_rejected += 1;
                claims.deinit(self.allocator);
                return error.MissingRequiredClaim;
            }
            if (std.mem.eql(u8, required, "iss") and claims.iss == null) {
                self.stats.tokens_rejected += 1;
                claims.deinit(self.allocator);
                return error.MissingRequiredClaim;
            }
        }

        self.stats.tokens_verified += 1;

        return Token{
            .raw = try self.allocator.dupe(u8, token_str),
            .header = header,
            .claims = claims,
            .signature = try self.allocator.dupe(u8, signature_b64),
            .verified = true,
        };
    }

    pub fn refreshToken(self: *JwtManager, old_token: *const Token) ![]const u8 {
        if (!old_token.verified) return error.TokenNotVerified;
        if (old_token.claims.isExpired()) return error.TokenExpired;

        var new_claims = Claims{
            .sub = if (old_token.claims.sub) |s| try self.allocator.dupe(u8, s) else null,
            .iss = self.config.issuer,
            .aud = self.config.audience,
        };

        if (old_token.claims.custom) |old_custom| {
            new_claims.custom = std.StringArrayHashMapUnmanaged([]const u8).empty;
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

    pub fn revokeToken(self: *JwtManager, token: *const Token) !void {
        if (!self.config.enable_blacklist) return error.BlacklistDisabled;

        self.mutex.lock();
        defer self.mutex.unlock();

        const jti = token.claims.jti orelse return error.NoTokenId;

        if (self.blacklist.count() >= self.config.max_blacklist_size) {
            try self.cleanupBlacklist();
        }

        const exp = token.claims.exp orelse wallClockSeconds() + 86400;
        try self.blacklist.put(self.allocator, try self.allocator.dupe(u8, jti), exp);

        self.stats.tokens_blacklisted += 1;
    }

    pub fn revokeTokenById(self: *JwtManager, jti: []const u8, exp: i64) !void {
        if (!self.config.enable_blacklist) return error.BlacklistDisabled;

        self.mutex.lock();
        defer self.mutex.unlock();

        try self.blacklist.put(self.allocator, try self.allocator.dupe(u8, jti), exp);
        self.stats.tokens_blacklisted += 1;
    }

    pub fn getStats(self: *JwtManager) JwtStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    fn sign(self: *JwtManager, input: []const u8) ![]const u8 {
        return switch (self.config.algorithm) {
            .hs256 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha256),
            .hs384 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha384),
            .hs512 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha512),
            .rs256 => return error.RsaSigningNotSupported,
            .none => try self.allocator.dupe(u8, ""),
        };
    }

    fn signHmac(self: *JwtManager, input: []const u8, comptime Hmac: type) ![]const u8 {
        var mac: [Hmac.mac_length]u8 = undefined;
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

    fn verifySignature(self: *JwtManager, input: []const u8, signature_b64: []const u8, alg: Algorithm) !bool {
        if (alg == .none) {
            return signature_b64.len == 0;
        }

        const expected_b64 = switch (alg) {
            .hs256 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha256),
            .hs384 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha384),
            .hs512 => try self.signHmac(input, crypto.auth.hmac.sha2.HmacSha512),
            .rs256 => return error.RsaSigningNotSupported,
            .none => return error.AlgorithmNone,
        };
        defer self.allocator.free(expected_b64);

        return constantTimeEqlSlice(signature_b64, expected_b64);
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
        var alg: Algorithm = .hs256;

        if (std.mem.indexOf(u8, json, "\"alg\"")) |idx| {
            const start = idx + 6;
            if (start < json.len) {
                var i = start;
                while (i < json.len and json[i] != '"') : (i += 1) {}
                if (i < json.len) {
                    i += 1;
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
        var search_buf: [64]u8 = undefined;
        const search = std.fmt.bufPrint(&search_buf, "\"{s}\":\"", .{claim}) catch return null;

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
        var search_buf: [64]u8 = undefined;
        const search = std.fmt.bufPrint(&search_buf, "\"{s}\":", .{claim}) catch return null;

        if (std.mem.indexOf(u8, json, search)) |idx| {
            const start = idx + search.len;
            var end = start;
            while (end < json.len and (json[end] == ' ' or json[end] == '\t')) : (end += 1) {}
            const num_start = end;
            if (end < json.len and json[end] == '-') end += 1;
            while (end < json.len and (json[end] >= '0' and json[end] <= '9')) : (end += 1) {}
            if (end > num_start) {
                return std.fmt.parseInt(i64, json[num_start..end], 10) catch null;
            }
        }
        return null;
    }

    fn claimsToJson(self: *JwtManager, claims: Claims) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(self.allocator);

        try buffer.append(self.allocator, '{');

        var first = true;

        if (claims.sub) |sub| {
            if (!first) try buffer.append(self.allocator, ',');
            try appendFmt(&buffer, self.allocator, "\"sub\":\"{s}\"", .{sub});
            first = false;
        }
        if (claims.iss) |iss| {
            if (!first) try buffer.append(self.allocator, ',');
            try appendFmt(&buffer, self.allocator, "\"iss\":\"{s}\"", .{iss});
            first = false;
        }
        if (claims.aud) |aud| {
            if (!first) try buffer.append(self.allocator, ',');
            try appendFmt(&buffer, self.allocator, "\"aud\":\"{s}\"", .{aud});
            first = false;
        }
        if (claims.jti) |jti| {
            if (!first) try buffer.append(self.allocator, ',');
            var hex_buf: [64]u8 = undefined;
            const hex_chars = "0123456789abcdef";
            const hlen = @min(jti.len, 32);
            for (jti[0..hlen], 0..) |b, idx| {
                hex_buf[idx * 2] = hex_chars[b >> 4];
                hex_buf[idx * 2 + 1] = hex_chars[b & 0x0f];
            }
            try appendFmt(&buffer, self.allocator, "\"jti\":\"{s}\"", .{hex_buf[0 .. hlen * 2]});
            first = false;
        }
        if (claims.exp) |exp| {
            if (!first) try buffer.append(self.allocator, ',');
            try appendFmt(&buffer, self.allocator, "\"exp\":{d}", .{exp});
            first = false;
        }
        if (claims.nbf) |nbf| {
            if (!first) try buffer.append(self.allocator, ',');
            try appendFmt(&buffer, self.allocator, "\"nbf\":{d}", .{nbf});
            first = false;
        }
        if (claims.iat) |iat| {
            if (!first) try buffer.append(self.allocator, ',');
            try appendFmt(&buffer, self.allocator, "\"iat\":{d}", .{iat});
            first = false;
        }

        if (claims.custom) |custom| {
            var it = custom.iterator();
            while (it.next()) |entry| {
                if (!first) try buffer.append(self.allocator, ',');
                try appendFmt(&buffer, self.allocator, "\"{s}\":\"{s}\"", .{
                    entry.key_ptr.*,
                    entry.value_ptr.*,
                });
                first = false;
            }
        }

        try buffer.append(self.allocator, '}');

        return buffer.toOwnedSlice(self.allocator);
    }

    fn appendFmt(
        buffer: *std.ArrayListUnmanaged(u8),
        allocator: std.mem.Allocator,
        comptime fmt: []const u8,
        args: anytype,
    ) !void {
        const tmp = try std.fmt.allocPrint(allocator, fmt, args);
        defer allocator.free(tmp);
        try buffer.appendSlice(allocator, tmp);
    }

    fn cleanupBlacklist(self: *JwtManager) !void {
        const now = wallClockSeconds();
        const keys = self.blacklist.keys();
        const values = self.blacklist.values();
        var i: usize = keys.len;
        while (i > 0) {
            i -= 1;
            if (values[i] < now) {
                const key = keys[i];
                self.blacklist.swapRemoveAt(i);
                self.allocator.free(key);
            }
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}
