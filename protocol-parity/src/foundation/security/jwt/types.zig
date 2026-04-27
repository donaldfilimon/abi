//! JWT types, configuration, and utility functions.

const std = @import("std");
const time = @import("../../time.zig");

pub fn wallClockSeconds() i64 {
    if (@hasDecl(std.posix, "system")) {
        var ts: std.posix.timespec = undefined;
        if (std.posix.errno(std.posix.system.clock_gettime(.REALTIME, &ts)) == .SUCCESS) {
            return @intCast(@max(ts.sec, 0));
        }
    }
    return time.unixSeconds();
}

pub fn constantTimeEqlSlice(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    if (a.len == 0) return true;
    var diff: u8 = 0;
    for (a, b) |x, y| {
        diff |= x ^ y;
    }
    return diff == 0;
}

pub const Algorithm = enum {
    hs256,
    hs384,
    hs512,
    rs256,
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

pub const Claims = struct {
    iss: ?[]const u8 = null,
    sub: ?[]const u8 = null,
    aud: ?[]const u8 = null,
    exp: ?i64 = null,
    nbf: ?i64 = null,
    iat: ?i64 = null,
    jti: ?[]const u8 = null,
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

    pub fn isExpired(self: Claims) bool {
        if (self.exp) |exp| {
            return wallClockSeconds() > exp;
        }
        return false;
    }

    pub fn isValidYet(self: Claims) bool {
        if (self.nbf) |nbf| {
            return wallClockSeconds() >= nbf;
        }
        return true;
    }

    pub fn validateTime(self: Claims, clock_skew: i64) bool {
        const now = wallClockSeconds();
        if (self.exp) |exp| {
            if (now > exp + clock_skew) return false;
        }
        if (self.nbf) |nbf| {
            if (now < nbf - clock_skew) return false;
        }
        if (self.iat) |iat| {
            if (iat > now + clock_skew) return false;
        }
        return true;
    }
};

pub const Token = struct {
    raw: []const u8,
    header: Header,
    claims: Claims,
    signature: []const u8,
    verified: bool,

    pub const Header = struct {
        alg: Algorithm,
        typ: []const u8 = "JWT",
        kid: ?[]const u8 = null,
    };

    pub fn deinit(self: *Token, allocator: std.mem.Allocator) void {
        allocator.free(self.raw);
        allocator.free(self.signature);
        if (self.header.kid) |kid| allocator.free(kid);
        self.claims.deinit(allocator);
    }
};

pub const JwtConfig = struct {
    algorithm: Algorithm = .hs256,
    token_lifetime: i64 = 3600,
    refresh_lifetime: i64 = 86400 * 7,
    issuer: ?[]const u8 = null,
    audience: ?[]const u8 = null,
    clock_skew: i64 = 60,
    enable_blacklist: bool = true,
    max_blacklist_size: usize = 10000,
    allow_none_algorithm: bool = false,
    required_claims: []const []const u8 = &.{ "sub", "exp" },
};

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
    RsaSigningNotSupported,
    AlgorithmNone,
    OutOfMemory,
};

pub const JwtHeader = Token.Header;
pub const JwtPayload = Claims;
pub const JwtToken = Token;

pub fn supportedAlgorithms() []const Algorithm {
    return &[_]Algorithm{ .hs256, .hs384, .hs512 };
}

pub fn getRsaUnavailableReason() []const u8 {
    return "RSA signing requires external crypto library (mbedtls/openssl); only HMAC algorithms available";
}

pub fn validateExpiry(claims: anytype, now: i64) bool {
    const T = @TypeOf(claims);
    if (@hasField(T, "exp")) {
        if (claims.exp) |exp| {
            return exp > now;
        }
    }
    return true;
}

test {
    std.testing.refAllDecls(@This());
}
