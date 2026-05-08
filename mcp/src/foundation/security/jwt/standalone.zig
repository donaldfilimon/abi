//! Standalone stateless JWT functions.

const std = @import("std");
const crypto = std.crypto;
const csprng = @import("../csprng.zig");
const types = @import("types.zig");

pub const Algorithm = types.Algorithm;
pub const Claims = types.Claims;
pub const Token = types.Token;
pub const wallClockSeconds = types.wallClockSeconds;
pub const constantTimeEqlSlice = types.constantTimeEqlSlice;

pub fn decode(allocator: std.mem.Allocator, token_str: []const u8) !Token {
    var parts = std.mem.splitScalar(u8, token_str, '.');

    const header_b64 = parts.next() orelse return error.InvalidToken;
    const payload_b64 = parts.next() orelse return error.InvalidToken;
    const signature_b64 = parts.next() orelse return error.InvalidToken;

    if (parts.next() != null) return error.InvalidToken;

    const header_json = try base64UrlDecode(allocator, header_b64);
    defer allocator.free(header_json);

    const header = parseHeaderStandalone(header_json);

    const payload_json = try base64UrlDecode(allocator, payload_b64);
    defer allocator.free(payload_json);

    const claims = try parseClaimsStandalone(allocator, payload_json);

    const signature = try base64UrlDecode(allocator, signature_b64);

    return Token{
        .raw = try allocator.dupe(u8, token_str),
        .header = header,
        .claims = claims,
        .signature = signature,
        .verified = false,
    };
}

pub fn verify(allocator: std.mem.Allocator, token_str: []const u8, secret: []const u8) !Token {
    var parts = std.mem.splitScalar(u8, token_str, '.');

    const header_b64 = parts.next() orelse return error.InvalidToken;
    const payload_b64 = parts.next() orelse return error.InvalidToken;
    const signature_b64 = parts.next() orelse return error.InvalidToken;

    if (parts.next() != null) return error.InvalidToken;

    const header_json = try base64UrlDecode(allocator, header_b64);
    defer allocator.free(header_json);
    const header = parseHeaderStandalone(header_json);

    if (header.alg == .none) return error.AlgorithmNotAllowed;
    if (header.alg == .rs256) return error.RsaSigningNotSupported;

    const signing_input_len = header_b64.len + 1 + payload_b64.len;
    const signing_input = try allocator.alloc(u8, signing_input_len);
    defer allocator.free(signing_input);
    @memcpy(signing_input[0..header_b64.len], header_b64);
    signing_input[header_b64.len] = '.';
    @memcpy(signing_input[header_b64.len + 1 ..], payload_b64);

    const sig_valid = switch (header.alg) {
        .hs256 => try verifyHmacSignature(allocator, signing_input, signature_b64, secret, crypto.auth.hmac.sha2.HmacSha256),
        .hs384 => try verifyHmacSignature(allocator, signing_input, signature_b64, secret, crypto.auth.hmac.sha2.HmacSha384),
        .hs512 => try verifyHmacSignature(allocator, signing_input, signature_b64, secret, crypto.auth.hmac.sha2.HmacSha512),
        .none, .rs256 => unreachable,
    };

    if (!sig_valid) {
        return error.InvalidSignature;
    }

    const payload_json = try base64UrlDecode(allocator, payload_b64);
    defer allocator.free(payload_json);

    const claims = try parseClaimsStandalone(allocator, payload_json);

    const signature_bytes = try base64UrlDecode(allocator, signature_b64);

    return Token{
        .raw = try allocator.dupe(u8, token_str),
        .header = header,
        .claims = claims,
        .signature = signature_bytes,
        .verified = true,
    };
}

pub fn isExpired(token: Token) bool {
    return token.claims.isExpired();
}

pub fn base64UrlEncode(allocator: std.mem.Allocator, data: []const u8) ![]const u8 {
    const encoder = std.base64.url_safe_no_pad;
    const size = encoder.Encoder.calcSize(data.len);
    const buf = try allocator.alloc(u8, size);
    _ = encoder.Encoder.encode(buf, data);
    return buf;
}

pub fn base64UrlDecode(allocator: std.mem.Allocator, data: []const u8) ![]const u8 {
    const decoder = std.base64.url_safe_no_pad;
    const size = decoder.Decoder.calcSizeForSlice(data) catch return error.InvalidBase64;
    const buf = try allocator.alloc(u8, size);
    decoder.Decoder.decode(buf, data) catch {
        allocator.free(buf);
        return error.InvalidBase64;
    };
    return buf;
}

pub fn extractBearerToken(auth_header: []const u8) ?[]const u8 {
    const prefix = "Bearer ";
    if (std.mem.startsWith(u8, auth_header, prefix)) {
        return auth_header[prefix.len..];
    }
    return null;
}

pub fn generateSecretKey(allocator: std.mem.Allocator, length: usize) ![]u8 {
    const key = try allocator.alloc(u8, length);
    try csprng.fillRandom(key);
    return key;
}

fn verifyHmacSignature(
    allocator: std.mem.Allocator,
    signing_input: []const u8,
    signature_b64: []const u8,
    secret: []const u8,
    comptime Hmac: type,
) !bool {
    var key: [Hmac.key_length]u8 = undefined;
    if (secret.len >= Hmac.key_length) {
        @memcpy(&key, secret[0..Hmac.key_length]);
    } else {
        @memset(&key, 0);
        @memcpy(key[0..secret.len], secret);
    }

    var mac: [Hmac.mac_length]u8 = undefined;
    var h = Hmac.init(&key);
    h.update(signing_input);
    h.final(&mac);

    const expected_b64 = try base64UrlEncode(allocator, &mac);
    defer allocator.free(expected_b64);

    return constantTimeEqlSlice(signature_b64, expected_b64);
}

fn parseHeaderStandalone(json: []const u8) Token.Header {
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
    return Token.Header{ .alg = alg, .typ = "JWT", .kid = null };
}

fn parseClaimsStandalone(allocator: std.mem.Allocator, json: []const u8) !Claims {
    var claims = Claims{};
    claims.sub = try extractStringClaimStandalone(allocator, json, "sub");
    claims.iss = try extractStringClaimStandalone(allocator, json, "iss");
    claims.aud = try extractStringClaimStandalone(allocator, json, "aud");
    claims.jti = try extractStringClaimStandalone(allocator, json, "jti");
    claims.exp = extractIntClaimStandalone(json, "exp");
    claims.nbf = extractIntClaimStandalone(json, "nbf");
    claims.iat = extractIntClaimStandalone(json, "iat");

    const standard_keys = [_][]const u8{ "sub", "iss", "aud", "jti", "exp", "nbf", "iat" };
    var pos: usize = 0;
    while (pos < json.len) {
        const key_start_quote = std.mem.indexOfPos(u8, json, pos, "\"") orelse break;
        const key_start = key_start_quote + 1;
        const key_end = std.mem.indexOfPos(u8, json, key_start, "\"") orelse break;
        const key = json[key_start..key_end];

        const colon_pos = std.mem.indexOfPos(u8, json, key_end + 1, ":") orelse break;
        const value_start = colon_pos + 1;

        var is_standard = false;
        for (standard_keys) |sk| {
            if (std.mem.eql(u8, key, sk)) {
                is_standard = true;
                break;
            }
        }

        if (!is_standard) {
            if (extractStringClaimStandalone(allocator, json, key) catch null) |value| {
                if (claims.custom == null) {
                    claims.custom = std.StringArrayHashMapUnmanaged([]const u8).empty;
                }
                const owned_key = try allocator.dupe(u8, key);
                try claims.custom.?.put(allocator, owned_key, value);
            }
        }

        pos = value_start;
        if (pos < json.len and json[pos] == ' ') pos += 1;
        if (pos < json.len and json[pos] == '"') {
            const v_start = pos + 1;
            const v_end = std.mem.indexOfPos(u8, json, v_start, "\"") orelse break;
            pos = v_end + 1;
        } else {
            while (pos < json.len and json[pos] != ',' and json[pos] != '}') : (pos += 1) {}
        }
    }

    return claims;
}

fn extractStringClaimStandalone(allocator: std.mem.Allocator, json: []const u8, claim: []const u8) !?[]const u8 {
    var search_buf: [64]u8 = undefined;
    const search = std.fmt.bufPrint(&search_buf, "\"{s}\":\"", .{claim}) catch return null;
    if (std.mem.indexOf(u8, json, search)) |idx| {
        const start = idx + search.len;
        var end = start;
        while (end < json.len and json[end] != '"') : (end += 1) {}
        if (end > start) {
            return try allocator.dupe(u8, json[start..end]);
        }
    }
    return null;
}

fn extractIntClaimStandalone(json: []const u8, claim: []const u8) ?i64 {
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

test {
    std.testing.refAllDecls(@This());
}
