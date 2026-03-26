//! Parsing helpers for encoded password hashes.

const std = @import("std");
const types = @import("types.zig");

pub fn parseArgon2Encoded(encoded: []const u8) ?types.ParsedArgon2 {
    if (!std.mem.startsWith(u8, encoded, "$argon2id$")) return null;

    var parts = std.mem.splitScalar(u8, encoded, '$');

    _ = parts.next() orelse return null;
    _ = parts.next() orelse return null;
    _ = parts.next() orelse return null;

    const params_str = parts.next() orelse return null;
    var memory_cost: u32 = 0;
    var time_cost: u32 = 0;
    var parallelism: u8 = 0;

    var param_parts = std.mem.splitScalar(u8, params_str, ',');
    while (param_parts.next()) |param| {
        if (std.mem.startsWith(u8, param, "m=")) {
            memory_cost = std.fmt.parseInt(u32, param[2..], 10) catch return null;
        } else if (std.mem.startsWith(u8, param, "t=")) {
            time_cost = std.fmt.parseInt(u32, param[2..], 10) catch return null;
        } else if (std.mem.startsWith(u8, param, "p=")) {
            parallelism = std.fmt.parseInt(u8, param[2..], 10) catch return null;
        }
    }

    const salt_b64 = parts.next() orelse return null;
    const hash_b64 = parts.next() orelse return null;

    const decoder = std.base64.standard.Decoder;
    var salt_buf: [64]u8 = undefined;
    var hash_buf: [64]u8 = undefined;

    const salt_len = decoder.calcSizeForSlice(salt_b64) catch return null;
    const hash_len = decoder.calcSizeForSlice(hash_b64) catch return null;

    if (salt_len > salt_buf.len or hash_len > hash_buf.len) return null;

    decoder.decode(salt_buf[0..salt_len], salt_b64) catch return null;
    decoder.decode(hash_buf[0..hash_len], hash_b64) catch return null;

    return types.ParsedArgon2{
        .salt = salt_buf[0..salt_len],
        .hash = hash_buf[0..hash_len],
        .memory_cost = memory_cost,
        .time_cost = time_cost,
        .parallelism = parallelism,
    };
}

pub fn parsePbkdf2Encoded(encoded: []const u8) ?types.ParsedPbkdf2 {
    if (!std.mem.startsWith(u8, encoded, "$pbkdf2-")) return null;

    var parts = std.mem.splitScalar(u8, encoded, '$');

    _ = parts.next() orelse return null;
    _ = parts.next() orelse return null;

    const iter_str = parts.next() orelse return null;
    var iterations: u32 = 0;
    if (std.mem.startsWith(u8, iter_str, "i=")) {
        iterations = std.fmt.parseInt(u32, iter_str[2..], 10) catch return null;
    } else {
        return null;
    }

    const salt_b64 = parts.next() orelse return null;
    const hash_b64 = parts.next() orelse return null;

    const decoder = std.base64.standard.Decoder;
    var salt_buf: [64]u8 = undefined;
    var hash_buf: [64]u8 = undefined;

    const salt_len = decoder.calcSizeForSlice(salt_b64) catch return null;
    const hash_len = decoder.calcSizeForSlice(hash_b64) catch return null;

    if (salt_len > salt_buf.len or hash_len > hash_buf.len) return null;

    decoder.decode(salt_buf[0..salt_len], salt_b64) catch return null;
    decoder.decode(hash_buf[0..hash_len], hash_b64) catch return null;

    return types.ParsedPbkdf2{
        .salt = salt_buf[0..salt_len],
        .hash = hash_buf[0..hash_len],
        .iterations = iterations,
    };
}

pub fn parseScryptEncoded(encoded: []const u8) ?types.ParsedScrypt {
    if (!std.mem.startsWith(u8, encoded, "$scrypt$")) return null;

    var parts = std.mem.splitScalar(u8, encoded, '$');

    _ = parts.next() orelse return null;
    _ = parts.next() orelse return null;

    const params_str = parts.next() orelse return null;
    var log_n: u6 = 0;
    var r: u30 = 0;
    var p: u30 = 0;

    var param_parts = std.mem.splitScalar(u8, params_str, ',');
    while (param_parts.next()) |param| {
        if (std.mem.startsWith(u8, param, "ln=")) {
            log_n = std.fmt.parseInt(u6, param[3..], 10) catch return null;
        } else if (std.mem.startsWith(u8, param, "r=")) {
            r = std.fmt.parseInt(u30, param[2..], 10) catch return null;
        } else if (std.mem.startsWith(u8, param, "p=")) {
            p = std.fmt.parseInt(u30, param[2..], 10) catch return null;
        }
    }

    const salt_b64 = parts.next() orelse return null;
    const hash_b64 = parts.next() orelse return null;

    const decoder = std.base64.standard.Decoder;
    var salt_buf: [64]u8 = undefined;
    var hash_buf: [64]u8 = undefined;

    const salt_len = decoder.calcSizeForSlice(salt_b64) catch return null;
    const hash_len = decoder.calcSizeForSlice(hash_b64) catch return null;

    if (salt_len > salt_buf.len or hash_len > hash_buf.len) return null;

    decoder.decode(salt_buf[0..salt_len], salt_b64) catch return null;
    decoder.decode(hash_buf[0..hash_len], hash_b64) catch return null;

    return types.ParsedScrypt{
        .salt = salt_buf[0..salt_len],
        .hash = hash_buf[0..hash_len],
        .log_n = log_n,
        .r = r,
        .p = p,
    };
}

pub fn parseBlake3Encoded(encoded: []const u8) ?types.ParsedBlake3 {
    if (!std.mem.startsWith(u8, encoded, "$blake3$")) return null;

    var parts = std.mem.splitScalar(u8, encoded, '$');

    _ = parts.next() orelse return null;
    _ = parts.next() orelse return null;

    const salt_b64 = parts.next() orelse return null;
    const hash_b64 = parts.next() orelse return null;

    const decoder = std.base64.standard.Decoder;
    var salt_buf: [32]u8 = undefined;
    var hash_buf: [32]u8 = undefined;

    const salt_len = decoder.calcSizeForSlice(salt_b64) catch return null;
    const hash_len = decoder.calcSizeForSlice(hash_b64) catch return null;

    if (salt_len > salt_buf.len or hash_len > hash_buf.len) return null;

    decoder.decode(salt_buf[0..salt_len], salt_b64) catch return null;
    decoder.decode(hash_buf[0..hash_len], hash_b64) catch return null;

    return types.ParsedBlake3{
        .salt = salt_buf[0..salt_len],
        .hash = hash_buf[0..hash_len],
    };
}

pub fn detectAlgorithm(encoded: []const u8) ?types.Algorithm {
    if (std.mem.startsWith(u8, encoded, "$argon2id$")) return .argon2id;
    if (std.mem.startsWith(u8, encoded, "$pbkdf2-sha256$")) return .pbkdf2_sha256;
    if (std.mem.startsWith(u8, encoded, "$pbkdf2-sha512$")) return .pbkdf2_sha512;
    if (std.mem.startsWith(u8, encoded, "$scrypt$")) return .scrypt;
    if (std.mem.startsWith(u8, encoded, "$blake3$")) return .blake3_kdf;
    return null;
}
